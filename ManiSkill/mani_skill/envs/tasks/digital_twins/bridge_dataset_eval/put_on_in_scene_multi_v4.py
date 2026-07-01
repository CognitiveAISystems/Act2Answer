"""Act2Answer embodied evaluation environment (``Act2AnswerV4-v1``).

Act2Answer turns a VLM knowledge question into a short tabletop episode: the scene shows two
image tiles (left / right) that are the candidate answers, and a small cube pre-grasped by the
WidowX gripper. The agent reads a natural-language instruction (e.g. "put cube on <answer>") and
answers *by action* -- it must place the cube on the tile it believes is correct.

Assets live under ``ManiSkill/mani_skill/assets/carrot/<assets>/`` and are authored per task set:
``pairs.json`` (one entry per question: ``left``/``right`` tile names, ``question``, ``answer``),
``model_db.json`` (per-tile metadata) and ``shapes/<tile>/{textured.glb,collision.obj}``.

Scoring (see ``evaluate``): ``is_answered`` = cube ended on *either* tile; ``is_success`` = cube on
the *correct* tile. Each question is run in both original and left/right-swapped layouts and the two
success rates are averaged to remove positional bias (``do_swap``). This is the only environment used
by the Act2Answer eval harness (``simpler_env``).
"""
import os
import re
from pathlib import Path
import cv2
import numpy as np
import sapien
import torch
import torch.nn.functional as F
from sapien.physx import PhysxMaterial
from transforms3d.euler import euler2quat
import sapien.physx as physx

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import BRIDGE_DATASET_ASSET_PATH, \
    WidowX250SBridgeDatasetFlatTable
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, SimConfig
from mani_skill.utils.registration import register_env

from typing import Dict, List, Optional, Sequence, Union


CARROT_DATASET_DIR = Path(__file__).parent / ".." / ".." / ".." / ".." / "assets" / "carrot"


TABLE_Z = 0.88           # Table top height used in this file


CUBE_HALF_SIZE = 0.015 #0.015 #0.015  # ~3.0cm cubes


GRASP_STEPS_REQ = 5     # number of consecutive steps to consider "grasped"


LIFT_Z_THRESH = TABLE_Z + 0.03  # lifted if 3+ cm above the table


GRAY = np.array([200, 200, 200, 255], dtype=np.float32) / 255.0


class _PickCubeBase(BaseEnv):
    SUPPORTED_OBS_MODES = ["rgb+segmentation"]
    SUPPORTED_REWARD_MODES = ["none"]

    rgb_camera_name: str = "3rd_view_camera"

    def __init__(self, initial_qpos = None, **kwargs):
        # keep consistent robot pose and qpos with other Bridge-table envs here
        if initial_qpos is None:
            self.initial_qpos = np.array([
                -0.01840777, 0.0398835, 0.22242722,
                -0.00460194, 1.36524296, 0.00153398,
                0.037, 0.037,
            ])
        self.initial_robot_pos = sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])
        self.safe_robot_pos = sapien.Pose([0.147, 0.028, 1.870], q=[0, 0, 0, 1])

        # book-keeping
        self.consecutive_grasp = None
        self.episode_stats = {}
        self._target_key = None  # which cube is target (color or direction key)
        self.cubes: Dict[str, Actor] = {}

        super().__init__(robot_uids=WidowX250SBridgeDatasetFlatTable, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5, spacing=20)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.127, 0.060, 0.85], q=[0, 0, 0, 1]))

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 0, -1], [2.2, 2.2, 2.2], shadow=False, shadow_scale=5, shadow_map_size=2048)
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _load_scene(self, options: dict):
        # Make robot shiny like others in this file
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(self.agent.robot._objs[i], specular=0.9, roughness=0.3)

        # Background stage (Bridge table)
        builder = self.scene.create_actor_builder()
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])
        scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v1.glb")
        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)
        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        # build the cubes in-place in child classes
        self._build_cubes()

    def _build_cubes(self):
        raise NotImplementedError

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # child classes decide which target to pick (color or direction)
        raise NotImplementedError

    def _place_cube(self, actor: Actor, xyz: np.ndarray):
        # batched set_pose
        b = self.num_envs
        p = torch.tensor(xyz, device=self.device, dtype=torch.float32).reshape(1, 3).repeat(b, 1)
        q = torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float32).reshape(1, 4).repeat(b, 1)
        actor.set_pose(Pose.create_from_pq(p=p, q=q))

    def _settle(self, t=0.5):
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
        n = int(self.sim_freq * t / self.control_freq)
        for _ in range(n):
            self.scene.step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Let subclass set target and positions
        self._initialize_episode_pre(env_idx, options)

        # Set robot to safe pose while placing
        self.agent.robot.set_pose(self.safe_robot_pos)

        # # Place cubes (implemented by child via self._scene_positions dict)
        # for key, (actor, pos_xy) in self._scene_positions.items():
        #     self._place_cube(actor, np.array([pos_xy[0], pos_xy[1], TABLE_Z + CUBE_HALF_SIZE])) # TODO: make Z coordinate optimal, was " + CUBE_HALF_SIZE"


        # Place cubes (implemented by child via self._scene_positions dict)
        for key, (actor, pos) in self._scene_positions.items():
            if len(pos) == 2:
                self._place_cube(actor, np.array([pos[0], pos[1], TABLE_Z + CUBE_HALF_SIZE])) # TODO: make Z coordinate optimal, was " + CUBE_HALF_SIZE"
            elif len(pos) == 3: 
                self._place_cube(actor, np.array([pos[0], pos[1], pos[2]])) # TODO: make Z coordinate optimal, was " + CUBE_HALF_SIZE"


        # self._settle(0.5)

        # Move robot to nominal start and reset joints
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # init stats
        b = self.num_envs
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            target_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            wrong_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            target_height=torch.zeros((b,), dtype=torch.float32, device=self.device),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # Track whether we have ever achieved a long enough consecutive grasp
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
        )

    def evaluate(self):
        # compute success based on grasp or lift (per environment)
        b = self.num_envs
        
        # Initialize per-environment results
        target_z = torch.zeros(b, device=self.device, dtype=torch.float32)
        grasp_target = torch.zeros(b, device=self.device, dtype=torch.bool)
        wrong = torch.zeros(b, device=self.device, dtype=torch.bool)
        
        # Check each environment's target
        for i in range(b):
            target_key = self._target_keys[i]
            target_actor = self.cubes[target_key]
            
            # z height of target (get i-th env's pose)
            target_p = target_actor.pose.p[i]  # [3]
            target_z[i] = target_p[2]
            
            # grasp check for this environment
            grasp_target[i] = self.agent.is_grasping(target_actor)[i]
            
            # wrong grasp if grasping any other cube
            for k, a in self.cubes.items():
                if k == target_key:
                    continue
                wrong[i] = wrong[i] | self.agent.is_grasping(a)[i]
        
        self.episode_stats["target_height"] = target_z
        
        # Update consecutive grasp counter (per-step)
        self.consecutive_grasp += grasp_target
        self.consecutive_grasp[~grasp_target] = 0
        target_grasp_ok = self.consecutive_grasp >= GRASP_STEPS_REQ

        # Per-episode flag: has the target ever been grasped for long enough?
        consec_flag = target_grasp_ok
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] | consec_flag
        )

        self.episode_stats["target_grasped"] = self.episode_stats["target_grasped"] | target_grasp_ok
        self.episode_stats["wrong_grasp"] = wrong
        self.episode_stats["is_src_obj_grasped"] = self.episode_stats["is_src_obj_grasped"] | grasp_target

        success = target_grasp_ok | (target_z > LIFT_Z_THRESH)
        return dict(**self.episode_stats, success=success)

    def is_final_subtask(self):
        return True

    # camera for human render (same as above)
    @property
    def _default_human_render_camera_configs(self):
        sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose([0.00, -0.16, 0.336], [0.909182, -0.0819809, 0.347277, 0.214629]),
            width=512,
            height=512,
            intrinsic=np.array([[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]),
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["base_link"],
        )

    def get_obs(self, info: dict = None):
        # default observation is fine (rgb+seg supported)
        return super().get_obs(info)


def _build_by_type(
    builder: ActorBuilder,
    name,
    body_type,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    if scene_idxs is not None:
        builder.set_scene_idxs(scene_idxs)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    else:
        raise ValueError(f"Unknown body type {body_type}")
    return actor


def build_colorful_cube_my(
    scene: ManiSkillScene,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()

    if add_collision:
        builder._mass = 0.000001
        cube_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=50, dynamic_friction=30, restitution=0
        )
        builder.add_box_collision(
            half_size=[half_size] * 3,
            material=cube_material,
        )
    builder.add_box_visual(
        half_size=[half_size] * 3,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def _load_overlay_images(rgb_overlay_paths: Optional[Dict[str, str]]):
    if rgb_overlay_paths is None:
        return None
    overlay_imgs = {}
    for cam, path in rgb_overlay_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"rgb_overlay_path {path} not found for camera '{cam}'. "
                f"Provide correct paths to real background images."
            )
        overlay_imgs[cam] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)  # (H,W,3) uint8
    return overlay_imgs


@register_env(
    "Act2AnswerV4-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class Act2AnswerV4(_PickCubeBase):
    """Act2Answer environment: two answer tiles (left/right) + one pre-grasped cube.

    The agent answers a knowledge question by placing the cube on the tile it believes is correct.
    Tiles are textured from the task set in ``carrot/<assets>/`` (``pairs.json`` gives the
    left/right tile names, the instruction and the ground-truth ``answer``). ``evaluate`` reports
    ``is_answered`` (cube on any tile) and ``is_success`` (cube on the correct tile); ``do_swap``
    mirrors the layout left<->right so scores can be averaged over both sides.

    Key constructor knobs: ``assets`` (task-set folder), ``ids`` (which questions), ``do_swap``,
    ``cube_pose``/``initial_qpos`` (fixed start so the cube begins in the gripper), and the
    SIMPLER-style visual-matching hooks below (``MODEL_JSON``, ``rgb_overlay_*``).
    """

    # ------------------------
    # (A) Texture matching hook (SIMPLER-style "MODEL_JSON")
    # ------------------------
    # Baked-texture evals can swap model DB via MODEL_JSON. Here the default is
    # "model_db.json", but you can set e.g. "model_db_baked_tex.json".
    MODEL_JSON: str = "model_db.json"

    # ------------------------
    # (B) Greenscreen overlay toggles (SIMPLER-style)
    # ------------------------
    rgb_overlay_paths: Optional[Dict[str, str]] = None
    """
    dict: camera_name -> path to real background image
    Example:
        rgb_overlay_paths = {
            "base_camera": "/path/to/real_bg_base.png",
            "hand_camera": "/path/to/real_bg_wrist.png",
        }
    """
    rgb_overlay_mode: str = "background"
    """
    same meaning as SIMPLER code:
      - "background": overlay only background, keep robot + target objects from sim
      - "background+object": overlay everything except robot links
      - "debug": 50/50 blend for visualization
    """
    rgb_always_overlay_objects: List[str] = []
    """
    actor/link names that should be considered background (always covered by overlay).
    Use this for non-interactive scene stuff that you want to "look real".
    """

    def __init__(
        self,
        assets: str,
        ids: Optional[Sequence[int]] = None,
        do_swap: str = "False",
        cube_pose: Optional[Sequence[float]] = None,
        initial_qpos: Optional[Sequence[float]] = None,
        # --- NEW knobs for visual matching ---
        model_json: Optional[str] = None,
        rgb_overlay_paths: Optional[Dict[str, str]] = None,
        rgb_overlay_mode: Optional[str] = None,
        rgb_always_overlay_objects: Optional[List[str]] = None,
        **kwargs,
    ):
        self._kin_hold_steps = 20
        self._kin_hold_left = 0

        self.initial_qpos = initial_qpos
        self.dataset_dir = CARROT_DATASET_DIR / assets
        self.do_swap = bool(do_swap)

        # ids mapping
        self._pair_ids_raw: Optional[List[int]] = (
            [int(i) for i in ids] if ids is not None else None
        )
        self._pair_ids_config: Optional[List[int]] = None

        self.cube_pose = cube_pose

        # ------------------------
        # NEW: choose model db json (texture matching entry point)
        # ------------------------
        if model_json is not None:
            self.MODEL_JSON = str(model_json)

        # ------------------------
        # NEW: greenscreen config (visual matching entry point)
        # ------------------------
        if rgb_overlay_paths is not None:
            self.rgb_overlay_paths = dict(rgb_overlay_paths)
        if rgb_overlay_mode is not None:
            self.rgb_overlay_mode = str(rgb_overlay_mode)
        if rgb_always_overlay_objects is not None:
            self.rgb_always_overlay_objects = list(rgb_always_overlay_objects)

        # Load overlay images (CPU, resized later per camera)
        self._rgb_overlay_images = _load_overlay_images(self.rgb_overlay_paths)

        super().__init__(initial_qpos=self.initial_qpos, **kwargs)


    def _get_tcp_p(self) -> torch.Tensor:
        """
        Best-effort end-effector / TCP position accessor.
        Returns: [B,3] torch tensor on self.device (or render device).
        Uses only attributes that may exist; falls back safely.
        """
        if hasattr(self.agent, "tcp") and hasattr(self.agent.tcp, "pose") and hasattr(self.agent.tcp.pose, "p"):
            return self.agent.tcp.pose.p

        # 2) Some agents expose ee_link
        if hasattr(self.agent, "ee_link") and hasattr(self.agent.ee_link, "pose") and hasattr(self.agent.ee_link.pose, "p"):
            return self.agent.ee_link.pose.p

        if hasattr(self.agent, "robot") and hasattr(self.agent.robot, "get_links"):
            links = self.agent.robot.get_links()
            if len(links) > 0 and hasattr(links[-1], "pose") and hasattr(links[-1].pose, "p"):
                return links[-1].pose.p

        raise AttributeError("Could not find TCP/EE position (agent.tcp / agent.ee_link / robot.get_links fallback all failed).")


    def _steps_for_seconds(self, seconds: float, default_hz: float = 30.0) -> int:
        """
        Convert seconds -> steps using environment control frequency or sim dt if available.
        Falls back to default_hz if neither exists.
        """
        # Many ManiSkill envs expose control_freq (Hz)
        if hasattr(self, "control_freq") and self.control_freq is not None:
            try:
                hz = float(self.control_freq)
                return max(1, int(round(seconds * hz)))
            except Exception:
                pass

        # Sometimes sim_dt exists (seconds per sim step)
        if hasattr(self, "sim_dt") and self.sim_dt is not None:
            try:
                dt = float(self.sim_dt)
                if dt > 0:
                    return max(1, int(round(seconds / dt)))
            except Exception:
                pass

        return max(1, int(round(seconds * float(default_hz))))
    # ---------------------------------------------------------------------
    # Greenscreen plumbing: actor/link ids cache + overlay image resizing
    # ---------------------------------------------------------------------
    def _after_reconfigure(self, options: dict):
        super()._after_reconfigure(options)

        # If overlay is disabled, nothing to do
        if self.rgb_overlay_paths is None or self._rgb_overlay_images is None:
            return

        # actor ids to treat as "foreground objects" (keep sim pixels)
        # We exclude: ground, goal_site, arena, and also anything in rgb_always_overlay_objects
        target_object_actor_ids = [
            x._objs[0].per_scene_id
            for x in self.scene.actors.values()
            if x.name
            not in (["ground", "goal_site", "", "arena"] + self.rgb_always_overlay_objects)
        ]
        self.target_object_actor_ids = torch.tensor(
            target_object_actor_ids, dtype=torch.int16, device=self.device
        )

        # robot link ids (keep sim pixels)
        robot_links = self.agent.robot.get_links()
        self.robot_link_ids = torch.tensor(
            [lnk._objs[0].entity.per_scene_id for lnk in robot_links],
            dtype=torch.int16,
            device=self.device,
        )

        # Resize overlay images to match camera resolution and move to device
        for camera_name in self.rgb_overlay_paths.keys():
            sensor = self._sensor_configs[camera_name]
            # only handle CameraConfig-like sensors (same assumption as SIMPLER code)
            if not hasattr(sensor, "width") or not hasattr(sensor, "height"):
                continue

            # if already tensor, assume it's ready
            if isinstance(self._rgb_overlay_images[camera_name], torch.Tensor):
                continue

            rgb_overlay_img = cv2.resize(
                self._rgb_overlay_images[camera_name], (sensor.width, sensor.height)
            )
            self._rgb_overlay_images[camera_name] = common.to_tensor(
                rgb_overlay_img, device=self.device
            )

    def _green_screen_rgb(self, rgb: torch.Tensor, segmentation: torch.Tensor, overlay_img: torch.Tensor):
        """
        rgb:          [B,H,W,3] float in [0,1] (ManiSkill convention)
        segmentation: [B,H,W,K] or [B,H,W,?] with actor id in channel 0 (ManiSkill convention here)
        overlay_img:  [H,W,3] or [1,H,W,3] tensor (we broadcast)
        """
        actor_seg = segmentation[..., 0]
        mask = torch.ones_like(actor_seg, device=actor_seg.device)

        # align id tensors to seg device (CPU vs GPU render backend mismatch)
        if actor_seg.device != self.robot_link_ids.device:
            self.robot_link_ids = self.robot_link_ids.to(actor_seg.device)
            self.target_object_actor_ids = self.target_object_actor_ids.to(actor_seg.device)

        mode = self.rgb_overlay_mode

        if ("background" in mode) or ("debug" in mode):
            if ("object" not in mode) or ("debug" in mode):
                # overlay only background: keep robot links + target objects from sim
                keep_ids = torch.concatenate([self.robot_link_ids, self.target_object_actor_ids])
                mask[torch.isin(actor_seg, keep_ids)] = 0
            else:
                # overlay everything except robot links
                mask[torch.isin(actor_seg, self.robot_link_ids)] = 0
        else:
            raise NotImplementedError(f"rgb_overlay_mode='{mode}' not supported")

        mask = mask[..., None]  # [B,H,W,1]

        # Broadcast overlay to batch
        if overlay_img.dim() == 3:
            overlay_img = overlay_img.unsqueeze(0)  # [1,H,W,3]

        if "debug" not in mode:
            rgb = rgb * (1 - mask) + overlay_img * mask
        else:
            rgb = rgb * 0.5 + overlay_img * 0.5
        return rgb

    def get_obs(self, info: dict = None):
        obs = super().get_obs(info)

        # Apply greenscreen overlay only if:
        #  - overlay assets exist
        #  - obs provides rgb + segmentation
        if self.rgb_overlay_paths is None or self._rgb_overlay_images is None:
            return obs

        # This matches SIMPLER behavior: requires segmentation
        # Sensor layout: obs["sensor_data"][camera_name]["rgb"/"segmentation"]
        if "sensor_data" not in obs:
            return obs

        for camera_name in self._rgb_overlay_images.keys():
            if camera_name not in obs["sensor_data"]:
                continue
            cam_data = obs["sensor_data"][camera_name]
            if ("rgb" not in cam_data) or ("segmentation" not in cam_data):
                # overlay requires segmentation in obs_mode
                continue

            rgb = cam_data["rgb"]
            seg = cam_data["segmentation"]

            overlay_img = self._rgb_overlay_images[camera_name]
            if overlay_img.device != rgb.device:
                overlay_img = overlay_img.to(rgb.device)
                self._rgb_overlay_images[camera_name] = overlay_img

            cam_data["rgb"] = self._green_screen_rgb(rgb, seg, overlay_img)

        return obs

    # ---------------------------------------------------------------------
    # Your original logic below (only small change: model_db_path uses MODEL_JSON)
    # ---------------------------------------------------------------------
    def _build_actor_helper(self, name: str, path: Path, density: float, scale: float, pose: Pose):
        physical_material = PhysxMaterial(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        builder = self.scene.create_actor_builder()
        collision_file = str(path / "collision.obj")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )
        visual_file = str(path / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(path / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(path / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)
        builder.initial_pose = pose
        actor = builder.build(name=name)
        return actor

    def _build_cubes(self):
        self.cubes = {}
        a = build_colorful_cube_my(
            self.scene,
            half_size=CUBE_HALF_SIZE,
            color=GRAY,
            name="cube_white",
            body_type="dynamic",
            add_collision=True,
        )
        self.cubes["white"] = a
        self._white_key = "white"

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        pairs_path = self.dataset_dir / "pairs.json"
        # ------------------------
        # CHANGED: model_db_path uses MODEL_JSON (SIMPLER baked-tex style)
        # ------------------------
        model_db_path = self.dataset_dir / self.MODEL_JSON

        assert pairs_path.exists(), f"pairs.json not found at {pairs_path}"
        assert model_db_path.exists(), (
            f"{self.MODEL_JSON} not found at {model_db_path}. "
            f"Provide correct MODEL_JSON or generate the baked/matched model db."
        )

        self.pairs_meta = io_utils.load_json(pairs_path)

        index_to_pos: Dict[int, int] = {}
        for pos, p in enumerate(self.pairs_meta):
            idx = int(p.get("index", pos))
            index_to_pos[idx] = pos

        if self._pair_ids_raw is not None:
            self._pair_ids_config = []
            for idx in self._pair_ids_raw:
                assert idx in index_to_pos, f"Pair id {idx} from ids not found in pairs.json"
                self._pair_ids_config.append(index_to_pos[idx])

        # board model db (now can be baked/matched)
        self.model_db_board: dict[str, dict] = io_utils.load_json(model_db_path)

        if self._pair_ids_config is not None:
            used_board_names = set()
            for pos in self._pair_ids_config:
                p = self.pairs_meta[pos]
                used_board_names.add(str(p["left"]))
                used_board_names.add(str(p["right"]))
            self.model_db_board = {k: v for k, v in self.model_db_board.items() if k in used_board_names}

        self.board_names = list(self.model_db_board.keys())

        self.objs_board: dict[str, Actor] = {}
        self.model_bbox_sizes: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.board_names):
            model_path = self.dataset_dir / "shapes" / name
            info = self.model_db_board[name]
            density = info.get("density", 1000)
            scale_list = info.get("scales", [1.0])
            scale = scale_list[0] if isinstance(scale_list, (list, tuple)) and len(scale_list) > 0 else 1.0
            pose = Pose.create_from_pq(torch.tensor([2.0, 0.3 * idx, 1.5]))  # off-screen
            self.objs_board[name] = self._build_actor_helper(name, model_path, density, scale, pose)

            bbox = info.get("bbox", None)
            if bbox is not None:
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                self.model_bbox_sizes[name] = common.to_tensor(bbox_size * scale, device=self.device)
            else:
                self.model_bbox_sizes[name] = torch.tensor([0.30, 0.20, 0.02], device=self.device, dtype=torch.float32)

        self.left_names: List[str] = []
        self.right_names: List[str] = []
        for p in self.pairs_meta:
            self.left_names.append(p["left"])
            self.right_names.append(p["right"])

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        assert b == self.num_envs

        if self._pair_ids_config is not None:
            assert len(self._pair_ids_config) == b, f"len(ids)={len(self._pair_ids_config)} must equal num_envs={b}"
            episode_id = torch.tensor(self._pair_ids_config, device=self.device, dtype=torch.long)
        else:
            num_pairs = len(self.pairs_meta)
            episode_id = options.get(
                "episode_id",
                torch.randint(low=0, high=max(1, num_pairs), size=(b,), device=self.device),
            )
            episode_id = (episode_id.reshape(b) % max(1, num_pairs)).to(dtype=torch.long)

        self._pair_ids = episode_id

        self._swap_lr = torch.full((b,), self.do_swap, device=self.device, dtype=torch.bool)

        self._answers = []
        self._questions = []
        for i in range(b):
            pair_idx = int(self._pair_ids[i].item())
            p = self.pairs_meta[pair_idx]
            ans = str(p.get("answer", "")).strip().lower()

            if ans in ["left", "right"]:
                if bool(self._swap_lr[i]):
                    ans = "right" if ans == "left" else "left"
                self._answers.append(ans)
            else:
                self._answers.append("right" if bool(self._swap_lr[i]) else "left")

            self._questions.append(str(p.get("question", "")))

        self.init_x_cube = self.cube_pose[0]
        self.init_y_cube = self.cube_pose[1]
        self.init_z_cube = self.cube_pose[2]
        self._scene_positions = {
            self._white_key: (
                self.cubes[self._white_key],
                (self.init_x_cube, self.init_y_cube, self.init_z_cube),
            ),
        }

        # ------------------------
        # NEW: init per-episode accumulators for soft metrics
        # ------------------------
        b = self.num_envs
        device = self.device

        # step counter
        self.episode_stats["soft_step"] = torch.zeros((b,), dtype=torch.long, device=device)

        # dwell over target: total + consecutive(current/max)
        self.episode_stats["tcp_dwell_target_steps"] = torch.zeros((b,), dtype=torch.long, device=device)
        self.episode_stats["tcp_dwell_target_consec"] = torch.zeros((b,), dtype=torch.long, device=device)
        self.episode_stats["tcp_dwell_target_consec_max"] = torch.zeros((b,), dtype=torch.long, device=device)

        # first-enter times (large init means "never")
        BIG = torch.full((b,), 10**9, dtype=torch.long, device=device)
        self.episode_stats["tcp_first_enter_target_t"] = BIG.clone()
        self.episode_stats["tcp_first_enter_wrong_t"] = BIG.clone()

        # min distances (xy) to target / wrong centers
        INF = torch.full((b,), 1e9, dtype=torch.float32, device=device)
        self.episode_stats["tcp_min_dist_target_xy"] = INF.clone()
        self.episode_stats["tcp_min_dist_wrong_xy"] = INF.clone()

        # side-time correctness (fraction can be derived)
        self.episode_stats["tcp_time_correct_side_steps"] = torch.zeros((b,), dtype=torch.long, device=device)

        # velocity alignment stats
        self.episode_stats["tcp_prev_xy"] = torch.full((b, 2), float("nan"), dtype=torch.float32, device=device)
        self.episode_stats["tcp_vel_align_sum"] = torch.zeros((b,), dtype=torch.float32, device=device)
        self.episode_stats["tcp_vel_align_pos_steps"] = torch.zeros((b,), dtype=torch.long, device=device)
        self.episode_stats["tcp_vel_align_cnt"] = torch.zeros((b,), dtype=torch.long, device=device)

        # progress-to-target stats
        self.episode_stats["tcp_prev_dist_target"] = torch.full((b,), float("nan"), dtype=torch.float32, device=device)
        self.episode_stats["tcp_progress_steps"] = torch.zeros((b,), dtype=torch.long, device=device)
        self.episode_stats["tcp_progress_cnt"] = torch.zeros((b,), dtype=torch.long, device=device)
        self.episode_stats["tcp_net_progress"] = torch.zeros((b,), dtype=torch.float32, device=device)
        self.episode_stats["tcp_d0"] = torch.full((b,), float("nan"), dtype=torch.float32, device=device)

        # release-near-target detection
        self.episode_stats["prev_is_grasped"] = torch.zeros((b,), dtype=torch.bool, device=device)
        self.episode_stats["release_near_target"] = torch.zeros((b,), dtype=torch.bool, device=device)
        self.episode_stats["release_recorded"] = torch.zeros((b,), dtype=torch.bool, device=device)

        # first board entered correctness (boolean derived after episode, but store now)
        self.episode_stats["tcp_first_choice_correct"] = torch.zeros((b,), dtype=torch.bool, device=device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = self.num_envs

        left_sel = [self.left_names[int(pid.item())] for pid in self._pair_ids]
        right_sel = [self.right_names[int(pid.item())] for pid in self._pair_ids]

        for i in range(b):
            if bool(self._swap_lr[i]):
                left_sel[i], right_sel[i] = right_sel[i], left_sel[i]

        def board_pose_tensor(names: List[str], x: float, y: float) -> torch.Tensor:
            sizes = torch.stack([self.model_bbox_sizes[n] for n in names], dim=0)
            half_h = sizes[:, 2] / 2.0
            z = torch.tensor(TABLE_Z, device=self.device, dtype=torch.float32) + half_h
            p = torch.stack(
                [
                    torch.full((b,), x, device=self.device, dtype=torch.float32),
                    torch.full((b,), y, device=self.device, dtype=torch.float32),
                    z.to(dtype=torch.float32),
                ],
                dim=1,
            )
            return p

        p_left = board_pose_tensor(left_sel, x=-0.25, y=-0.155)
        p_right = board_pose_tensor(right_sel, x=-0.25, y=+0.155)

        q_reset = torch.tensor(euler2quat(0.0, 0.0, 1.5707963267948966), device=self.device, dtype=torch.float32).reshape(1, 4).repeat(b, 1)

        for idx, name in enumerate(self.board_names):
            actor = self.objs_board[name]
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.5], device=self.device, dtype=torch.float32).reshape(1, 3).repeat(b, 1)

            is_left = torch.tensor([name == ln for ln in left_sel], device=self.device, dtype=torch.bool)
            is_right = torch.tensor([name == rn for rn in right_sel], device=self.device, dtype=torch.bool)

            p_cur = p_reset.clone()
            p_cur = torch.where(is_left.unsqueeze(1), p_left, p_cur)
            p_cur = torch.where(is_right.unsqueeze(1), p_right, p_cur)

            actor.set_pose(Pose.create_from_pq(p=p_cur, q=q_reset))

        self._current_left_names = left_sel
        self._current_right_names = right_sel



    def _ensure_soft_buffers(self, b: int, device: torch.device):
        """Ensure episode_stats has all soft-metric buffers (works even during reset/reconfigure)."""
        def _need(k, shape=None, dtype=None):
            if k not in self.episode_stats:
                return True
            v = self.episode_stats[k]
            if not torch.is_tensor(v):
                return True
            if shape is not None and tuple(v.shape) != tuple(shape):
                return True
            if dtype is not None and v.dtype != dtype:
                return True
            return False

        BIG = 10**9
        if _need("soft_step", (b,), torch.long):
            self.episode_stats["soft_step"] = torch.zeros((b,), dtype=torch.long, device=device)

        if _need("tcp_dwell_target_steps", (b,), torch.long):
            self.episode_stats["tcp_dwell_target_steps"] = torch.zeros((b,), dtype=torch.long, device=device)
        if _need("tcp_dwell_target_consec", (b,), torch.long):
            self.episode_stats["tcp_dwell_target_consec"] = torch.zeros((b,), dtype=torch.long, device=device)
        if _need("tcp_dwell_target_consec_max", (b,), torch.long):
            self.episode_stats["tcp_dwell_target_consec_max"] = torch.zeros((b,), dtype=torch.long, device=device)

        if _need("tcp_first_enter_target_t", (b,), torch.long):
            self.episode_stats["tcp_first_enter_target_t"] = torch.full((b,), BIG, dtype=torch.long, device=device)
        if _need("tcp_first_enter_wrong_t", (b,), torch.long):
            self.episode_stats["tcp_first_enter_wrong_t"] = torch.full((b,), BIG, dtype=torch.long, device=device)

        if _need("tcp_min_dist_target_xy", (b,), torch.float32):
            self.episode_stats["tcp_min_dist_target_xy"] = torch.full((b,), 1e9, dtype=torch.float32, device=device)
        if _need("tcp_min_dist_wrong_xy", (b,), torch.float32):
            self.episode_stats["tcp_min_dist_wrong_xy"] = torch.full((b,), 1e9, dtype=torch.float32, device=device)

        if _need("tcp_time_correct_side_steps", (b,), torch.long):
            self.episode_stats["tcp_time_correct_side_steps"] = torch.zeros((b,), dtype=torch.long, device=device)

        if _need("tcp_prev_xy", (b, 2), torch.float32):
            self.episode_stats["tcp_prev_xy"] = torch.full((b, 2), float("nan"), dtype=torch.float32, device=device)
        if _need("tcp_vel_align_sum", (b,), torch.float32):
            self.episode_stats["tcp_vel_align_sum"] = torch.zeros((b,), dtype=torch.float32, device=device)
        if _need("tcp_vel_align_pos_steps", (b,), torch.long):
            self.episode_stats["tcp_vel_align_pos_steps"] = torch.zeros((b,), dtype=torch.long, device=device)
        if _need("tcp_vel_align_cnt", (b,), torch.long):
            self.episode_stats["tcp_vel_align_cnt"] = torch.zeros((b,), dtype=torch.long, device=device)

        if _need("tcp_prev_dist_target", (b,), torch.float32):
            self.episode_stats["tcp_prev_dist_target"] = torch.full((b,), float("nan"), dtype=torch.float32, device=device)
        if _need("tcp_progress_steps", (b,), torch.long):
            self.episode_stats["tcp_progress_steps"] = torch.zeros((b,), dtype=torch.long, device=device)
        if _need("tcp_progress_cnt", (b,), torch.long):
            self.episode_stats["tcp_progress_cnt"] = torch.zeros((b,), dtype=torch.long, device=device)
        if _need("tcp_net_progress", (b,), torch.float32):
            self.episode_stats["tcp_net_progress"] = torch.zeros((b,), dtype=torch.float32, device=device)
        if _need("tcp_d0", (b,), torch.float32):
            self.episode_stats["tcp_d0"] = torch.full((b,), float("nan"), dtype=torch.float32, device=device)

        if _need("prev_is_grasped", (b,), torch.bool):
            self.episode_stats["prev_is_grasped"] = torch.zeros((b,), dtype=torch.bool, device=device)
        if _need("release_near_target", (b,), torch.bool):
            self.episode_stats["release_near_target"] = torch.zeros((b,), dtype=torch.bool, device=device)
        if _need("release_recorded", (b,), torch.bool):
            self.episode_stats["release_recorded"] = torch.zeros((b,), dtype=torch.bool, device=device)

        if _need("tcp_first_choice_correct", (b,), torch.bool):
            self.episode_stats["tcp_first_choice_correct"] = torch.zeros((b,), dtype=torch.bool, device=device)

        # Optional: make sure base flag exists too (some reset paths call evaluate before it's made)
        if _need("is_src_obj_grasped", (b,), torch.bool):
            self.episode_stats["is_src_obj_grasped"] = torch.zeros((b,), dtype=torch.bool, device=device)

    def evaluate(self):
        b = self.num_envs
        device = self.device

        self._ensure_soft_buffers(b=b, device=device)

        success = torch.zeros((b,), dtype=torch.bool, device=device)
        is_answered = torch.zeros((b,), dtype=torch.bool, device=device)

        # existing cube-based soft metrics
        is_success_soft_v1 = torch.zeros((b,), dtype=torch.bool, device=device)

        cube_actor = self.cubes[self._white_key]
        is_grasped = self.agent.is_grasping(cube_actor)

        # keep your episode grasp flag (ever grasped)
        self.episode_stats["is_src_obj_grasped"] = (self.episode_stats["is_src_obj_grasped"] | is_grasped)

        cube_p = cube_actor.pose.p  # [b,3]
        cube_half = torch.full((b,), CUBE_HALF_SIZE, device=device)
        margin_xy = 0.08  # your existing tolerance

        # ------------------------
        # NEW: TCP/EE position for intent metrics
        # ------------------------
        tcp_p = self._get_tcp_p()  # [b,3]

        # step counter
        t_step = self.episode_stats.get("soft_step", torch.zeros((b,), dtype=torch.long, device=device))
        self.episode_stats["soft_step"] = t_step + 1

        # thresholds in steps
        steps_3s = self._steps_for_seconds(3.0)  # used for ">= 3 seconds" style dwell
        # safety margins for "above board"
        tcp_z_margin = 0.02  # 2cm above board top is "over target"

        # previous grasp for release detection
        prev_is_grasped = self.episode_stats.get("prev_is_grasped", torch.zeros((b,), dtype=torch.bool, device=device))
        self.episode_stats["prev_is_grasped"] = is_grasped.clone()

        # prev tcp xy for velocity
        prev_xy = self.episode_stats.get("tcp_prev_xy", torch.full((b,2), float("nan"), dtype=torch.float32, device=device))

        # accumulators (already initialized in _initialize_episode_pre)
        dwell_steps = self.episode_stats["tcp_dwell_target_steps"]
        dwell_consec = self.episode_stats["tcp_dwell_target_consec"]
        dwell_consec_max = self.episode_stats["tcp_dwell_target_consec_max"]

        first_t_tgt = self.episode_stats["tcp_first_enter_target_t"]
        first_t_wrong = self.episode_stats["tcp_first_enter_wrong_t"]

        min_d_tgt = self.episode_stats["tcp_min_dist_target_xy"]
        min_d_wrong = self.episode_stats["tcp_min_dist_wrong_xy"]

        side_correct_steps = self.episode_stats["tcp_time_correct_side_steps"]

        vel_sum = self.episode_stats["tcp_vel_align_sum"]
        vel_pos = self.episode_stats["tcp_vel_align_pos_steps"]
        vel_cnt = self.episode_stats["tcp_vel_align_cnt"]

        prev_dist = self.episode_stats["tcp_prev_dist_target"]
        prog_steps = self.episode_stats["tcp_progress_steps"]
        prog_cnt = self.episode_stats["tcp_progress_cnt"]
        net_prog = self.episode_stats["tcp_net_progress"]
        d0 = self.episode_stats["tcp_d0"]

        release_near_target = self.episode_stats["release_near_target"]
        release_recorded = self.episode_stats["release_recorded"]

        # ------------------------
        # per-env loop (b is small; safe and clear)
        # ------------------------
        for i in range(b):
            ans = self._answers[i]  # "left"/"right"

            # helper: cube on board (your hard metric)
            def check_on_board(board_name: str) -> bool:
                board_actor = self.objs_board[board_name]
                bp = board_actor.pose.p[i]
                size = self.model_bbox_sizes[board_name]
                half_x, half_y, half_z = size[0].item() / 2.0, size[1].item() / 2.0, size[2].item() / 2.0

                dx = abs(float(cube_p[i][0] - bp[0]))
                dy = abs(float(cube_p[i][1] - bp[1]))
                dz = float(cube_p[i][2] - bp[2])

                on_xy = (dx <= half_x + margin_xy) and (dy <= half_y + margin_xy)
                above = dz >= (half_z + cube_half[i].item() - 0.01)
                return on_xy and above

            left_name = self._current_left_names[i]
            right_name = self._current_right_names[i]

            on_left = check_on_board(left_name)
            on_right = check_on_board(right_name)

            success[i] = on_left if ans == "left" else on_right
            is_answered[i] = on_left or on_right

            # existing cube soft metrics (your originals)
            cube_y = float(cube_p[i][1])
            cube_x = float(cube_p[i][0])

            x_delta = 2 * CUBE_HALF_SIZE
            y_delta = 2 * CUBE_HALF_SIZE

            if ans == "left":
                is_success_soft_v1[i] = cube_y < (self.init_y_cube - x_delta) and cube_x < (self.init_x_cube - y_delta)
            else:
                is_success_soft_v1[i] = cube_y > (self.init_y_cube + x_delta) and cube_x < (self.init_x_cube - y_delta)
            # ------------------------
            # NEW: intent metrics using TCP (or EE) over target
            # ------------------------
            target_name = left_name if ans == "left" else right_name
            wrong_name = right_name if ans == "left" else left_name

            tgt_actor = self.objs_board[target_name]
            wrong_actor = self.objs_board[wrong_name]

            tgt_p = tgt_actor.pose.p[i]          # [3]
            wrong_p = wrong_actor.pose.p[i]      # [3]
            tgt_size = self.model_bbox_sizes[target_name]   # [3]
            wrong_size = self.model_bbox_sizes[wrong_name]  # [3]

            # --- Over-target check (TCP inside expanded bbox + above board)
            tx, ty, tz = float(tcp_p[i][0]), float(tcp_p[i][1]), float(tcp_p[i][2])

            tgt_half_x = float(tgt_size[0].item() / 2.0)
            tgt_half_y = float(tgt_size[1].item() / 2.0)
            tgt_half_z = float(tgt_size[2].item() / 2.0)

            dx_t = abs(tx - float(tgt_p[0]))
            dy_t = abs(ty - float(tgt_p[1]))
            dz_t = tz - float(tgt_p[2])  # relative to board center z

            tcp_on_target_xy = (dx_t <= tgt_half_x + margin_xy) and (dy_t <= tgt_half_y + margin_xy)
            tcp_above_target = dz_t >= (tgt_half_z + tcp_z_margin)
            tcp_over_target = tcp_on_target_xy and tcp_above_target

            # dwell totals + consecutive max
            if tcp_over_target:
                dwell_steps[i] += 1
                dwell_consec[i] += 1
                if dwell_consec[i] > dwell_consec_max[i]:
                    dwell_consec_max[i] = dwell_consec[i]
            else:
                dwell_consec[i] = 0

            # first-enter times
            cur_t = int(self.episode_stats["soft_step"][i].item())
            if tcp_over_target and int(first_t_tgt[i].item()) >= 10**9:
                first_t_tgt[i] = cur_t

            # wrong enter check
            wx, wy, wz = float(wrong_p[0]), float(wrong_p[1]), float(wrong_p[2])
            w_half_x = float(wrong_size[0].item() / 2.0)
            w_half_y = float(wrong_size[1].item() / 2.0)
            w_half_z = float(wrong_size[2].item() / 2.0)

            dx_w = abs(tx - wx)
            dy_w = abs(ty - wy)
            dz_w = tz - wz

            tcp_on_wrong_xy = (dx_w <= w_half_x + margin_xy) and (dy_w <= w_half_y + margin_xy)
            tcp_above_wrong = dz_w >= (w_half_z + tcp_z_margin)
            tcp_over_wrong = tcp_on_wrong_xy and tcp_above_wrong

            if tcp_over_wrong and int(first_t_wrong[i].item()) >= 10**9:
                first_t_wrong[i] = cur_t

            # min distance to centers (xy)
            d_t = ((tx - float(tgt_p[0])) ** 2 + (ty - float(tgt_p[1])) ** 2) ** 0.5
            d_w = ((tx - wx) ** 2 + (ty - wy) ** 2) ** 0.5
            if d_t < float(min_d_tgt[i].item()):
                min_d_tgt[i] = d_t
            if d_w < float(min_d_wrong[i].item()):
                min_d_wrong[i] = d_w

            # correct side time (since your left/right differ by sign of y)
            # "left" board is y < 0, "right" board is y > 0 (your placement uses ±0.155)
            target_sign = -1.0 if ans == "left" else 1.0
            if (target_sign * ty) > 0.0:
                side_correct_steps[i] += 1

            # velocity alignment + progress (need prev_xy)
            cur_xy = torch.tensor([tx, ty], device=device, dtype=torch.float32)
            if not torch.isnan(prev_xy[i]).any():
                v = cur_xy - prev_xy[i]  # [2]
                # direction from current pos to target center
                dir_vec = torch.tensor([float(tgt_p[0]) - tx, float(tgt_p[1]) - ty], device=device, dtype=torch.float32)
                norm = torch.linalg.norm(dir_vec) + 1e-8
                u = dir_vec / norm
                s = float((v * u).sum().item())
                vel_sum[i] += s
                vel_cnt[i] += 1
                if s > 0:
                    vel_pos[i] += 1

            # progress: d(t) should decrease
            dist_t = torch.tensor(d_t, device=device, dtype=torch.float32)
            if torch.isnan(d0[i]):
                d0[i] = dist_t
            if not torch.isnan(prev_dist[i]):
                prog_cnt[i] += 1
                if dist_t < prev_dist[i]:
                    prog_steps[i] += 1
            prev_dist[i] = dist_t

            # net progress (from start -> best)
            # track via: net_prog = max(net_prog, d0 - min_dist_seen)
            # here min_d_tgt holds min seen so far
            net_prog[i] = torch.maximum(net_prog[i], d0[i] - min_d_tgt[i])

            # release-near-target (first time grasp goes True->False)
            if (not bool(release_recorded[i])) and bool(prev_is_grasped[i]) and (not bool(is_grasped[i])):
                release_recorded[i] = True
                release_near_target[i] = torch.tensor(bool(tcp_over_target), device=device)

        # store updated tensors back (some modified in-place, but keep explicit)
        self.episode_stats["tcp_dwell_target_steps"] = dwell_steps
        self.episode_stats["tcp_dwell_target_consec"] = dwell_consec
        self.episode_stats["tcp_dwell_target_consec_max"] = dwell_consec_max
        self.episode_stats["tcp_first_enter_target_t"] = first_t_tgt
        self.episode_stats["tcp_first_enter_wrong_t"] = first_t_wrong
        self.episode_stats["tcp_min_dist_target_xy"] = min_d_tgt
        self.episode_stats["tcp_min_dist_wrong_xy"] = min_d_wrong
        self.episode_stats["tcp_time_correct_side_steps"] = side_correct_steps
        self.episode_stats["tcp_vel_align_sum"] = vel_sum
        self.episode_stats["tcp_vel_align_pos_steps"] = vel_pos
        self.episode_stats["tcp_vel_align_cnt"] = vel_cnt
        self.episode_stats["tcp_prev_xy"] = tcp_p[:, :2].to(dtype=torch.float32)
        self.episode_stats["tcp_prev_dist_target"] = prev_dist
        self.episode_stats["tcp_progress_steps"] = prog_steps
        self.episode_stats["tcp_progress_cnt"] = prog_cnt
        self.episode_stats["tcp_net_progress"] = net_prog
        self.episode_stats["tcp_d0"] = d0
        self.episode_stats["release_near_target"] = release_near_target
        self.episode_stats["release_recorded"] = release_recorded

        # ------------------------
        # NEW: derived boolean soft success signals (per-step updated, but meaningful at episode end)
        # ------------------------
        # 1) dwell >= 3 seconds (total or max consecutive)
        dwell_3s = (dwell_steps >= steps_3s) | (dwell_consec_max >= steps_3s)
        self.episode_stats["tcp_dwell_target_3s"] = dwell_3s

        # 2) first choice correctness: entered target before wrong (and entered something at all)
        BIG = 10**9
        entered_any = (first_t_tgt < BIG) | (first_t_wrong < BIG)
        first_choice_correct = (first_t_tgt < first_t_wrong) & entered_any
        self.episode_stats["tcp_first_choice_correct"] = first_choice_correct

        # 3) min-dist winner: got closer to target than wrong
        self.episode_stats["tcp_closer_to_target_than_wrong"] = (min_d_tgt < min_d_wrong)

        # 4) velocity alignment positive fraction (avoid div by 0)
        vel_cnt_safe = torch.clamp(vel_cnt, min=1)
        self.episode_stats["tcp_vel_align_mean"] = vel_sum / vel_cnt_safe.to(torch.float32)
        self.episode_stats["tcp_vel_align_pos_frac"] = vel_pos.to(torch.float32) / vel_cnt_safe.to(torch.float32)

        # 5) progress fraction
        prog_cnt_safe = torch.clamp(prog_cnt, min=1)
        self.episode_stats["tcp_progress_frac"] = prog_steps.to(torch.float32) / prog_cnt_safe.to(torch.float32)

        # 6) side correctness fraction
        step_cnt = self.episode_stats["soft_step"].to(torch.float32)
        step_cnt_safe = torch.clamp(step_cnt, min=1.0)
        self.episode_stats["tcp_correct_side_frac"] = side_correct_steps.to(torch.float32) / step_cnt_safe

        # ------------------------
        # keep your original episode_stats outputs
        # ------------------------
        self.episode_stats["is_answered"] = is_answered
        self.episode_stats["is_success"] = success
        self.episode_stats["is_success_soft_v1"] = is_success_soft_v1

        # only-grasped versions (your originals)
        is_src_obj_grasped = self.episode_stats["is_src_obj_grasped"]

        self.episode_stats["tcp_dwell_target_3s_only_grasped"] = dwell_3s & is_src_obj_grasped
        self.episode_stats["tcp_first_choice_correct_only_grasped"] = first_choice_correct & is_src_obj_grasped
        self.episode_stats["tcp_closer_to_target_than_wrong_only_grasped"] = self.episode_stats["tcp_closer_to_target_than_wrong"] & is_src_obj_grasped
        self.episode_stats["release_near_target_only_grasped"] = self.episode_stats["release_near_target"] & is_src_obj_grasped

        return dict(**self.episode_stats, success=success)

    def get_language_instruction(self) -> List[str]:
        return self._questions

    def get_target_name(self) -> List[str]:
        return [("Left" if a == "left" else "Right") for a in self._answers]

    def where_target(self) -> List[str]:
        return self._answers

    def get_ids(self) -> List[int]:
        ids: List[int] = []
        for i in range(self.num_envs):
            pair_pos = int(self._pair_ids[i].item())
            pair_meta = self.pairs_meta[pair_pos]
            idx = int(pair_meta.get("index", pair_pos))
            ids.append(idx)
        return ids
