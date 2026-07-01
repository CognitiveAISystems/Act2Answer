import os
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated
import torch
import numpy as np
import tyro
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
from dataclasses import field
from typing import Optional
from shutil import make_archive
from os.path import isfile
import json

from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _overlay_instruction(img, text):
    """Draw the task instruction as a centered banner at the bottom of a frame (RGB uint8)."""
    import numpy as np
    try:
        import cv2
    except Exception:
        return img
    img = np.ascontiguousarray(img).astype(np.uint8)
    if img.ndim != 3 or img.shape[2] != 3 or not text:
        return img
    text = str(text).upper()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = w / 640.0 * 0.8
    thick = max(1, int(round(scale * 1.6)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    while tw > w * 0.92 and scale > 0.3:
        scale -= 0.05
        thick = max(1, int(round(scale * 1.6)))
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    padx, pady = int(round(scale * 18)), int(round(scale * 12))
    bw, bh = tw + 2 * padx, th + base + 2 * pady
    x0 = (w - bw) // 2
    y1 = h - int(round(h * 0.10)); y0 = y1 - bh
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bw, y1), (40, 120, 95), -1)  # teal-green (RGB)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(img, text, (x0 + padx, y1 - pady - base), font, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
    return img


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "Act2AnswerV4-v1"
    """The Act2Answer environment ID. The release benchmark uses Act2AnswerV4-v1."""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "VLThink-eval"

    # env
    num_envs: int = field(default=0, init=False)
    episode_len: int = 80
    use_same_init: bool = False

    obj_set: str = "test"

    steps_max: int = 2000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 10
    interval_save: int = 40

    # buffer
    buffer_inferbatch: int = 16
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95

    # vla
    vla_kind: Annotated[str, tyro.conf.arg(aliases=["-v"])] = "spatialvla"
    vla_path: Annotated[str, tyro.conf.arg(aliases=["-p"])] | None = None
    vla_unnorm_key: str = "sft"
    vla_load_path: str = ""
    vla_lora_rank: int = 32

    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    # other
    alg_grpo_fix: bool = True
    only_render: bool = False
    render_info: bool = False

    assets: str = "more_state"
    do_swap: bool = True

    rgb_overlay_paths: dict[str, str] = field(
        default_factory=lambda: {"3rd_view_camera": "./bridge_real_eval_1.png"}
    )

    shard_index: Optional[int] = None
    num_shards: Optional[int] = None

    ids: list[int] = field(default_factory=list, init=False)
    total_envs: int = field(default=0, init=False)
    shard_start: int = field(default=0, init=False)
    shard_end: int = field(default=0, init=False)

    init_grasp_steps: int = 10
    hold_cube_steps: int = 15

    archive_path: Optional[str] = None
    asset_path: str = "."
    output_dir: Optional[str] = None


def resolve_eval_ids(args: Args) -> None:
    pairs_path = Path(args.asset_path) / args.assets / "pairs.json"
    total_envs = len(json.load(open(pairs_path, "r")))

    has_shard_index = args.shard_index is not None
    has_num_shards = args.num_shards is not None
    if has_shard_index != has_num_shards:
        raise ValueError("--shard_index and --num_shards must be provided together")

    if has_shard_index:
        if args.num_shards <= 0:
            raise ValueError(f"--num_shards must be positive, got {args.num_shards}")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError(
                f"--shard_index must be in [0, {args.num_shards}), got {args.shard_index}"
            )

        shard_start = total_envs * args.shard_index // args.num_shards
        shard_end = total_envs * (args.shard_index + 1) // args.num_shards
    else:
        shard_start = 0
        shard_end = total_envs

    if shard_start >= shard_end:
        raise ValueError(
            f"Empty shard: total_envs={total_envs}, shard_index={args.shard_index}, "
            f"num_shards={args.num_shards}"
        )

    args.total_envs = total_envs
    args.shard_start = shard_start
    args.shard_end = shard_end
    args.ids = list(range(shard_start, shard_end))
    args.num_envs = len(args.ids)

    print(
        f"Selected {args.num_envs}/{args.total_envs} envs: "
        f"ids=[{args.shard_start}, {args.shard_end})"
    )


def get_archive_path(args: Args) -> Optional[str]:
    if not args.archive_path:
        return None

    shard_suffix = ""
    if args.num_shards is not None and args.num_shards > 1:
        shard_suffix = f"-{args.shard_start}-{args.shard_end}"

    return (
        f"{args.archive_path}/glob-{args.vla_kind}-{args.assets}"
        f"{shard_suffix}{'-swap' if args.do_swap else ''}"
    )


class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args
        if not self.args.ids:
            resolve_eval_ids(self.args)

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        default_output_dir = Path(__file__).resolve().parents[2] / "outputs"
        output_root = Path(
            all_args.output_dir or os.environ.get("A2A_OUTPUT_DIR") or default_output_dir
        ).expanduser()
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root
        self.save_dir = output_root / self.args.name
        self.glob_dir = self.save_dir / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Act2Answer] writing videos and stats to {self.glob_dir}", flush=True)

        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))

        if all_args.vla_kind == "openvla":
            from simpler_env.policies.openvla.openvla_adapter import OpenVLAInference

            policy_setup = "widowx_bridge"
            self.policy = OpenVLAInference(all_args, device_id_other)
        elif all_args.vla_kind == "spatialvla":
            from simpler_env.policies.spatialvla.spatialvla import SpatialVLAPolicy

            policy_setup = "widowx_bridge"
            vla_path = all_args.vla_path or "IPEC-COMMUNITY/spatialvla-4b-224-pt"
            self.policy = SpatialVLAPolicy(
                device_id_other, all_args, vla_path, None, policy_setup
            )
        elif all_args.vla_kind == "magma":
            from simpler_env.policies.magma.magma_model import MagmaInference

            policy_setup = "widowx_bridge"
            vla_path = all_args.vla_path or "microsoft/Magma-8B"
            self.policy = MagmaInference(device_id_other, all_args, vla_path)
        elif all_args.vla_kind == "internvla":
            from simpler_env.policies.internvla.internvla import M1Inference

            policy_setup = "widowx_bridge"
            vla_path = all_args.vla_path
            self.policy = M1Inference(vla_path, policy_setup=policy_setup)
        elif all_args.vla_kind == "xiaomi":
            from simpler_env.policies.xiaomi.xiaomi import XiaomiRoboticsPolicy

            self.policy = XiaomiRoboticsPolicy()
        elif all_args.vla_kind == "molmoact":
            from simpler_env.policies.molmoact.molmoact import MolmoActPolicy

            self.policy = MolmoActPolicy()
        elif all_args.vla_kind == "pi0":
            from simpler_env.policies.pi0.pi0_adapter import Pi0Inference

            self.policy = Pi0Inference(all_args, device_id_other)
        else:
            raise ValueError("Unknown VLA kind")

        self.env = SimplerWrapper(self.args)

        # buffer
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
        )
        minibatch_count = self.buffer.get_minibatch_count()
        print(f"Buffer minibatch count: {minibatch_count}")

    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]

        actions = []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {
                k: v[i : i + self.args.buffer_inferbatch] for k, v in obs.items()
            }
            action = self.policy.get_action(obs_batch, deterministic)
            actions.append(action)

        return torch.cat(actions, dim=0).to(self.device)

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str) -> dict:
        self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [
            {
                "image": [],  # obs_t: [0, T-1]
                "instruction": "",
                "action": [],  # a_t: [0, T-1]
                "info": [],  # info after executing a_t: [1, T]
            }
            for idx in range(self.args.num_envs)
        ]

        obs_img, instruction, info = self.env.reset(obj_set)
        print("instruction[:3]:", instruction[:3])

        # data dump: instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction, proprio=info['proprio'], pi_0=info.get('pi_0'))
            action = self._get_action(obs, deterministic=True)

            obs_img_new, _, _, env_info = self.env.step(action)

            # info
            print(
                {
                    k: round(v.to(torch.float32).mean().tolist(), 4)
                    for k, v in env_info.items()
                    if k != "episode" and k != 'proprio'
                }
            )
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                log_image = obs_img[i].cpu().numpy()
                log_action = action[i].cpu().numpy().tolist()
                log_info = {
                    k: v[i].tolist() for k, v in env_info.items() if k != "episode" and k != 'proprio'
                }
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # update obs_img
            obs_img = obs_img_new
            info = env_info

        # data dump: last image
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j], extras=[f"Ins: {instruction[i]}"]
                    )

            for _k in range(len(images)):
                images[_k] = _overlay_instruction(images[_k], instruction[i])

            success = int(infos[-1]["success"])
            images_to_video(
                images, str(exp_dir), f"video_{i}-s_{success}", fps=10, verbose=False
            )

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # save stats
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }

        save_stats = {}
        save_stats["env_name"] = self.args.env_id
        save_stats["ep_len"] = self.args.episode_len
        save_stats["epoch"] = epoch
        save_stats["stats"] = {k: v.item() for k, v in env_stats.items()}
        save_stats["instruction"] = {idx: ins for idx, ins in enumerate(instruction)}
        save_stats["last_info"] = last_info

        for k, v in env_info.items():
            if k != "episode" and k != 'proprio':
                save_stats["stats"][f"{k}"] = round(
                    v.to(torch.float32).mean().tolist(), 4
                )

        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret


def main():
    start = time.monotonic()

    args = tyro.cli(Args)
    resolve_eval_ids(args)
    archive_path = get_archive_path(args)

    if archive_path:
        if isfile(archive_path + ".zip"):
            print("Eval result already exists, skipping...")
            exit(0)

    runner = Runner(args)
    runner.render(epoch=0, obj_set=args.obj_set)

    if archive_path:
        print(f"Eval done in {time.monotonic() - start:2f} seconds")
        print("Archiving results")

        make_archive(
            archive_path,
            "zip",
            root_dir=runner.glob_dir,
        )


if __name__ == "__main__":
    main()
