import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv

class SimplerWrapper:
    def __init__(self, all_args, unnorm_state=None, extra_seed=0):
        self.args = all_args
        self.unnorm_state = unnorm_state

        self.num_envs = self.args.num_envs
        robot_control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"

        old_z = 0.88 + 0.015
        self.args.cube_pose = (-0.131, 0.0411, old_z + 0.1180)
        self.args.initial_qpos = np.array([
                    -0.01840777, 0.0398835, 0.22242722,
                    -0.00460194, 1.36524296, 0.00153398,
                    0.0326, 0.0326,
                ])
        
        self.init_grasp_steps = self.args.init_grasp_steps
        self.hold_cube_steps = self.args.hold_cube_steps

        env_config = dict(
            id=self.args.env_id,
            num_envs=self.args.num_envs,
            obs_mode="rgb+segmentation",
            control_mode=robot_control_mode,
            sim_backend="gpu",
            sim_config={
                "sim_freq": 500,
                "control_freq": 5,
            },
            max_episode_steps=self.args.episode_len,
            sensor_configs={"shader_pack": "default"},
            assets=self.args.assets,
            cube_pose = self.args.cube_pose,
            initial_qpos = self.args.initial_qpos,
            ids = self.args.ids,
            do_swap = self.args.do_swap,
            rgb_overlay_paths = self.args.rgb_overlay_paths,
        )

        self.env: BaseEnv = gym.make(**env_config)
        self.env.reset(seed=[self.args.seed for i in range(self.args.num_envs)])

        self.reward_old = torch.zeros(self.args.num_envs, 1, dtype=torch.float32)  # [B, 1]

        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    def _compute_eef_pos(self, obs):
        """
        Compute end-effector position in the format expected by INT-ACT's Pi0 model.
        Returns an 8-dimensional vector: [x, y, z, qw, qx, qy, qz, gripper_openness]
        
        This matches the format from INT-ACT's modified ManiSkill base_agent.py:
        https://github.com/Physical-Intelligence/INT-ACT/blob/main/third_party/ManiSkill/mani_skill/agents/base_agent.py#L338-L345
        """
        # Get agent from environment
        agent = self.env.unwrapped.agent
       
        # Get base pose and end-effector pose
        base_pose = agent.robot.pose  # Robot base pose (wxyz quaternion format)
        ee_pose = agent.tcp.pose  # End-effector TCP pose
        
        # Transform EE pose to base frame
        ee_in_base = base_pose.inv() * ee_pose
        gripper_closedness = agent.robot.get_qpos()[:, -1:]  # Last joint is typically gripper
        gripper_nwidth = 1 - gripper_closedness  # Convert to openness
        eef_pos = torch.cat([ee_in_base.raw_pose, gripper_nwidth], dim=1)
        
        return eef_pos
    
    def _xiaomi_eef_pos(self):
        eef_pos = self._compute_eef_pos(None)
        return [{"agent": {"eef_pos": x}} for x in eef_pos]

    def get_reward(self, info):
        reward = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(info["success"].device)  # [B, 1]
        reward_diff = reward 
        return reward_diff


    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        action_scale = 1.0

        # Extract predicted action tokens and translate into (normalized) continuous actions
        pact_token = raw_actions.cpu().numpy()  # [B, dim]
        dact = 32000 - pact_token  # [B, dim]
        dact = np.clip(dact - 1, a_min=0, a_max=254)  # [B, dim]
        normalized_actions = np.asarray([self.bin_centers[da] for da in dact])  # [B, dim]

        # Unnormalize actions
        action_norm_stats = self.unnorm_state
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))  # [dim]
        mask = np.asarray(mask).reshape(1, -1)  # [1, dim]
        action_high = np.array(action_norm_stats["q99"]).reshape(1, -1)  # [1, dim]
        action_low = np.array(action_norm_stats["q01"]).reshape(1, -1)  # [1, dim]
        raw_action_np = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        raw_action = {
            "world_vector": raw_action_np[:, :3],
            "rotation_delta": raw_action_np[:, 3:6],
            "open_gripper": raw_action_np[:, 6:7],  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * action_scale  # [B, 3]
        action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0  # [B, 1]

        # origin euler
        action["rot_axangle"] = raw_action["rotation_delta"]

        action = {k: torch.tensor(v) for k, v in action.items()}  # to float32 ?

        action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)

        # to tpdv
        action = action.to(raw_actions.device)

        return action

    def reset(self, obj_set: str, same_init: bool = False, specific_objects: list[str] = None):
        options = {}
        options["obj_set"] = obj_set
        # if specific_objects is not None:
        #     options["specific_objects"] = specific_objects
        # if same_init:
        #     options["episode_id"] = torch.randint(1000000000, (1,)).expand(self.num_envs).to(self.env.device)  # [B]

        obs, info = self.env.reset(options=options)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        instruction = self.env.unwrapped.get_language_instruction()

        self.reward_old = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(obs_image.device)  # [B, 1]
        
        # Compute eef_pos to match INT-ACT format
        info["proprio"] = self._xiaomi_eef_pos()
        info["pi_0"] = self._compute_eef_pos(None)

        for i in range(self.init_grasp_steps):
            action = torch.Tensor([ 0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  -1.0])
            action = action.unsqueeze(0).repeat(1, self.args.num_envs, 1)[0]
            # print(action)
            obs, reward, terminated, done, info = self.env.step(action)
            obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
            # print(f"{obs['agent']=}")
            # inspect_dict(obs)
            # info["proprio"] = [{"agent": {k: v[i].cpu().numpy() for k, v in obs["agent"].items()}} for i in range(self.num_envs)]
            info["proprio"] = self._xiaomi_eef_pos()
        info["pi_0"] = self._compute_eef_pos(None)


        return obs_image, instruction, info

    def step(self, raw_action):
        # if raw_action.dtype == torch.float32 or raw_action.dtype == torch.float64:
        #     action = raw_action
        # else:
        #     action = self._process_action(raw_action)

        action = raw_action

        current_step = self.env.unwrapped.elapsed_steps.cpu().numpy()[0]
        if current_step < self.init_grasp_steps + self.hold_cube_steps :
            action[:, 6] = -1.0


        obs, _reward, _terminated, truncated, info = self.env.step(action)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        truncated = truncated.reshape(-1, 1)  # [B, 1]

        # calculate reward
        reward = self.get_reward(info)

        # process episode info
        if truncated.any():
            info["episode"] = {}
            for k in ["success"]:
                v = [info[k][idx].item() for idx in range(self.num_envs)]
                info["episode"][k] = v

        # Compute eef_pos to match INT-ACT format
        # eef_pos = self._compute_eef_pos(obs)
        # agent_pos = {"eef_pos":eef_pos}
        
        # Add proprio to info for Pi0
        # info["proprio"] = [{"agent": {k: v[i].cpu().numpy() for k, v in obs["agent"].items()}} for i in range(self.num_envs)]

        # info["pi_0"] = agent_pos
        info["proprio"] = self._xiaomi_eef_pos()
        info["pi_0"] = self._compute_eef_pos(None)

        return obs_image, reward, truncated, info
