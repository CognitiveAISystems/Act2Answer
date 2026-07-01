import os
import sys
from pathlib import Path
import torch
import numpy as np
import draccus

# pi0 deps (agent/experiments/lerobot/utils) live in a separate checkout; override with PI0_DEPS_ROOT
_PI0_DEPS_ROOT = os.environ.get("PI0_DEPS_ROOT", "/home/jovyan/nkachaev/Act2Answer")
sys.path.insert(0, _PI0_DEPS_ROOT)

from agent.configuration_pipeline import TrainPipelineConfig, EvalConfig
from experiments.policies.policy_wrapper import LeRobotPolicyWrapper
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as LeRobotPI0Policy

class PI0Policy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        
        # Construct pipeline config
        eval_cfg = EvalConfig(
            simulator_name="simpler",
            env_adapter="BridgeSimplerAdapter",
            action_step=1,
        )
        
        configs_dir = Path(__file__).parent / "configs"
        config_path = str(configs_dir / "pi0_finetune_bridge_ev.yaml")

        def load_config_only():
            test_args = [
                "--config_path", config_path,
                "--seed", str(all_args.seed),
                "--use_bf16", "True",
                "--use_wandb", "False"
            ]
            # the yaml uses `!include ./pi0_finetune_bridge.json`; resolve it next to the yaml
            _cwd = os.getcwd()
            try:
                os.chdir(configs_dir)
                cfg = draccus.parse(TrainPipelineConfig, args=test_args)
            finally:
                os.chdir(_cwd)
            return cfg

        pipeline_cfg = load_config_only()
        # the env adapter reads dataset_statistics_path later (relative to cwd at runtime); pin absolute
        if hasattr(pipeline_cfg, "env") and hasattr(pipeline_cfg.env, "dataset_statistics_path"):
            pipeline_cfg.env.dataset_statistics_path = str(configs_dir / "bridge_statistics.json")
        
        self.wrapper = LeRobotPolicyWrapper(pipeline_cfg, LeRobotPI0Policy)
        self.wrapper._initialze_model_server(all_args.vla_path)
        self.wrapper.env_adapter = self.wrapper._initialize_env_adapter()
        
    def get_action(self, x: dict, deterministic=True, return_raw=False):
        images = x["image"].cpu().numpy() # [B, H, W, 3]
        tasks = x["task_description"]
        obs = x.get("pi_0")
        
        batch_size = images.shape[0]
        
        # print(f"Images shape: {images.shape}")
        # print(f"Tasks: {tasks}")
        # print(f"eef_pos: {obs['eef_pos'].shape}")
        # for i in range(batch_size):
        element = {
            "observation.images.top": images,
            "observation.state": obs['eef_pos'],
            "task": tasks
        }
            # if proprios is not None:
            #     # BridgeSimplerAdapter expects obs["observation.state"] to be passed to preprocess_proprio
            #     # preprocess_proprio expects the dict containing "agent"
            #     element["observation.state"] = proprios[i]
            
        # LeRobotPolicyWrapper.select_action returns env_actions (numpy)
        
        if return_raw:
            actions, raw_actions = self.wrapper.select_action(element, return_raw=return_raw)
        else:
            actions = self.wrapper.select_action(element, return_raw=return_raw)

        
        # Ensure actions is [B, D] if T=1
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
            
        # actions = np.array(actions) # [B, action_dim]
        
        # Return dummy values for value and logprob
        values = torch.zeros(batch_size, 1).to(self.wrapper.device)
        logprobs = torch.zeros(batch_size, 1).to(self.wrapper.device)
        
        actions_tensor = torch.tensor(actions).to(self.wrapper.device)
        
        if return_raw:
            return values, actions_tensor, logprobs, raw_actions
        
        return values, actions_tensor, logprobs

    def prep_rollout(self):
        self.wrapper.model.eval()
        
    def prep_training(self):
        self.wrapper.model.train()
        
    def save(self, path):
        pass
