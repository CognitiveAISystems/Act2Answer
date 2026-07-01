import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, BatchFeature, AutoModel, AutoProcessor


class ActionEnsembler:
    def __init__(self, pred_action_horizon, action_ensemble_temp=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_ensemble_temp = action_ensemble_temp
        self.action_history = deque(maxlen=self.pred_action_horizon)

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.action_history
                    )
                ]
            )
        # if temp > 0, more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.action_ensemble_temp * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


class SpatialVLAPolicy:
    def __init__(
        self,
        device_id: int,
        all_args,
        saved_model_path: str = "IPEC-COMMUNITY/spatialvla-4b-224-pt",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        exec_horizon: int = 1,
        image_size: "list[int]" = [224, 224],
        action_scale: float = 1.0,
        action_ensemble_temp: float = -0.8,
    ):

        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig/1.0.0" if unnorm_key is None else unnorm_key
            action_ensemble = True
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = (
                "fractal20220817_data/0.1.0" if unnorm_key is None else unnorm_key
            )
            action_ensemble = True
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.device_id = device_id

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.processor = AutoProcessor.from_pretrained(
            saved_model_path, trust_remote_code=True
        )
        self.vla = (
            AutoModel.from_pretrained(
                saved_model_path,
                # attn_implementation='flash_attention_2', # TODO FIX THIS
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=f'cuda:{self.device_id}'
            )
            .eval()
            # .to(device=f'cuda:{self.device_id}')
        )

        self.image_size = image_size
        self.action_scale = action_scale
        self.obs_horizon = (
            self.processor.num_obs_steps - 1
        ) * self.processor.obs_delta + 1
        self.obs_interval = self.processor.obs_delta
        self.pred_action_horizon = self.processor.action_chunk_size
        self.image_history = deque(maxlen=self.obs_horizon)
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp
            )
        else:
            self.action_ensembler = None

        self.task = None
        self.task_description = None

        self.args = all_args
        self.device_id = device_id
        self.tpdv = dict(
            device=torch.device("cuda:" + str(device_id)), dtype=torch.bfloat16
        )
        self.tpdv_vn = dict(
            device=torch.device("cuda:" + str(device_id)), dtype=torch.float32
        )
        self.action_scale = 1.0

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4
        assert images.shape[3] == 3
        assert images.dtype == torch.uint8

        assert isinstance(task_description, list)
        assert isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        # # prompt
        # if action is None:
        #     task_prompt = [
        #         f"In: What action should the robot take to {t.lower()}?\nOut: "
        #         for t in task_description
        #     ]
        # else:
        #     task_prompt = 

        task_prompt = task_description # This should be in-distribution for spatialvla

        inputs = self.processor(
            list(images.to("cpu").to(torch.float32)),
            task_prompt,
            unnorm_key=self.unnorm_key,
            return_tensors="pt",
            do_normalize=False,
            padding=True,
        )

        return inputs

    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        action_scale = 1.0

        assert self.policy_setup == "widowx_bridge"

        raw_action = {
            "world_vector": raw_actions[:, :3],
            "rotation_delta": raw_actions[:, 3:6],
            "open_gripper": raw_actions[:, 6:7],  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * action_scale  # [B, 3]
        action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0  # [B, 1]

        # origin euler
        action["rot_axangle"] = raw_action["rotation_delta"]

        action = torch.cat(
            [action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1
        )

        # to tpdv
        action = action.to(raw_actions.device)

        return action

    def get_action(self, x: dict, deterministic) -> torch.Tensor:
        # temperature = (
        #     self.args.vla_temperature_eval
        #     if deterministic
        #     else self.args.vla_temperature
        # )
        # do_sample = temperature != 0.0  # FIXME FIX THIS

        features = self._preprocess_obs(x)

        with torch.no_grad():
            generation_outputs = self.vla.predict_action(features)

            raw_actions = np.stack(
                [
                    self.processor.decode_actions(
                        generation_outputs=x[None, ...],
                        unnorm_key=self.unnorm_key,
                    )["actions"][0, :]
                    for x in list(generation_outputs)
                ]
            )

        action = torch.from_numpy(raw_actions).cpu()

        return self._process_action(action)

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.train()
