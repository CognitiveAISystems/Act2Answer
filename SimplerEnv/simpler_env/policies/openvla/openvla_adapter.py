"""Adapt RL4VLA's OpenVLAPolicy to the SimplerEnv run.py contract (get_action -> (B,7)
continuous action tensor).

IMPORTANT: OpenVLAPolicy.get_action (via predict_action_batch) returns *raw action TOKEN IDs*
(integers in [32000-256, 32000)), NOT continuous actions. In the original RL4VLA pipeline the
*environment wrapper* decodes those token ids (it is constructed with `unnorm_state`). Act2Answer_CODE's
env instead expects already-continuous actions (like pi0/magma feed), so we must decode here -- an
exact port of RL4VLA SimplerEnv's `_process_action`:
  dact = 32000 - token_id ; dact = clip(dact-1, 0, 254) ; norm = bin_centers[dact]
  raw = where(mask, 0.5*(norm+1)*(q99-q01)+q01, norm)
  action = [world_vector(3), rotation_delta(3) as rot_axangle, gripper = 2*(open>0.5)-1]
Without this decode, token ids (~31900) are applied as giant pose deltas and the arm sprawls."""
import numpy as np
import torch
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy


class OpenVLAInference:
    def __init__(self, all_args, device_id):
        self._p = OpenVLAPolicy(all_args, device_id)
        self.unnorm_state = self._p.vla.get_action_stats(all_args.vla_unnorm_key)
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0

    def prep_rollout(self):
        self._p.prep_rollout()

    def _decode(self, token_ids):
        pact = token_ids.cpu().numpy()                       # [B, 7] action token ids
        dact = 32000 - pact
        dact = np.clip(dact - 1, a_min=0, a_max=254)
        normalized = np.asarray([self.bin_centers[da] for da in dact])  # [B, 7] in [-1,1]
        s = self.unnorm_state
        mask = np.asarray(s.get("mask", np.ones_like(s["q01"], dtype=bool))).reshape(1, -1)
        high = np.array(s["q99"]).reshape(1, -1)
        low = np.array(s["q01"]).reshape(1, -1)
        raw = np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
        world_vector = raw[:, :3]
        rot_axangle = raw[:, 3:6]                              # raw euler used as axangle (RL4VLA convention)
        gripper = 2.0 * (raw[:, 6:7] > 0.5) - 1.0
        action = np.concatenate([world_vector, rot_axangle, gripper], axis=1)
        return torch.tensor(action, dtype=torch.float32, device=token_ids.device)

    def get_action(self, obs, deterministic=True):
        _values, token_ids, _logprobs = self._p.get_action(obs, deterministic)
        return self._decode(token_ids)
