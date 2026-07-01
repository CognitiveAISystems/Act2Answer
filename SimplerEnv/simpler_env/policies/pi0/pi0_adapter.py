"""Adapt Act2Answer's PI0Policy (PPO-style (values, actions, logprobs), expects
obs['pi_0']={'eef_pos':...}) to SimplerEnv run.py (get_action -> action tensor;
obs['pi_0'] arrives as the per-batch eef_pos tensor after _get_action's slicing).

PI0 is an action-chunking flow-matching policy: select_action returns a chunk of
shape (B, T, 7). The SimplerEnv runner applies one (B, 7) action per env.step, so
we run receding-horizon: take the first action of the predicted chunk each step."""
from simpler_env.policies.pi0.pi0_model import PI0Policy


class Pi0Inference:
    def __init__(self, all_args, device_id):
        self._p = PI0Policy(all_args, device_id)

    def prep_rollout(self):
        self._p.prep_rollout()

    def get_action(self, obs, deterministic=True):
        obs = dict(obs)
        pi0 = obs.get("pi_0")
        if pi0 is not None and not isinstance(pi0, dict):
            obs["pi_0"] = {"eef_pos": pi0}
        _values, actions, _logprobs = self._p.get_action(obs, deterministic)
        # pi0 returns an action chunk (B, T, 7); the env wants a single (B, 7).
        if hasattr(actions, "ndim") and actions.ndim == 3:
            actions = actions[:, 0, :]
        return actions
