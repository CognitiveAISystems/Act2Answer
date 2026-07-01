"""MolmoAct2 SimplerEnv client policy (mirror of XiaomiRoboticsPolicy).

Thin client over the MolmoAct2 SimplerEnv inference server
(`molmoact2/examples/simpler/host_server_simpler.py`, json_numpy /act endpoint).

The base `allenai/MolmoAct2` checkpoint with norm_tag=widowx_bridge emits 7-D
delta-EEF actions [x, y, z, roll, pitch, yaw, gripper] (euler rotation,
gripper in [0, 1]) -- the same convention the Xiaomi bridge path uses -- which we
convert to SimplerEnv's [world_vector(3), rot_axangle(3), gripper(1)].
"""

import os
import time
from collections import deque

import numpy as np
import torch
import requests
import json_numpy
from transforms3d.euler import euler2axangle, mat2euler
from transforms3d.quaternions import quat2mat

json_numpy.patch()


def preprocess_proprio_bridge(proprio: np.ndarray) -> np.ndarray:
    """eef_pos [x,y,z, qw,qx,qy,qz, gripper_openness] -> [x,y,z, roll,pitch,yaw, gripper].

    Matches simpler_env.policies.xiaomi.preprocess_proprio_bridge: rotates the EE
    orientation into the top-down frame before converting to euler.
    """
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    rm_bridge = quat2mat(proprio[3:7])
    rpy = mat2euler(rm_bridge @ default_rot.T)
    gripper_openness = proprio[7]
    return np.concatenate([proprio[:3], rpy, [gripper_openness]])


def build_bridge_state(eef_pos: np.ndarray) -> np.ndarray:
    """Build the 8-D widowx_bridge state [x,y,z, roll,pitch,yaw, pad, gripper]."""
    s7 = preprocess_proprio_bridge(eef_pos)  # [x,y,z, roll,pitch,yaw, gripper]
    return np.concatenate([s7[:6], [0.0], s7[6:]]).astype(np.float32)


def postprocess_gripper_bridge(action: float) -> float:
    # model gripper trained in [0, 1] (1=open); convert to simpler -1=close, 1=open.
    return 2.0 * (action > 0.5) - 1.0


class Client:
    """json_numpy HTTP client for the MolmoAct2 /act server."""

    def __init__(self, host="localhost", port=8000, timeout=180.0):
        self.url = f"http://{host}:{port}/act"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False  # ignore env http(s)_proxy for localhost
        self._wait_for_server()
        print(f"MolmoAct client connected to {self.url}")

    def _wait_for_server(self, max_wait=600, interval=2.0):
        deadline = time.time() + max_wait
        while True:
            try:
                r = self.session.get(self.url, timeout=5)
                if r.status_code == 200:
                    print(f"Server health: {r.text[:200]}", flush=True)
                    return
                else:
                    print(f"_wait_for_server: status {r.status_code}", flush=True)
            except Exception as e:
                print(f"_wait_for_server: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if time.time() > deadline:
                raise ConnectionError(f"MolmoAct server at {self.url} not reachable")
            time.sleep(interval)

    def __call__(self, image, instruction, state, num_steps):
        payload = {
            "image": np.asarray(image, dtype=np.uint8),
            "instruction": str(instruction),
            "state": np.asarray(state, dtype=np.float32),
            "num_steps": int(num_steps),
        }
        r = self.session.post(
            self.url,
            headers={"Content-Type": "application/json"},
            data=json_numpy.dumps(payload),
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"MolmoAct server error {r.status_code}: {r.text[:500]}")
        data = r.json()
        actions = data["actions"] if isinstance(data, dict) and "actions" in data else data
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        return actions


class MolmoActPolicy:
    def __init__(self, replan_steps: int = None) -> None:
        host = os.environ.get("MOLMOACT_HOST", "localhost")
        port = int(os.environ.get("MOLMOACT_PORT", "8000"))
        self.client = Client(host=host, port=port)
        self.norm_tag = os.environ.get("MOLMOACT_NORM_TAG", "widowx_bridge")
        self.num_steps = int(os.environ.get("MOLMOACT_NUM_STEPS", "10"))
        if replan_steps is None:
            replan_steps = int(os.environ.get("MOLMOACT_REPLAN_STEPS", "5"))
        self.replan_steps = replan_steps
        print(
            f"MolmoActPolicy norm_tag={self.norm_tag} num_steps={self.num_steps} "
            f"replan_steps={self.replan_steps}"
        )
        self.action_plans = []
        self.task_descriptions = []

    def prep_rollout(self):
        pass

    def reset(self, task_descriptions):
        self.action_plans = [deque() for _ in task_descriptions]
        self.task_descriptions = task_descriptions

    def get_action(self, obs, _deterministic):
        return self.step(obs["image"], obs["task_description"], obs["proprio"])

    def compute_plan(self, images, task_descriptions, proprio):
        for image, instruction, pr, i in zip(
            images, task_descriptions, proprio, range(len(images))
        ):
            img = image.cpu().numpy().astype(np.uint8)  # (H, W, 3)
            state = build_bridge_state(pr["agent"]["eef_pos"].cpu().numpy())
            action_chunk = self.client(img, instruction, state, self.num_steps)  # (N, 7)
            n = min(self.replan_steps, action_chunk.shape[0])
            self.action_plans[i] = deque()
            self.action_plans[i].extend(action_chunk[:n, :7])

    def step(self, images, task_descriptions, proprio, *args, **kwargs):
        if not self.action_plans:
            self.reset(task_descriptions)
        assert task_descriptions == self.task_descriptions

        if any(len(plan) == 0 for plan in self.action_plans):
            self.compute_plan(images, task_descriptions, proprio)

        actions = []
        for i in range(len(images)):
            raw_action = self.action_plans[i].popleft()
            roll, pitch, yaw = raw_action[3:6]
            ax, angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = ax * angle
            action_gripper = postprocess_gripper_bridge(raw_action[-1])
            action = np.concatenate(
                [raw_action[:3], action_rotation_axangle, [action_gripper]]
            )
            actions.append(action)

        return torch.tensor(np.stack(actions))
