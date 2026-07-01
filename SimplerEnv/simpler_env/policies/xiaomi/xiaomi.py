import math
import os
import time
import pickle
import socket
import struct

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import sys
from pathlib import Path
import json
import collections
import dataclasses
import logging
import pathlib
import time
import hashlib
from typing import Literal
import pandas as pd
from collections import deque

import torch
import torchvision.transforms.functional as F
import numpy as np
import imageio
from PIL import Image
import tyro
from transforms3d.euler import euler2axangle, mat2euler, quat2mat
from transforms3d.quaternions import mat2quat

class Client:
    def __init__(self, host="localhost", port=10086):
        self.host = host
        self.port = port
        self._connect_with_retry(max_retries=None, retry_interval=1)
        print(f"Client connected to server at {self.host}:{self.port}.")

    def _connect_with_retry(self, max_retries=None, retry_interval=1):
        """Connect with retry logic. max_retries=None implies infinite."""
        retry_count = 0
        while True:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                return
            except (ConnectionRefusedError, socket.error) as e:
                retry_count += 1
                time.sleep(retry_interval)
                if max_retries is not None and retry_count >= max_retries:
                    raise ConnectionError(f"Failed to connect to {self.host}:{self.port} after {retry_count} retries: {e}") from e

    def _send_with_length_prefix(self, data):
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        self.client_socket.sendall(struct.pack(">I", len(serialized)) + serialized)

    def _recv_with_length_prefix(self):
        len_data = self.client_socket.recv(4)
        if not len_data or len(len_data) < 4:
            raise ConnectionError("Failed to receive response length prefix.")
        data_len = struct.unpack(">I", len_data)[0]
        data = b""
        while len(data) < data_len:
            packet = self.client_socket.recv(data_len - len(data))
            if not packet:
                raise ConnectionError("Connection closed while receiving response.")
            data += packet
        return pickle.loads(data)

    def __call__(self, **data):
        self._send_with_length_prefix(data)
        response = self._recv_with_length_prefix()

        return response

    def close(self):
        self.client_socket.close()
        print("Client connection closed.")

def hash_data_to_seed(data, max_bytes=4):
    """
    Computes a stable SHA256 hash for a dictionary containing Numpy arrays
    and PIL Images, ensuring consistency across different times and machines.
    """

    def custom_encoder(obj):
        """
        Serializer for non-JSON serializable objects.
        """
        # Handle Numpy arrays
        if isinstance(obj, np.ndarray):
            return {"__type__": "numpy", "dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tobytes().hex()}

        # Handle PIL Images
        if isinstance(obj, Image.Image):
            # Calculate a digest of the raw image bytes to keep the JSON small
            img_hash = hashlib.md5(obj.tobytes()).hexdigest()
            return {"__type__": "PIL.Image", "mode": obj.mode, "size": obj.size, "content_hash": img_hash}

        # Handle Sets
        if isinstance(obj, set):
            return sorted(list(obj))

        # Let it crash if type is unknown (as requested, no try-except)
        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    # Generate canonical JSON string
    json_str = json.dumps(
        data,
        default=custom_encoder,
        sort_keys=True,  # Enforce deterministic key order
        separators=(",", ":"),  # Remove whitespace for compact representation
        ensure_ascii=False,  # Preserve non-ASCII characters
    )

    hex_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    seed_int = int(hex_hash, 16)
    if max_bytes > 0:
        seed_int = seed_int % (2 ** (8 * max_bytes))

    return seed_int


def preprocess_proprio_fractal(proprio: np.ndarray) -> np.ndarray:
    """convert wxyz quat from simpler to xyzw used in fractal"""
    gripper_quat_wxyz = proprio[3:7]
    gripper_rotm = quat2mat(gripper_quat_wxyz)
    gripper_xyzw = mat2quat(gripper_rotm)[[1, 2, 3, 0]]
    gripper_width = proprio[7]  # from simpler, 0 for close, 1 for open
    gripper_closedness = 1 - gripper_width

    raw_proprio = np.concatenate(
        (
            proprio[:3],
            gripper_xyzw,
            [gripper_closedness],
        )
    )
    return raw_proprio


def postprocess_gripper_fractal(action: float) -> float:
    # trained with [0, 1], 0 for close, 1 for open
    # convert to -1 open, 1 close for simpler
    action = (action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

    # without sticky
    relative_gripper_action = -action
    relative_gripper_action = np.clip(relative_gripper_action, -1, 1)
    return relative_gripper_action


def preprocess_proprio_bridge(proprio: np.ndarray) -> np.ndarray:
    # convert ee rotation to the frame of top-down
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    rm_bridge = quat2mat(proprio[3:7])
    rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
    gripper_openness = proprio[7]
    raw_proprio = np.concatenate(
        [
            proprio[:3],
            rpy_bridge_converted,
            [gripper_openness],
        ]
    )
    return raw_proprio

def postprocess_gripper_bridge(action: float) -> float:
    # trained with [0, 1], 0 for close, 1 for open
    # convert to -1 close, 1 open for simpler
    action_gripper = 2.0 * (action > 0.5) - 1.0
    return action_gripper

def client_process(task_id, state, base_obs: Image.Image, instruction):
    if "bridge" in task_id:
        base_obs = base_obs.resize((256, 256))
        instruction = f"<|im_start|>user\nThe following observations are captured from multiple views.\n# Base View\n<|vision_start|><|image_pad|><|vision_end|>\nGenerate robot actions for the task:\n{instruction} /no_cot<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n"
        zero_state = np.zeros_like(state)[..., -1:]
        state = np.concatenate([state[..., :-1], zero_state, state[..., -1:]], axis=-1)
    else:
        base_obs = base_obs.resize((320, 256))
        instruction = f"<|im_start|>user\nThe following observations are captured from multiple views.\n# Ego View\n<|vision_start|><|image_pad|><|vision_end|>\nGenerate robot actions for the task:\n{instruction} /no_cot<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n"

    # center crop
    crop_ratio = 0.95
    h, w = base_obs.size[1], base_obs.size[0]
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    crop_y = (h - crop_h) // 2
    crop_x = (w - crop_w) // 2
    base_obs = F.crop(base_obs, crop_y, crop_x, crop_h, crop_w)
    assert (base_obs.size[0], base_obs.size[1]) == (
        crop_w,
        crop_h,
    ), f"{(base_obs.size[0], base_obs.size[1])} != ({crop_w}, {crop_h})"
    base_obs = F.resize(base_obs, (h, w))
    assert (base_obs.size[0], base_obs.size[1]) == (w, h), f"{(base_obs.size[0], base_obs.size[1])} != ({w}, {h})"

    state = torch.from_numpy(state)[None, None]
    state = torch.nn.functional.pad(state, (0, 32 - state.shape[-1]))
    model_inputs = {
        "task_id": task_id,
        "state": state.numpy(),
        "language": instruction,
        "base": base_obs,
    }

    return model_inputs

class XiaomiRoboticsPolicy:
    def __init__(
        self,
        replan_steps: int = 4,
        # unnorm_key: Optional[str] = None,
        # policy_setup: str = "widowx_bridge",
        # horizon: int = 0,
        # action_ensemble_horizon: Optional[int] = None,
        # image_size: list[int] = [224, 224],
        # action_scale: float = 1.0,
        # cfg_scale: float = 1.5,
        # use_ddim: bool = True,
        # num_ddim_steps: int = 10,
        # action_ensemble = False,
        # adaptive_ensemble_alpha = 0.1,
    ) -> None:
        self.client = Client()
        self.replan_steps = replan_steps
        self.task_id = os.environ.get("XIAOMI_TASK_ID", "bridge_delta")
        print(f"XiaomiRoboticsPolicy task_id={self.task_id}")
        
        self.action_plans = []
        self.task_descriptions = []
    
    def prep_rollout(self):
        pass

    def reset(self, task_descriptions: list[str]) -> None:
        self.action_plans = [deque() for _ in task_descriptions]
        self.task_descriptions = task_descriptions
    
    def get_action(self, obs, _deterministic):
        return self.step(obs['image'], obs['task_description'], obs['proprio'])
    
    def compute_plan(self, images, task_descriptions, proprio):
        # TODO This would be much faster batched
        for (image, instruction, pr, i) in zip(images, task_descriptions, proprio, range(len(images))):
            base_obs = Image.fromarray(image.cpu().numpy(), mode="RGB")
            task_id = self.task_id
            if "fractal" in task_id:
                state = preprocess_proprio_fractal(pr["agent"]["eef_pos"].cpu().numpy())
            else:
                state = preprocess_proprio_bridge(pr["agent"]["eef_pos"].cpu().numpy())
            instruction = instruction[0].upper() + instruction[1:] + "."
            model_inputs = client_process(task_id, state, base_obs, instruction)
            temp_seed = hash_data_to_seed(model_inputs)
            model_inputs["seed"] = temp_seed
            action_chunk = self.client(**model_inputs)[0]
            assert (
                self.replan_steps <= action_chunk.shape[0]
            ), f"Replan steps must be less than or equal to the number of steps in the action chunk. {self.replan_steps} > {action_chunk.shape[0]}"

            action_chunk = action_chunk[: self.replan_steps, :7].cpu().numpy()
            self.action_plans[i] = deque()
            self.action_plans[i].extend(action_chunk)

    def step(
        self, images, task_descriptions, proprio, *args, **kwargs
    ):
        # """
        # Input:
        #     image: np.ndarray of shape (H, W, 3), uint8
        #     task_description: Optional[str], task description; if different from previous task description, policy state is reset
        # Output:
        #     raw_action: dict; raw policy action output
        #     action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
        #         - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
        #         - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
        #         - 'gripper': np.ndarray of shape (1,), gripper action
        #         - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        # """
        
        if not self.action_plans:
            self.reset(task_descriptions)

        # print(f"{self.task_descriptions=}")
        # print("\n\n")
        # print(f"{task_descriptions=}")
        assert task_descriptions == self.task_descriptions

        if any(len(plan) == 0 for plan in self.action_plans):
            self.compute_plan(images, task_descriptions, proprio)
        
        actions = []
        
        for i in range(len(images)):
            raw_action = self.action_plans[i].popleft()
            roll, pitch, yaw = raw_action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle

            if "fractal" in self.task_id:
                action_gripper = postprocess_gripper_fractal(raw_action[-1])
            else:
                action_gripper = postprocess_gripper_bridge(raw_action[-1])

            action = np.concatenate(
                [
                    raw_action[:3],
                    action_rotation_axangle,
                    [action_gripper],
                ]
            )

            actions.append(action)
        
        return torch.tensor(np.stack(actions))
