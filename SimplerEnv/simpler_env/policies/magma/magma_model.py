import numpy as np
from PIL import Image
import random
import torch
import torchvision
import json
import sys
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LogitsProcessor, LogitsProcessorList


class _CastFloat32(LogitsProcessor):
    """Cast logits to fp32 before generation's softmax/multinomial.
    bf16 softmax can overflow to inf/nan -> multinomial 'probability tensor contains nan'.
    Casting to fp32 keeps sampling (do_sample=True) numerically stable."""
    def __call__(self, input_ids, scores):
        # bf16 forward can emit nan/inf logits at some vocab positions -> softmax/multinomial nan.
        # Cast to fp32 and replace nan/+-inf so sampling stays on the (finite) real logits.
        scores = scores.float()
        return torch.nan_to_num(scores, nan=-1e4, posinf=50.0, neginf=-1e4)
from transforms3d.euler import euler2axangle


action_norm_stats = {
    "bridge_orig": {
        "mask": [True, True, True, True, True, True, False],
        "max": [
            0.41691166162490845,
            0.25864794850349426,
            0.21218234300613403,
            3.122201919555664,
            1.8618112802505493,
            6.280478477478027,
            1.0,
        ],
        "mean": [
            0.0002334194869035855,
            0.00013004911306779832,
            -0.00012762474943883717,
            -0.0001556558854645118,
            -0.0004039328487124294,
            0.00023557482927571982,
            0.5764579176902771,
        ],
        "min": [
            -0.4007510244846344,
            -0.13874775171279907,
            -0.22553899884223938,
            -3.2010786533355713,
            -1.8618112802505493,
            -6.279075622558594,
            0.0,
        ],
        "q01": [
            -0.02872725307941437,
            -0.04170349963009357,
            -0.026093858778476715,
            -0.08092105075716972,
            -0.09288699507713317,
            -0.20718276381492615,
            0.0,
        ],
        "q99": [
            0.028309678435325586,
            0.040855254605412394,
            0.040161586627364146,
            0.08192047759890528,
            0.07792850524187081,
            0.20382574498653397,
            1.0,
        ],
        "std": [
            0.009765930473804474,
            0.013689135201275349,
            0.012667362578213215,
            0.028534092009067535,
            0.030637972056865692,
            0.07691419124603271,
            0.4973701536655426,
        ],
    },
    "google_robot": {
        "mask": [True, True, True, True, True, True, False],
        "max": [
            2.9984593391418457,
            22.09052848815918,
            2.7507524490356445,
            1.570636510848999,
            1.5321086645126343,
            1.5691522359848022,
            1.0,
        ],
        "mean": [
            0.006987582892179489,
            0.006265917327255011,
            -0.01262515690177679,
            0.04333311319351196,
            -0.005756212864071131,
            0.0009130256366916001,
            0.5354204773902893,
        ],
        "min": [
            -2.0204520225524902,
            -5.497899532318115,
            -2.031663417816162,
            -1.569917917251587,
            -1.569892168045044,
            -1.570419430732727,
            0.0,
        ],
        "q01": [
            -0.22453527510166169,
            -0.14820013284683228,
            -0.231589707583189,
            -0.3517994859814644,
            -0.4193011274933815,
            -0.43643461108207704,
            0.0,
        ],
        "q99": [
            0.17824687153100965,
            0.14938379630446405,
            0.21842354819178575,
            0.5892666035890578,
            0.35272657424211445,
            0.44796681255102094,
            1.0,
        ],
        "std": [
            0.0692116990685463,
            0.05970962345600128,
            0.07353084534406662,
            0.15610496699810028,
            0.13164450228214264,
            0.14593800902366638,
            0.497110515832901,
        ],
    },
}

# microsoft/magma-8b-hf


class MagmaInference:
    def __init__(
        self,
        device_id: int,
        all_args,
        model_name = "microsoft/Magma-8B",
        action_scale=1.0,
        sticky_gripper_num_repeat=10,
        unnorm_key=None,
        sample=True,
    ):
        # if policy_setup == "widowx_bridge":
        #     self.unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
        # elif policy_setup == "google_robot":
        #     self.unnorm_key = "google_robot" if unnorm_key is None else unnorm_key

        self.unnorm_key = "bridge_orig"
        self.sticky_gripper_num_repeat = sticky_gripper_num_repeat  # Note: this is 1 for widowx_bridge in openvla and spatialvla, but 10 for magma. Is it right?

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.vla = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=f"cuda:{device_id}",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._fix_vision_layerscale(model_name)

        self.task_description = None

        self.policy_setup = "widowx_bridge"
        self.action_scale = action_scale
        self.sample = sample

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.action_norm_stats = action_norm_stats[self.unnorm_key]
        self.n_action_bins = 256
        self.vocab_size = self.processor.tokenizer.vocab_size
        self.bins = np.linspace(-1, 1, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def reset(self, task_description):
        self.task_description = task_description

    def get_action(self, x: dict, deterministic=True):
        images: torch.Tensor = x["image"]
        task_descriptions: list[str] = x["task_description"]
        convs = [
            [
                {
                    "role": "user",
                    "content": f"<image>\nWhat action should the robot take to {prompt}?",
                },
                {
                    "role": "system",
                    "content": "You are agent that can see, talk and act.",
                },
            ]
            for prompt in task_descriptions
        ]

        prompts = self.processor.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True,
        )

        if self.vla.config.mm_use_image_start_end:
            prompts = [prompt.replace("<image>", "<image_start><image><image_end>") for prompt in prompts]

        pil_images = [Image.fromarray(image.cpu().to(torch.uint8).numpy()) for image in images]  # uint8: PIL cannot build RGB from float32
        pil_images = [image.resize((256, 256)) for image in pil_images]

        inputs = self.processor(images=pil_images, texts=prompts, return_tensors='pt')
        # move to model device; BatchFeature.to(dtype) casts float tensors (pixel_values) to
        # bf16 and leaves integer input_ids untouched (same as step_one).
        inputs = inputs.to("cuda").to(torch.bfloat16)
        # Magma expects pixel_values (B, num_crops, C, H, W) and image_sizes (B, num_crops, 2).
        # The batched processor returns (B, C, H, W) and (B, 2); add the num_crops=1 axis
        # (step_one does the equivalent unsqueeze(0) for the single-image case).
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(1)
        inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(1)

        self.vla.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        with torch.inference_mode():
            output_ids = self.vla.generate(
                **inputs,
                temperature=0.7,
                do_sample=self.sample,           # sampling (teammate's working config)
                num_beams=1,
                max_new_tokens=1000,             # run to EOS; the 7 action tokens are the last ones
                use_cache=True,
                logits_processor=LogitsProcessorList([_CastFloat32()]),  # fp32 -> no bf16 multinomial NaN
            )
            # Robustly extract the 7 action tokens preceding EOS for EACH sequence. In batched
            # generation, sequences that emit EOS early are right-padded, so a fixed [:, -8:-1]
            # slice grabs pad tokens -> garbage actions. Find each sequence's first EOS in the
            # generated region and take the 7 tokens before it.
            eos_id = self.processor.tokenizer.eos_token_id
            in_len = inputs["input_ids"].shape[1]
            gen = output_ids[:, in_len:].cpu()
            rows = []
            for row in gen:
                pos = (row == eos_id).nonzero()
                e = int(pos[0]) if len(pos) > 0 else row.shape[0]
                seg = row[max(0, e - 7):e]
                if seg.shape[0] < 7:  # pad-safe fallback
                    seg = output_ids[0, -8:-1].cpu()
                rows.append(seg)
            action_ids = torch.stack(rows)
        
        # TODO Replace this with an actual batch computation, replace np with torch
        def decode_actions(ids: torch.Tensor) -> torch.Tensor:
            predicted_action_ids = np.array(ids).astype(np.int64)
            discretized_actions = self.vocab_size - predicted_action_ids
            discretized_actions = np.clip(
                discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
            )
            normalized_actions = self.bin_centers[discretized_actions]

            # Unnormalize actions
            mask = self.action_norm_stats.get(
                "mask", np.ones_like(self.action_norm_stats["q01"], dtype=bool)
            )
            action_high, action_low = np.array(self.action_norm_stats["q99"]), np.array(
                self.action_norm_stats["q01"]
            )
            return torch.from_numpy(np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            ))
        
        raw_actions = torch.stack([decode_actions(row) for row in list(action_ids)])

        return self._process_action(raw_actions)

    # This is taken from spatialvla but matches what magma does
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

        action = {
            k: torch.tensor(v).to(torch.float32) for k, v in action.items()
        }  # to float32 ?

        action = torch.cat(
            [action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1
        )

        # to tpdv
        action = action.to(raw_actions.device)

        return action

    def step_one(self, image: np.ndarray, task_description: str | None = None):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        convs = [
            {
                "role": "user",
                "content": f"<image>\nWhat action should the robot take to {self.task_description}?",
            },
        ]
        convs = [
            {
                "role": "system",
                "content": "You are agent that can see, talk and act.",
            },
        ] + convs
        prompt = self.processor.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )
        if self.vla.config.mm_use_image_start_end:
            prompt = prompt.replace("<image>", "<image_start><image><image_end>")

        image = Image.fromarray(image)

        # resize image to 256x256
        # image = image.resize((512, 512))
        image = image.resize((256, 256))
        inputs = self.processor(images=image, texts=prompt, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
        inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
        inputs = inputs.to("cuda").to(torch.bfloat16)
        # Magma expects pixel_values (B, num_crops, C, H, W) and image_sizes (B, num_crops, 2).
        # The batched processor returns (B, C, H, W) and (B, 2); add the num_crops=1 axis
        # (step_one does the equivalent unsqueeze(0) for the single-image case).
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(1)
        inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(1)

        self.vla.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        with torch.inference_mode():
            output_ids = self.vla.generate(
                **inputs,
                temperature=0.7,
                do_sample=self.sample,
                num_beams=1,
                max_new_tokens=1000,
                use_cache=True,
            )
            action_ids = output_ids[0, -8:-1].cpu().tolist()

            if random.random() < 0.1:
                print("Action ids", action_ids)

        predicted_action_ids = np.array(action_ids).astype(np.int64)
        discretized_actions = self.vocab_size - predicted_action_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        mask = self.action_norm_stats.get(
            "mask", np.ones_like(self.action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(self.action_norm_stats["q99"]), np.array(
            self.action_norm_stats["q01"]
        )
        raw_actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(
                raw_actions[6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }
        # print(raw_action)
        # Process raw_action to obtain the action for the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta

        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def _fix_vision_layerscale(self, model_name):
        """The microsoft/Magma-8B checkpoint stores the ConvNeXt vision-tower LayerScale params
        as `...blocks.M.weight`, but the installed timm/open_clip name them `...blocks.M.gamma`,
        so HF `from_pretrained` cannot match them and leaves them at the ~1e-5 init. That collapses
        the vision tower output and the model emits gibberish (no actions). Re-map those checkpoint
        keys (`...blocks.M.weight` -> `...blocks.M.gamma`) and load the real vision weights."""
        import glob, re
        try:
            from safetensors import safe_open
            from huggingface_hub import snapshot_download
            snap = model_name if os.path.isdir(model_name) else snapshot_download(model_name, local_files_only=True)
            model_keys = set(self.vla.state_dict().keys())
            remap = {}
            for fp in glob.glob(os.path.join(snap, "*.safetensors")):
                with safe_open(fp, framework="pt") as st:
                    for k in st.keys():
                        if "vision_tower" not in k:
                            continue
                        nk = k[:-6] + "gamma" if re.search(r"blocks\.\d+\.weight$", k) else k
                        if nk in model_keys:
                            remap[nk] = st.get_tensor(k)
            if remap:
                self.vla.load_state_dict(remap, strict=False)
            print(f"MAGMA vision-tower LayerScale remap: loaded {len(remap)} vision weights", flush=True)
        except Exception as e:
            print(f"MAGMA vision-tower remap FAILED ({e}); vision may be untrained", flush=True)

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.train()
