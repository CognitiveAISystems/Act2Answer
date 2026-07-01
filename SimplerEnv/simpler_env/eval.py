"""Small command-line entrypoint for Act2Answer VLA evaluation.

This module prepares ``simpler_env.run.Args`` for one model, selects a contiguous
slice of tasks from an Act2Answer asset set, runs one layout (noswap or swap), and
prints machine-readable markers for the shell wrappers and logs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from simpler_env.run import Args, Runner


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_PATH = REPO_ROOT / "ManiSkill" / "mani_skill" / "assets" / "carrot"


@dataclass(frozen=True)
class VLAConfig:
    vla_kind: str
    default_assets: str
    default_count: int
    default_buffer_inferbatch: int
    default_vla_path: str
    env_var_path: str | None = None
    default_unnorm_key: str = "sft"
    name_prefix: str | None = None

    @property
    def prefix(self) -> str:
        return self.name_prefix or self.vla_kind


VLA_CONFIGS: dict[str, VLAConfig] = {
    "pi0": VLAConfig(
        vla_kind="pi0",
        default_assets="test_colors",
        default_count=6,
        default_buffer_inferbatch=1,
        default_vla_path="juexzz/INTACT-pi0-finetune-bridge",
    ),
    "magma": VLAConfig(
        vla_kind="magma",
        default_assets="test_colors",
        default_count=6,
        default_buffer_inferbatch=6,
        default_vla_path="microsoft/Magma-8B",
    ),
    "openvla": VLAConfig(
        vla_kind="openvla",
        default_assets="test_colors",
        default_count=6,
        default_buffer_inferbatch=6,
        default_vla_path="gen-robot/openvla-7b-rlvla-sft_16k",
        env_var_path="VLA_PATH",
        default_unnorm_key="sft",
    ),
    "spatialvla": VLAConfig(
        vla_kind="spatialvla",
        default_assets="safe_school",
        default_count=9,
        default_buffer_inferbatch=9,
        default_vla_path="IPEC-COMMUNITY/spatialvla-4b-224-pt",
        env_var_path="SPATIALVLA_CKPT",
    ),
    "xiaomi": VLAConfig(
        vla_kind="xiaomi",
        default_assets="safe_school",
        default_count=9,
        default_buffer_inferbatch=9,
        default_vla_path="",
    ),
    "internvla": VLAConfig(
        vla_kind="internvla",
        default_assets="safe_school",
        default_count=9,
        default_buffer_inferbatch=9,
        default_vla_path=(
            "/home/jovyan/nkachaev/InternVLA-M1/playground/Pretrained_models/"
            "InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt"
        ),
        env_var_path="INTERNVLA_CKPT",
    ),
    "molmoact": VLAConfig(
        vla_kind="molmoact",
        default_assets="more_celeb_v2",
        default_count=4,
        default_buffer_inferbatch=4,
        default_vla_path="",
    ),
}


def _read_pairs(asset_path: Path, assets: str) -> list[dict]:
    pairs_path = asset_path / assets / "pairs.json"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Act2Answer asset set not found: {pairs_path}")
    return json.loads(pairs_path.read_text())


def _select_ids(total: int, start_id: int, count: int) -> list[int]:
    if start_id < 0:
        raise ValueError(f"--start-id must be non-negative, got {start_id}")
    end_id = total if count <= 0 else min(start_id + count, total)
    ids = list(range(start_id, end_id))
    if not ids:
        raise ValueError(f"No ids selected: start={start_id} count={count} total={total}")
    return ids


def _resolve_vla_path(config: VLAConfig, cli_path: str | None) -> str:
    if cli_path is not None:
        return cli_path
    if config.env_var_path:
        value = os.environ.get(config.env_var_path)
        if value:
            return value
    if config.vla_kind == "internvla":
        repo = os.environ.get("INTERNVLA_REPO")
        if repo:
            return str(
                Path(repo)
                / "playground"
                / "Pretrained_models"
                / "InternVLA-M1-Pretrain-RT-1-Bridge"
                / "checkpoints"
                / "steps_50000_pytorch_model.pt"
            )
    return config.default_vla_path


def build_runner_args(ns: argparse.Namespace, config: VLAConfig) -> Args:
    asset_path = Path(ns.asset_path).expanduser().resolve()
    pairs = _read_pairs(asset_path, ns.assets)
    ids = _select_ids(len(pairs), ns.start_id, ns.count)

    args = Args()
    args.env_id = "Act2AnswerV4-v1"
    args.seed = ns.seed
    args.name = ns.name or f"{config.prefix}-{ns.assets}-{'swap' if ns.do_swap else 'noswap'}"
    args.obj_set = ns.obj_set
    args.episode_len = ns.episode_len
    args.vla_kind = config.vla_kind
    args.vla_path = _resolve_vla_path(config, ns.vla_path)
    args.vla_unnorm_key = ns.vla_unnorm_key
    args.vla_load_path = ns.vla_load_path
    args.vla_lora_rank = ns.vla_lora_rank
    args.only_render = True
    args.render_info = ns.render_info
    args.assets = ns.assets
    args.do_swap = bool(ns.do_swap)
    args.buffer_inferbatch = ns.buffer_inferbatch
    args.buffer_minibatch = ns.buffer_minibatch
    args.archive_path = ns.archive_path
    args.asset_path = str(asset_path)
    args.output_dir = ns.output_dir
    args.ids = ids
    args.total_envs = len(pairs)
    args.shard_start = ids[0]
    args.shard_end = ids[-1] + 1
    args.num_envs = len(ids)
    args.init_grasp_steps = ns.init_grasp_steps
    args.hold_cube_steps = ns.hold_cube_steps
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Act2Answer VLA evaluation layout.")
    parser.add_argument("--vla", required=True, choices=sorted(VLA_CONFIGS))
    parser.add_argument("--assets", default=None)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--count", type=int, default=None, help="Number of tasks; <=0 means all.")
    parser.add_argument("--obj-set", default="test")
    parser.add_argument("--do-swap", action="store_true")
    parser.add_argument("--episode-len", type=int, default=80)
    parser.add_argument("--buffer-inferbatch", type=int, default=None)
    parser.add_argument("--buffer-minibatch", type=int, default=8)
    parser.add_argument("--vla-path", default=None)
    parser.add_argument("--vla-unnorm-key", default=None)
    parser.add_argument("--vla-load-path", default="")
    parser.add_argument("--vla-lora-rank", type=int, default=32)
    parser.add_argument("--asset-path", default=str(DEFAULT_ASSET_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--render-info", action="store_true")
    parser.add_argument("--init-grasp-steps", type=int, default=10)
    parser.add_argument("--hold-cube-steps", type=int, default=15)
    ns = parser.parse_args()

    config = VLA_CONFIGS[ns.vla]
    if ns.assets is None:
        ns.assets = config.default_assets
    if ns.count is None:
        ns.count = config.default_count
    if ns.buffer_inferbatch is None:
        ns.buffer_inferbatch = config.default_buffer_inferbatch
    if ns.vla_unnorm_key is None:
        ns.vla_unnorm_key = os.environ.get("UNNORM", config.default_unnorm_key)
    return ns


def main() -> None:
    start = time.monotonic()
    ns = parse_args()
    config = VLA_CONFIGS[ns.vla]
    args = build_runner_args(ns, config)

    print(
        f"SELECTED_IDS {args.ids[0]}..{args.ids[-1]} count={len(args.ids)} "
        f"total={args.total_envs} do_swap={args.do_swap}",
        flush=True,
    )
    print(f"VLA {args.vla_kind} path={args.vla_path}", flush=True)
    runner = Runner(args)
    print(f"OUTPUT_DIR {runner.save_dir}", flush=True)
    print(f"GLOB_DIR {runner.glob_dir}", flush=True)
    stats = runner.render(epoch=0, obj_set=args.obj_set)
    stats = {key: float(value) for key, value in stats.items()}
    print(f"FINAL_STATS {stats}", flush=True)
    print(f"EVAL_DONE_SECONDS {time.monotonic() - start:.3f}", flush=True)


if __name__ == "__main__":
    main()
