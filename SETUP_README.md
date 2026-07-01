# Act2Answer Setup Guide

This guide assumes a fresh machine: no conda envs, no Xiaomi/InternVLA/MolmoAct/RL4VLA repos, and no
model weights already cached. Each model stack gets its own conda env because the dependency versions
conflict.

## 0. Prerequisites

- Linux with an NVIDIA GPU and working CUDA driver.
- Conda or Miniconda available at `/opt/conda` or configured with `CONDA_ROOT`.
- `git`, `tmux`, and enough disk for model weights.
- Optional but recommended: `huggingface-cli login`.

Clone Act2Answer:

```bash
git clone <this-repo-url> Act2Answer
cd Act2Answer
```

## 1. Shared Configuration

All scripts source `scripts/env.sh`. Important variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `A2A_EXTERNAL_DIR` | parent of `Act2Answer` | where external repos are cloned |
| `CONDA_ENVS_DIR` | `/home/jovyan/.mlspace/envs` | conda env location |
| `A2A_LOG_DIR` | `$REPO_ROOT/logs` | eval/setup logs |
| `A2A_OUTPUT_DIR` | `$REPO_ROOT/outputs` | videos, config YAML, stats YAML |
| `PI0_DEPS_ROOT` | `$A2A_EXTERNAL_DIR/RL4VLA` | pi0/RL4VLA checkout |
| `XIAOMI_REPO` | `$A2A_EXTERNAL_DIR/Xiaomi-Robotics-0` | Xiaomi server repo |
| `INTERNVLA_REPO` | `$A2A_EXTERNAL_DIR/InternVLA-M1` | InternVLA server repo |
| `MOLMOACT_REPO` | `$A2A_EXTERNAL_DIR/molmoact2` | MolmoAct2 server repo |
| `SPATIALVLA_CKPT` | `IPEC-COMMUNITY/spatialvla-4b-224-pt` | HF id or local checkpoint |
| `MOLMOACT_CKPT` | `$A2A_EXTERNAL_DIR/molmoact2_ckpt` | MolmoAct2 local weights |

Override any of these before running setup/eval, for example:

```bash
export CONDA_ENVS_DIR=$HOME/.conda/envs
export A2A_EXTERNAL_DIR=$HOME/src/act2answer_external
```

## 2. Clone External Repos

```bash
bash scripts/setup/clone_external_repos.sh
```

This clones:

- `XiaomiRobotics/Xiaomi-Robotics-0` for the Xiaomi policy server.
- `InternRobotics/InternVLA-M1` for the InternVLA policy server.
- `allenai/molmoact2` for the MolmoAct2 policy server.

Model weights are not committed here. HF-hosted weights download on first run unless you point the
corresponding env variable at a local checkpoint.

## 3. Build Conda Envs From Scratch

The requirements files are curated direct-dependency lists, not full freezes. Setup scripts install
CUDA-specific torch/flash-attn wheels first, then editable local packages.

In-process eval envs:

```bash
bash scripts/setup/setup_spatialvla_env.sh
bash scripts/setup/setup_magma_env.sh
bash scripts/setup/setup_openvla_env.sh
bash scripts/setup/setup_pi0_env.sh
```

Server/client envs:

```bash
bash scripts/setup/setup_eval_client_env.sh
bash scripts/setup/setup_xiaomi_server_env.sh
bash scripts/setup/setup_internvla_env.sh
bash scripts/setup/setup_molmoact2_server_env.sh
```

You only need the envs for the models you plan to run. For example, SpatialVLA only needs
`setup_spatialvla_env.sh`; Xiaomi needs both `setup_eval_client_env.sh` and
`setup_xiaomi_server_env.sh`.

## 4. Run In-Process Models

Each wrapper runs noswap and swap:

```bash
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_spatialvla.sh
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_magma.sh
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_openvla.sh
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_pi0.sh
```

OpenVLA examples:

```bash
VLA_PATH=gen-robot/openvla-7b-rlvla-sft_16k UNNORM=sft \
  ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_openvla.sh

VLA_PATH=gen-robot/openvla-7b-rlvla-rl UNNORM=bridge_orig \
  ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_openvla.sh
```

## 5. Run Server-Based Models

Use tmux for long-lived servers.

Xiaomi:

```bash
tmux new-session -s xiaomi_srv "GPU=0 bash scripts/servers/run_xiaomi_policy_server.sh"
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_xiaomi.sh
```

InternVLA-M1:

```bash
tmux new-session -s internvla_srv "GPU=0 bash scripts/servers/run_internvla_server.sh"
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_internvla.sh
```

MolmoAct2:

```bash
tmux new-session -s molmoact_srv "GPU=0 bash scripts/servers/run_molmoact2_policy_server.sh"
ASSETS=test_colors COUNT=4 EVAL_GPU=3 bash scripts/eval_molmoact.sh
```

`scripts/eval_all3.sh <asset> [count] [eval_gpu]` starts/checks Xiaomi and InternVLA servers, then
runs Xiaomi, InternVLA, and SpatialVLA and prints a compact score table.

## 6. Outputs

Evaluation creates local files only:

- `$A2A_OUTPUT_DIR/<run-name>/glob/config.yaml`
- `$A2A_OUTPUT_DIR/<run-name>/glob/vis_0_<obj-set>/video_*.mp4`
- `$A2A_OUTPUT_DIR/<run-name>/glob/vis_0_<obj-set>/stats.yaml`
- `$A2A_LOG_DIR/<model>_<asset>_eval.log`
