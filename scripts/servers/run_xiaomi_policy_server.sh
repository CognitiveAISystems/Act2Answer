#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"

LOG=${A2A_LOG_DIR}/xiaomi_policy_server.log
ENV_PATH="${CONDA_ENVS_DIR}/mibot"
REPO=${XIAOMI_REPO}
MODEL=${MODEL:-XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX}
PORT=${PORT:-10086}
GPU=${GPU:-0}

exec > >(tee -a "$LOG") 2>&1

echo "START_XIAOMI_POLICY_SERVER $(date)"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
cd "$REPO"

CUDA_VISIBLE_DEVICES="$GPU" python deploy/server.py --model "$MODEL" --host localhost --port "$PORT"
