#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true
LOG=${A2A_LOG_DIR}/internvla_server.log
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENVS_DIR}/internvla"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=0
cd ${INTERNVLA_REPO}
CKPT=./playground/Pretrained_models/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt
: > "$LOG"; exec >> "$LOG" 2>&1
echo "START_INTERNVLA_SERVER $(date -u) gpu=${GPU:-3}"
CUDA_VISIBLE_DEVICES=${GPU:-3} python deployment/model_server/server_policy_M1.py \
  --ckpt_path "$CKPT" --port 10093 --use_bf16
