#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"
LOG=${A2A_LOG_DIR}/molmoact2_policy_server.log
ENV_PATH="${CONDA_ENVS_DIR}/molmoact2"
REPO=${MOLMOACT_REPO}
LOCAL_DIR=${MOLMOACT_LOCAL_DIR:-${MOLMOACT_CKPT}}
PORT=${PORT:-8000}
GPU=${GPU:-0}
NORM_TAG=${MOLMOACT_NORM_TAG:-widowx_bridge}
exec > >(tee -a "$LOG") 2>&1
echo "START_MOLMOACT2_POLICY_SERVER $(date -u) gpu=$GPU port=$PORT norm_tag=$NORM_TAG local_dir=$LOCAL_DIR"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" python examples/simpler/host_server_simpler.py \
  --host localhost --port "$PORT" --norm-tag "$NORM_TAG" --local-dir "$LOCAL_DIR"
