#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-test_colors}
COUNT=${COUNT:-6}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
LOG=${A2A_LOG_DIR}/magma_${ASSETS}_eval.log

conda activate "${CONDA_ENVS_DIR}/magma_act2answer"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${PYTHONPATH:-}"

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "START_MAGMA_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU"
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_MAGMA ${swap_arg} $(date -u)"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla magma --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" "${extra[@]}"
done
echo "DONE_MAGMA_EVAL $(date -u)"
