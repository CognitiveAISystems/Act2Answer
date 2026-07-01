#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-test_colors}
COUNT=${COUNT:-6}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
LOG=${A2A_LOG_DIR}/pi0_${ASSETS}_eval.log

conda activate "${CONDA_ENVS_DIR}/pi0_act2answer"
export VLA_DATA_DIR="${PI0_DEPS_ROOT}"
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29555}"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${PI0_DEPS_ROOT}:${PYTHONPATH:-}"

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "START_PI0_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU"
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_PI0 ${swap_arg} $(date -u)"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla pi0 --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch 1 "${extra[@]}"
done
echo "DONE_PI0_EVAL $(date -u)"
