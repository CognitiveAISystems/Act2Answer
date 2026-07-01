#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-safe_school}
COUNT=${COUNT:-9}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
LOG=${A2A_LOG_DIR}/spatialvla_${ASSETS}_eval.log

conda activate "${CONDA_ENVS_DIR}/spatialvla_act2answer"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${PYTHONPATH:-}"

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "START_SPATIALVLA_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU ckpt=$SPATIALVLA_CKPT"
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_SPATIALVLA ${swap_arg} $(date -u)"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla spatialvla --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" "${extra[@]}"
done
echo "DONE_SPATIALVLA_EVAL $(date -u)"
