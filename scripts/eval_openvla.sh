#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-test_colors}
COUNT=${COUNT:-6}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
VLA_PATH=${VLA_PATH:-gen-robot/openvla-7b-rlvla-sft_16k}
UNNORM=${UNNORM:-sft}
LOG=${A2A_LOG_DIR}/openvla_${ASSETS}_eval.log

conda activate "${CONDA_ENVS_DIR}/openvla_rl4vla"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${REPO_ROOT}/openvla:${PYTHONPATH:-}"

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "START_OPENVLA_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU vla=$VLA_PATH unnorm=$UNNORM"
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_OPENVLA ${swap_arg} $(date -u)"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla openvla --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" \
      --vla-path "$VLA_PATH" --vla-unnorm-key "$UNNORM" "${extra[@]}"
done
echo "DONE_OPENVLA_EVAL $(date -u)"
