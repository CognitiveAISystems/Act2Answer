#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-safe_school}
COUNT=${COUNT:-9}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
XIAOMI_PORT=${XIAOMI_PORT:-10086}
LOG=${A2A_LOG_DIR}/xiaomi_${ASSETS}_eval.log

conda activate "${CONDA_ENVS_DIR}/act2ans"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${PYTHONPATH:-}"
export XIAOMI_TASK_ID="${XIAOMI_TASK_ID:-bridge_delta}"

portopen() {
  python3 - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(3)
s.connect(("localhost", int(sys.argv[1])))
s.close()
PY
}

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "START_XIAOMI_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU port=$XIAOMI_PORT"
until portopen "$XIAOMI_PORT"; do
  echo "WAIT_XIAOMI_SERVER_${XIAOMI_PORT}"
  sleep 10
done
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_XIAOMI ${swap_arg} $(date -u)"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla xiaomi --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" "${extra[@]}"
done
echo "DONE_XIAOMI_EVAL $(date -u)"
