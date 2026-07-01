#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-more_celeb_v2}
COUNT=${COUNT:-4}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
MOLMOACT_PORT=${MOLMOACT_PORT:-8000}
LOG=${A2A_LOG_DIR}/molmoact_${ASSETS}_eval.log

conda activate "${CONDA_ENVS_DIR}/act2ans"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${PYTHONPATH:-}"
export MOLMOACT_HOST="${MOLMOACT_HOST:-127.0.0.1}"
export MOLMOACT_PORT
export MOLMOACT_NORM_TAG="${MOLMOACT_NORM_TAG:-widowx_bridge}"

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

echo "START_MOLMOACT_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU port=$MOLMOACT_PORT"
until portopen "$MOLMOACT_PORT"; do
  echo "WAIT_MOLMOACT_SERVER_${MOLMOACT_PORT}"
  sleep 10
done
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_MOLMOACT ${swap_arg} $(date -u)"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla molmoact --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" "${extra[@]}"
done
echo "DONE_MOLMOACT_EVAL $(date -u)"
