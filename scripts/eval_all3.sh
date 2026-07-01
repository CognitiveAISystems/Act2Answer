#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSET=${1:-}
if [ -z "$ASSET" ]; then
  echo "Usage: bash scripts/eval_all3.sh <asset_name> [count] [eval_gpu]"
  exit 1
fi

ASSET_DIR="$REPO_ROOT/ManiSkill/mani_skill/assets/carrot/$ASSET"
if [ ! -d "$ASSET_DIR" ]; then
  echo "ERROR: asset '$ASSET' not found at $ASSET_DIR"
  exit 1
fi

COUNT=${2:-$(python3 -c "import json;print(len(json.load(open('$ASSET_DIR/pairs.json'))))")}
EVAL_GPU=${3:-3}
XIAOMI_SRV_GPU=${XIAOMI_SRV_GPU:-2}
INTERNVLA_SRV_GPU=${INTERNVLA_SRV_GPU:-3}

portopen() {
  python3 - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(2)
s.connect(("localhost", int(sys.argv[1])))
s.close()
PY
}

waitport() {
  local port=$1
  for _ in $(seq 1 90); do
    portopen "$port" && return 0
    sleep 5
  done
  return 1
}

echo "############ Act2Answer eval | asset=$ASSET | tasks=$COUNT | eval_gpu=$EVAL_GPU ############"

if portopen 10086; then
  echo "[ok] Xiaomi server already up (10086)"
else
  echo "[..] starting Xiaomi WidowX server on GPU $XIAOMI_SRV_GPU"
  tmux kill-session -t xiaomi_srv 2>/dev/null || true
  tmux new-session -d -s xiaomi_srv \
    "GPU=$XIAOMI_SRV_GPU PORT=10086 bash $REPO_ROOT/scripts/servers/run_xiaomi_policy_server.sh"
  waitport 10086 && echo "[ok] Xiaomi server up" || echo "[ERR] Xiaomi server failed; check logs"
fi

if portopen 10093; then
  echo "[ok] InternVLA server already up (10093)"
else
  echo "[..] starting InternVLA-M1 server on GPU $INTERNVLA_SRV_GPU"
  tmux kill-session -t internvla_srv 2>/dev/null || true
  tmux new-session -d -s internvla_srv \
    "GPU=$INTERNVLA_SRV_GPU bash $REPO_ROOT/scripts/servers/run_internvla_server.sh"
  waitport 10093 && echo "[ok] InternVLA server up" || echo "[ERR] InternVLA server failed; check logs"
fi

echo
echo ">>> [1/3] Xiaomi WidowX"
ASSETS=$ASSET COUNT=$COUNT EVAL_GPU=$EVAL_GPU bash "$REPO_ROOT/scripts/eval_xiaomi.sh" >/dev/null 2>&1
echo ">>> [2/3] InternVLA-M1"
ASSETS=$ASSET COUNT=$COUNT EVAL_GPU=$EVAL_GPU bash "$REPO_ROOT/scripts/eval_internvla.sh" >/dev/null 2>&1
echo ">>> [3/3] SpatialVLA-4B"
ASSETS=$ASSET COUNT=$COUNT EVAL_GPU=$EVAL_GPU bash "$REPO_ROOT/scripts/eval_spatialvla.sh" >/dev/null 2>&1

python3 - "$ASSET" "$COUNT" "$A2A_LOG_DIR" <<'PY'
import ast
import re
import sys
from pathlib import Path

asset, count, log_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
models = [
    ("Xiaomi WidowX", "xiaomi"),
    ("InternVLA-M1", "internvla"),
    ("SpatialVLA-4B", "spatialvla"),
]


def vals(key):
    path = log_dir / f"{key}_{asset}_eval.log"
    if not path.exists():
        return (None, None)
    stats = []
    for match in re.findall(r"FINAL_STATS (\{.*?\})", path.read_text()):
        try:
            stats.append(ast.literal_eval(match).get("success"))
        except Exception:
            pass
    return (stats[0] if len(stats) > 0 else None, stats[1] if len(stats) > 1 else None)


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "  -  "


line = "=" * 64
print(line)
print(f"  Act2Answer scores  |  asset = {asset}  ({count} tasks)")
print(line)
print(f"  {'Model':<16}{'noswap':>9}{'swap':>9}{'combined':>11}")
print("-" * 64)
for name, key in models:
    noswap, swap = vals(key)
    combined = (noswap + swap) / 2 if noswap is not None and swap is not None else None
    print(f"  {name:<16}{fmt(noswap):>9}{fmt(swap):>9}{fmt(combined):>11}")
print(line)
print("  combined = mean(noswap, swap); chance ~= 0.50")
PY
