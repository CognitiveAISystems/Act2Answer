#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"

ENV_PATH=${CONDA_ENVS_DIR}/openvla_rl4vla
LOG=${A2A_LOG_DIR}/openvla_setup.log

pip4() { python "$REPO_ROOT/scripts/setup/ipv4_pip.py" "$@"; }
retry() {
  local n=1
  until "$@"; do
    [ "$n" -ge 6 ] && return 1
    echo "RETRY $n: $*"
    n=$((n + 1))
    sleep 8
  done
}

exec > >(tee -a "$LOG") 2>&1
echo "START_OPENVLA_SETUP $(date -u)"

if [ ! -d "$ENV_PATH" ]; then
  conda create --solver libmamba --override-channels -c defaults -p "$ENV_PATH" -y python=3.10 pip
fi
conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1

retry pip4 install --upgrade pip wheel setuptools ninja packaging
retry pip4 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
retry pip4 install flash-attn==2.7.4.post1 --no-build-isolation
retry pip4 install -r "$REPO_ROOT/requirements/openvla.txt"
retry pip4 install -e "$REPO_ROOT/ManiSkill" -e "$REPO_ROOT/SimplerEnv" -e "$REPO_ROOT/openvla"

python - <<'PY'
import torch, transformers
print("openvla_env_ok", torch.__version__, transformers.__version__, "cuda", torch.cuda.is_available())
PY
echo "DONE_OPENVLA_SETUP $(date -u)"
