#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"

ENV_PATH=${CONDA_ENVS_DIR}/pi0_act2answer
LOG=${A2A_LOG_DIR}/pi0_setup.log

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
echo "START_PI0_SETUP $(date -u)"

if [ ! -d "$PI0_DEPS_ROOT" ]; then
  echo "ERROR: PI0_DEPS_ROOT does not exist: $PI0_DEPS_ROOT"
  echo "Run: bash scripts/setup/clone_external_repos.sh"
  exit 1
fi

if [ ! -d "$ENV_PATH" ]; then
  conda create --solver libmamba --override-channels -c defaults -p "$ENV_PATH" -y python=3.10 pip
fi
conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1

retry pip4 install --upgrade pip wheel setuptools ninja packaging
retry pip4 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
retry pip4 install -r "$REPO_ROOT/requirements/pi0.txt"
retry pip4 install -e "$REPO_ROOT/ManiSkill" -e "$REPO_ROOT/SimplerEnv"

python - <<'PY'
import os, sys, torch
sys.path.insert(0, os.environ["PI0_DEPS_ROOT"])
import draccus
print("pi0_env_ok", torch.__version__, "cuda", torch.cuda.is_available())
PY
echo "DONE_PI0_SETUP $(date -u)"
