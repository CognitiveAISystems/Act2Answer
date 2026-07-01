#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"

LOG=${A2A_LOG_DIR}/xiaomi_server_setup.log
ENV_PATH=${CONDA_ENVS_DIR}/mibot
REPO=${XIAOMI_REPO}

exec > >(tee -a "$LOG") 2>&1

echo "START_XIAOMI_SERVER_SETUP $(date)"
source /opt/conda/etc/profile.d/conda.sh

if [ ! -d "$REPO" ]; then
  echo "ERROR: XIAOMI_REPO does not exist: $REPO"
  echo "Run: bash scripts/setup/clone_external_repos.sh"
  exit 1
fi

if [ ! -d "$ENV_PATH" ]; then
  /opt/conda/bin/conda create --solver libmamba --override-channels -c defaults -p "$ENV_PATH" -y python=3.12.2 pip
fi

conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1
python -V

python "$REPO_ROOT/scripts/setup/ipv4_pip.py" install --upgrade pip wheel setuptools ninja packaging
python "$REPO_ROOT/scripts/setup/ipv4_pip.py" install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python "$REPO_ROOT/scripts/setup/ipv4_pip.py" install -r "$REPO_ROOT/requirements/xiaomi_server.txt"

if ! python -c "import flash_attn" >/dev/null 2>&1; then
  python "$REPO_ROOT/scripts/setup/ipv4_pip.py" install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
fi

cd "$REPO"
python "$REPO_ROOT/scripts/setup/ipv4_pip.py" install -e xr0 --no-deps || true
python - <<'PY'
import torch
import transformers
import flash_attn
print("server_env_ok", torch.__version__, transformers.__version__, flash_attn.__version__)
PY

echo "DONE_XIAOMI_SERVER_SETUP $(date)"
