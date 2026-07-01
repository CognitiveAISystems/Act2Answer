#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"

mkdir -p "$A2A_EXTERNAL_DIR"

clone_if_missing() {
  local url=$1
  local dst=$2
  if [ -d "$dst/.git" ]; then
    echo "[ok] $(basename "$dst") already exists at $dst"
    return
  fi
  echo "[clone] $url -> $dst"
  git clone "$url" "$dst"
}

clone_if_missing https://github.com/gen-robot/RL4VLA.git "$PI0_DEPS_ROOT"
clone_if_missing https://github.com/XiaomiRobotics/Xiaomi-Robotics-0.git "$XIAOMI_REPO"
clone_if_missing https://github.com/InternRobotics/InternVLA-M1.git "$INTERNVLA_REPO"
clone_if_missing https://github.com/allenai/molmoact2.git "$MOLMOACT_REPO"

cat <<EOF

External repos are ready:
  PI0_DEPS_ROOT=$PI0_DEPS_ROOT
  XIAOMI_REPO=$XIAOMI_REPO
  INTERNVLA_REPO=$INTERNVLA_REPO
  MOLMOACT_REPO=$MOLMOACT_REPO

Set A2A_EXTERNAL_DIR before running this script if you want a different parent folder.
EOF
