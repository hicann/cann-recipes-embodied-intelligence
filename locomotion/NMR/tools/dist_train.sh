#!/usr/bin/env bash
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../NMR/tools
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                  # .../NMR

cd "$PROJECT_ROOT"

command -v conda || echo "conda not found"
conda info --base || true


if command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
  if [ -n "${_conda_base}" ] && [ -f "${_conda_base}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "${_conda_base}/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "[ERROR] conda.sh not found"; exit 1
  fi
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"; exit 1
fi

# In case conda was initialized by sourcing conda.sh above.
if [ -z "${_conda_base:-}" ] && command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
fi

if [ -n "${_conda_base:-}" ]; then
  ENV_PY="${_conda_base}/envs/PyTorch-2.7.1/bin/python"
elif [ -x "$HOME/anaconda3/envs/PyTorch-2.7.1/bin/python" ]; then
  ENV_PY="$HOME/anaconda3/envs/PyTorch-2.7.1/bin/python"
else
  ENV_PY="$HOME/miniconda3/envs/PyTorch-2.7.1/bin/python"
fi
[ -x "$ENV_PY" ] || { echo "[ERROR] env python not found: $ENV_PY"; exit 1; }

echo "[INFO] Available conda envs:"
conda env list || true
#conda env list | awk '{print $1}' | grep -qx PyTorch-2.7.1 || conda create -y -n PyTorch-2.7.1 python=3.10
#conda install -y -n PyTorch-2.7.1 pytorch
#conda run -n PyTorch-2.7.1 python -m pip install -e "$PROJECT_ROOT"

INSTALL_LOCK="/tmp/retarget_pip_install.lock"

{
  flock -x 9
  echo "[INFO] Installing deps with lock: $INSTALL_LOCK"

  CONSTRAINT_FILE="$PROJECT_ROOT/tools/pip-constraints-train.txt"
  cat > "$CONSTRAINT_FILE" << 'EOF'
numpy>=1.23,<2.0
scipy==1.15.3
EOF

  "$ENV_PY" -m pip install --prefer-binary -e "$PROJECT_ROOT" -c "$CONSTRAINT_FILE"
} 9>"$INSTALL_LOCK"


CONFIG="${CONFIG:-configs/retarget_fwd.py}"

TRAIN_OUT_PATH="${train_out_path:-${TRAIN_OUT_PATH:-}}"
if [ -n "$TRAIN_OUT_PATH" ]; then
WORD_DIR="$TRAIN_OUT_PATH"
else
WORD_DIR="$PROJECT_ROOT/work_dirs/retarget_pred_token_l_4_seq"
fi
mkdir -p "$WORD_DIR"
echo "[INFO] work dir: $WORD_DIR"

GPUS=${GPUS:-1}
if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a _devs <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  GPUS="${#_devs[@]}"
elif [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a _devs <<< "${ASCEND_VISIBLE_DEVICES}"
  GPUS="${#_devs[@]}"
fi
[ "$GPUS" -lt 1 ] && GPUS=1
echo "[INFO] GPUS=${GPUS}, ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}, ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-}"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((20000 + RANDOM % 30000))}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}



echo "PORT=${PORT}"
TRAIN_LAUNCH_LOCK="/tmp/retarget_train_launch_${MA_JOB_ID:-default}.lock"


# 平台若已注入 RANK/LOCAL_RANK，说明外层已做分布式拉起
if [ -n "${RANK:-}" ] || [ -n "${LOCAL_RANK:-}" ] || [ -n "${WORLD_SIZE:-}" ]; then
  echo "[INFO] external distributed launcher detected: RANK=${RANK:-}, LOCAL_RANK=${LOCAL_RANK:-}, WORLD_SIZE=${WORLD_SIZE:-}"
  PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
  "$ENV_PY" "$PROJECT_ROOT/tools/train.py" \
    --work-dir "$WORD_DIR" \
    "$CONFIG" \
    --launcher pytorch "${@}"
else
  echo "[INFO] no external launcher, use torchrun with nproc_per_node=$GPUS"
  {
  flock -n 8 || { echo "[WARN] another launcher already running, exit duplicate starter"; exit 0; }
  echo "[INFO] acquired train launch lock: $TRAIN_LAUNCH_LOCK"
  PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
  "$ENV_PY" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$PROJECT_ROOT/tools/train.py" \
    --work-dir "$WORD_DIR" \
    "$CONFIG" \
    --launcher pytorch "${@}"
} 8>"$TRAIN_LAUNCH_LOCK"
fi

