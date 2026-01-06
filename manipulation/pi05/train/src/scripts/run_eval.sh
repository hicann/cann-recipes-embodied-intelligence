#!/bin/bash
# Copyright (c) 2026 Institute of Software, Chinese Academy of Sciences (ISCAS). All rights reserved.
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================
# run_eval.sh
# 功能: 在 Ascend NPU 上运行 LeRobot 评估任务（支持 libero / aloha / pusht 等环境）
# 依赖: Xvfb + OSMesa（无头渲染），LeRobot 环境已激活
# =============================================

set -e  # 遇错退出

# -------------------------
# 图形与渲染环境配置（无头模式）
# -------------------------
echo "[INFO] Starting virtual display (Xvfb)..."
Xvfb :1 -screen 0 1024x768x24 > /tmp/xvfb.log 2>&1 &
export DISPLAY=:1
export LIBGL_ALWAYS_SOFTWARE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libOSMesa.so
export LD_LIBRARY_PATH=/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa

# -------------------------
# 项目路径
# -------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../../.." && pwd )/lerobot"
echo "Project root: $PROJECT_ROOT"

# -------------------------
# 评估函数：统一入口
# -------------------------
run_evaluation() {
    local policy_path="$1"
    local env_type="$2"
    local env_task="$3"
    local output_dir="$4"
    local batch_size="${5:-1}"
    local n_episodes="${6:-1}"

    echo
    echo "=============================================="
    echo "[EVAL] Policy: $policy_path"
    echo "[EVAL] Env: $env_type | Task: $env_task"
    echo "[EVAL] Output: $output_dir"
    echo "=============================================="

    # 创建输出目录
    mkdir -p "$output_dir"

    # 执行评估
    python -m lerobot.scripts.lerobot_eval \
        --policy.path="$policy_path" \
        --env.type="$env_type" \
        --env.task="$env_task" \
        --eval.batch_size="$batch_size" \
        --eval.n_episodes="$n_episodes" \
        --policy.device=npu \
        --output_dir="$output_dir"
}

# -------------------------
# 主逻辑：支持传参或运行默认任务
# -------------------------
if [[ $# -gt 0 ]]; then
    # 如果传入命令行参数，则直接透传给 lerobot_eval
    echo "[INFO] Running custom evaluation with args: $*"
    python -m lerobot.scripts.lerobot_eval "$@"
else
    # 默认运行官方 smolvla libero 评估任务
    echo "[INFO] No arguments provided. Running default evaluation task..."

    run_evaluation \
        "HuggingFaceVLA/smolvla_libero" \
        "libero" \
        "libero_object" \
        "./eval_output/smolvla_libero_official" \
        1 1     # batch_size=1, n_episodes=1
fi

echo "[INFO] Evaluation finished."