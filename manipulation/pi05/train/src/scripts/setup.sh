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

set -euo pipefail

# ===================== 自动获取当前路径 =====================
ROOT_DIR="$(pwd)"
LEROBOT_DIR="${ROOT_DIR}/lerobot"
CANN_RECIPES_DIR="${ROOT_DIR}/cann-recipes-embodied-intelligence"

# 固定 LeRobot 仓库版本
LEROBOT_COMMIT="58f70b6bd370864139a3795ac3497a9eae8c42d5"

# ===================== 工具函数 =====================
info() {
    echo -e "\033[32m[INFO] $1\033[0m"
}

progress() {
    echo -e "\033[34m[PROGRESS] $1\033[0m"
}

success() {
    echo -e "\033[32;1m[SUCCESS] $1\033[0m"
}

error() {
    echo -e "\033[31;1m[ERROR] $1\033[0m" >&2
    exit 1
}

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "命令 '$1' 未找到，请先安装"
    fi
}

# ===================== 前置检查 =====================
info "开始执行 Pi05 代码仓、模型、数据集一键下载脚本"
info "工作目录：${ROOT_DIR}"

check_command "git"
check_command "cp"

# 判断目标目录是否存在
if [[ -d "${CANN_RECIPES_DIR}" && -d "${LEROBOT_DIR}" ]]; then
    info "cann-recipes 与 lerobot 目录均已存在，将进行增量更新"
elif [[ -d "${CANN_RECIPES_DIR}" || -d "${LEROBOT_DIR}" ]]; then
    info "部分目录已存在，将补全缺失内容"
else
    info "未检测到目标目录，将执行全新克隆"
fi

# ===================== 步骤 1：拉取并锁定 LeRobot 代码版本 =====================
progress "拉取 LeRobot 代码仓库..."
if [[ -d "${LEROBOT_DIR}" ]]; then
    cd "${LEROBOT_DIR}"
    info "lerobot 目录已存在，跳过克隆"
else
    git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}" || error "LeRobot 仓库克隆失败"
    cd "${LEROBOT_DIR}"
fi

git fetch --all --quiet
git reset --hard "${LEROBOT_COMMIT}" || error "无法回退到指定提交：${LEROBOT_COMMIT}"
success "LeRobot 代码仓库已锁定至指定版本"

# ===================== 步骤 2：复制 Pi05 相关文件 =====================
progress "复制 Pi05 实现文件至 LeRobot 项目中..."
# 确保目标子目录存在
mkdir -p "${LEROBOT_DIR}/src/lerobot/policies/pi05/"
mkdir -p "${LEROBOT_DIR}/src/lerobot/scripts/"
mkdir -p "${LEROBOT_DIR}/src/lerobot/utils/"

cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/modeling_pi05.py" "${LEROBOT_DIR}/src/lerobot/policies/pi05/" || error "复制 modeling_pi05.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_train.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 lerobot_train.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_eval.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 lerobot_eval.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_train_profiling.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 run_train_profiling.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/utils.py" "${LEROBOT_DIR}/src/lerobot/utils/" || error "复制 utils.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/configs/"*.yaml "${LEROBOT_DIR}/src/lerobot/configs/" || error "复制配置文件失败"
success "Pi05 相关文件复制完成"

# ===================== 步骤 3：创建并激活 Conda 虚拟环境 =====================
progress "创建 Conda 虚拟环境 'lerobot' (Python 3.10)..."
check_command "conda"

# 使用 eval 确保在非交互式 shell 中正确激活环境
conda create -y -n lerobot python=3.10
eval "$(conda shell.bash hook)"
conda activate lerobot || error "无法激活 Conda 环境 'lerobot'"

cd "${LEROBOT_DIR}"

# 安装基础依赖
progress "安装 LeRobot 基础依赖..."
pip install -e . || error "LeRobot 基础安装失败"

# ===================== 步骤 4：安装 Git LFS（用于下载模型/数据集） =====================
progress "配置 Git LFS（用于模型与数据集大文件管理）..."
if command -v git-lfs >/dev/null 2>&1; then
    info "Git LFS 已安装"
else
    info "Git LFS 未安装，正在安装..."
    # 尝试直接安装，避免无谓的 apt update
    if ! apt install -y git-lfs >/dev/null 2>&1; then
        info "首次安装失败，尝试更新包列表后重试..."
        apt update -y >/dev/null || error "apt 更新失败"
        apt install -y git-lfs || error "Git LFS 安装失败"
    fi
fi

git lfs install --local || error "Git LFS 初始化失败"

# ===================== 步骤 5：安装 Pi 与 Libero 可选依赖 =====================
progress "安装 Pi05 相关可选依赖（pi）..."
# 检查 cmake 是否存在
 	  if ! command -v cmake &> /dev/null; then
 	      error "cmake 未安装，请先安装 cmake"
 	  fi
 	  CMAKE_VERSION= $ (cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
 	  MAJOR_VERSION= $ (echo " $ CMAKE_VERSION" | cut -d. -f1)
 	 
 	  # 检查主版本是否 >= 4
 	  if [ " $ MAJOR_VERSION" -ge 4 ]; then
 	      error "检测到 CMake 版本  $ CMAKE_VERSION，但 egl_probe 不兼容 CMake 4+。请使用 CMake 3.x。"
 	  fi
 	  progress "CMake 版本  $ CMAKE_VERSION 兼容，继续安装..."
 	 
 	  pip install -e ".[pi]" || error "Pi05 依赖安装失败"

progress "安装 Libero 仿真环境依赖（libero）..."
pip install -e ".[libero]" || error "Libero 依赖安装失败"

# ===================== 步骤 6：aarch64 架构下手动编译 torchcodec（可选） =====================
if [[ "$(uname -m)" == "aarch64" ]]; then
    progress "检测到 aarch64 架构，尝试安装 torchcodec..."
    
    # 检查是否已安装 torchcodec
    if python -c "import torchcodec" &>/dev/null; then
        info "torchcodec 已安装，跳过编译"
    else
        info "正在安装 pybind11"
        pip install pybind11
        info "正在克隆 torchcodec 仓库..."
        git clone https://github.com/meta-pytorch/torchcodec.git || error "torchcodec 仓库克隆失败"
        cd torchcodec

        info "正在编译并安装 torchcodec..."
        # 使用 --no-build-isolation 避免重复下载 PyTorch；-v 用于调试
        if pip install -e . --no-build-isolation -v; then
            success "torchcodec 安装成功"
        else
            warning "torchcodec 安装失败。若需视频解码功能，请手动排查编译依赖（如 ffmpeg、cmake 等）"
            # 不 exit，因为 torchcodec 仅为可选优化组件
        fi
        cd "${LEROBOT_DIR}"  # 返回主目录
    fi
else
    info "当前架构非 aarch64（$(uname -m)），跳过 torchcodec 编译"
fi

success "所有依赖安装完成"

# ====================== 安装torch_npu ======================
progress "安装 torch_npu..."
if python -c "import torch_npu" &>/dev/null; then
    info "torch_npu 已安装，跳过安装"
else
    wget https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_28_aarch64.whl
    pip3 install torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_28_aarch64.whl
    success "torch_npu 安装成功"
fi

# ===================== 完成提示 =====================
cd "${ROOT_DIR}"
success "========================================"
success "Pi05 项目环境部署成功！"
success "请运行 'conda activate lerobot' 启用环境"
success "========================================"