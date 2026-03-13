#!/bin/bash
# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 Shanghai Jiao Tong University
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -euo pipefail  # 开启严格模式：报错立即退出、未定义变量报错、管道失败报错

# ===================== 自动获取当前路径 =====================
ROOT_DIR="$(pwd)"
LEROBOT_DIR="${ROOT_DIR}/lerobot"
CANN_RECIPES_DIR="${ROOT_DIR}/cann-recipes-embodied-intelligence"
PI05_MODEL_DIR="${LEROBOT_DIR}/pi05_model"

# 固定版本号
LEROBOT_COMMIT="fc296548cb6438b3036d046b43fe91951c87ea9a"
PI05_MODEL_COMMIT="d856522a96167505ff48af2ec9f151154486fc9a"

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
    echo -e "\033[31;1m[ERROR] $1\033[0m"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "命令 $1 未找到，请先安装"
    fi
}

# ===================== 前置检查 =====================
info "开始执行 Pi0.5 代码仓、模型、数据集一键下载脚本"
info "自动获取的父目录：${ROOT_DIR}"

# 检查必要命令（git、cp仍需检查）
check_command "git"
check_command "cp"

# 验证当前目录是否已包含目标文件夹
if [ -d "${CANN_RECIPES_DIR}" ] && [ -d "${LEROBOT_DIR}" ]; then
    info "当前目录已存在 cann-recipes 和 lerobot 文件夹，将在原有基础上更新/补充内容"
elif [ -d "${CANN_RECIPES_DIR}" ] || [ -d "${LEROBOT_DIR}" ]; then
    info "当前目录仅存在一个目标文件夹，后续克隆将自动补全缺失的内容"
else
    info "当前目录无目标文件夹，将全新克隆所有代码仓/数据集"
fi

# ===================== 步骤1：拉取 lerobot 代码仓并回退版本 =====================
progress "开始拉取 lerobot 代码仓..."
if [ -d "${LEROBOT_DIR}" ]; then
    info "lerobot 目录已存在，跳过克隆，直接回退版本"
    cd "${LEROBOT_DIR}"
else
    git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}" || error "lerobot 代码仓克隆失败"
    cd "${LEROBOT_DIR}"
fi

progress "lerobot 代码仓拉取完成，开始回退到指定版本：${LEROBOT_COMMIT}"
git reset --hard "${LEROBOT_COMMIT}" || error "lerobot 版本回退失败"
success "lerobot 代码仓拉取并回退版本成功！"


# ===================== 步骤2：复制 cann-recipes-embodied-intelligence 中pi0.5相关文件到 lerobot =====================
progress "开始复制 cann-recipes-embodied-intelligence 文件夹中的 pi0.5 相关文件到 lerobot..."

cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/infer_with_torch/modeling_pi05.patch" "${LEROBOT_DIR}/src/lerobot/policies/pi05/" || error "复制 modeling_pi05.patch 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/infer_with_torch/pyproject.toml" "${LEROBOT_DIR}/" || error "复制 pyproject.toml 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/infer_with_torch/run_pi05_example.py" "${LEROBOT_DIR}/" || error "复制 run_pi05_example.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/infer_with_torch/run_pi05_inference.sh" "${LEROBOT_DIR}/" || error "复制 run_pi05_inference.sh 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/infer_with_torch/verify_pi05_accuracy_ascend.py" "${LEROBOT_DIR}/" || error "复制 verify_pi05_accuracy_ascend.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/infer_with_torch/infer_utils.py" "${LEROBOT_DIR}/" || error "复制 infer_utils.py 失败"
success "cann-recipes 文件复制成功！"

# ===================== 步骤3：应用 modeling_pi05.patch 补丁到 lerobot =====================
progress "开始应用 modeling_pi05.patch 补丁到 lerobot..."
cd "${LEROBOT_DIR}"
git apply "${CANN_RECIPES_DIR}/manipulation/pi05/infer_with_torch/modeling_pi05.patch" || error "应用 modeling_pi05.patch 补丁失败"
success "modeling_pi05.patch 补丁应用成功！"

# ===================== 步骤4：下载 pi05 模型权重 =====================
cd "${LEROBOT_DIR}"
progress "开始拉取/更新 pi05 模型仓库（跳过默认 LFS 下载）..."

if [ -d "${PI05_MODEL_DIR}" ]; then
    info "pi05_model 目录已存在，直接更新并切换到目标版本..."
    cd "${PI05_MODEL_DIR}"
    
    # 1. 拉取仓库最新的指针文件和提交历史（确保能找到目标版本）
    git fetch origin || error "拉取 pi05 模型仓库最新信息失败"
    
    # 2. 清理本地修改（若有，避免影响版本回退）
    git reset --hard HEAD || error "清理本地修改失败"
else
    info "pi05_model 目录不存在，开始克隆仓库..."
    # 克隆时跳过 LFS 自动下载，仅拉取指针和文本文件
    git lfs install --skip-smudge || error "Git LFS 跳过自动下载配置失败"
    git clone https://www.modelscope.cn/models/lerobot/pi05_base.git "${PI05_MODEL_DIR}" || error "pi05 模型仓库克隆失败"
    cd "${PI05_MODEL_DIR}" || error "进入 pi05_model 目录失败"
fi

progress "开始回退到指定版本：${PI05_MODEL_COMMIT}"
GIT_LFS_SKIP_SMUDGE=1 git reset --hard "${PI05_MODEL_COMMIT}" || error "pi05 模型版本回退失败"

progress "开始拉取/更新 pi05 模型 LFS 大文件（权重文件）..."
# 强制初始化 LFS（确保配置生效），拉取目标版本的 LFS 大文件（自动覆盖旧版本）
git lfs install --force || error "Git LFS 强制初始化失败"
git lfs pull || error "pi05 模型 LFS 大文件拉取失败"

success "pi05 模型权重下载/更新成功！"

# ===================== 全部完成 =====================
cd "${ROOT_DIR}"
success "========================================"
success "所有步骤执行完成！"
success "lerobot 代码仓：${LEROBOT_DIR}"
success "pi05 模型权重：${PI05_MODEL_DIR}"
success "========================================"