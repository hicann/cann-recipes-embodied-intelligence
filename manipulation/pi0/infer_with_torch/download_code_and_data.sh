#!/bin/bash
set -euo pipefail  # 开启严格模式：报错立即退出、未定义变量报错、管道失败报错

# ===================== 自动获取当前路径 =====================
ROOT_DIR="$(pwd)"
LEROBOT_DIR="${ROOT_DIR}/lerobot"
CANN_RECIPES_DIR="${ROOT_DIR}/cann-recipes-embodied-intelligence"
KOCH_TEST_DIR="${LEROBOT_DIR}/koch_test"
PI0_MODEL_DIR="${LEROBOT_DIR}/pi0_model"

# 固定版本号
LEROBOT_COMMIT="a27411022dd5f3ce6ebb75b460376cb844699df8"
PI0_MODEL_COMMIT="01189b1ffb1c9f2f9622c3b1ae773cd884bfd84f"

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
info "开始执行 Pi0 代码仓、模型、数据集一键下载脚本"
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


# ===================== 步骤2：复制 cann-recipes-embodied-intelligence 中pi0相关文件到 lerobot =====================
progress "开始复制 cann-recipes-embodied-intelligence 文件夹中的 pi0 相关文件到 lerobot..."

cp -f "${CANN_RECIPES_DIR}/manipulation/pi0/infer_with_torch/modeling_pi0.py" "${LEROBOT_DIR}/lerobot/common/policies/pi0/" || error "复制 modeling_pi0.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi0/infer_with_torch/paligemma_with_expert.py" "${LEROBOT_DIR}/lerobot/common/policies/pi0/" || error "复制 paligemma_with_expert.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi0/infer_with_torch/pyproject.toml" "${LEROBOT_DIR}/" || error "复制 pyproject.toml 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi0/infer_with_torch/run_pi0_inference.sh" "${LEROBOT_DIR}/" || error "复制 run_pi0_inference.sh 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi0/infer_with_torch/test_pi0_on_ascend.py" "${LEROBOT_DIR}/" || error "复制 test_pi0_on_ascend.py 失败"
success "cann-recipes 文件复制成功！"


# ===================== 步骤3：下载 koch_test 数据集 =====================
cd "${LEROBOT_DIR}"
progress "开始下载 koch_test 数据集（一站式平台兼容方式）..."
info "检查并安装依赖 huggingface_hub ..."
pip install -U huggingface_hub &> /dev/null || error "huggingface_hub 安装失败"
if [ -d "${KOCH_TEST_DIR}" ]; then
  DIR_SIZE_MB=$(du -sm "${KOCH_TEST_DIR}" | cut -f1)
  info "当前数据集大小：${DIR_SIZE_MB} MB"
  if [ ${DIR_SIZE_MB} -ge 900 ]; then
    info "数据集已完整，跳过下载"
  else
    info "数据集不完整，删除重新下载"
    rm -rf "${KOCH_TEST_DIR}"
  fi
fi

if [ ! -d "${KOCH_TEST_DIR}" ]; then
  python3 -c "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com';
from huggingface_hub import snapshot_download;
snapshot_download(
repo_id='danaaubakirova/koch_test',
repo_type='dataset',
local_dir='${KOCH_TEST_DIR}'
)" || error "koch_test 数据集下载失败"
fi
success "koch_test 数据集（视频、图像、机械臂数据）下载成功！"


# ===================== 步骤4：下载 pi0 模型权重 =====================
cd "${LEROBOT_DIR}"
progress "检查 Git LFS 是否已安装..."

# 提示信息（安装失败时显示）
LFS_INSTALL_HELP="请参考 Git LFS 官网，根据你的主机CPU架构（arm64/x86_64）下载对应二进制版本手动安装：https://git-lfs.com/"

if command -v git-lfs &> /dev/null; then
    info "Git LFS 已安装，跳过 apt 安装步骤"
    git lfs install || { error "Git LFS 初始化失败！${LFS_INSTALL_HELP}"; }
else
    info "Git LFS 未安装，尝试自动安装..."

    # 尝试直接安装
    if apt install git-lfs -y &> /dev/null; then
        info "Git LFS 自动安装成功"
        git lfs install || { error "Git LFS 初始化失败！${LFS_INSTALL_HELP}"; }
    else
        info "自动安装失败，尝试更新 apt 后安装..."

        # 尝试更新apt
        if ! apt update -y &> /dev/null; then
            error "apt 更新失败！${LFS_INSTALL_HELP}"
        fi

        # 再次尝试安装
        if ! apt install git-lfs -y &> /dev/null; then
            error "Git LFS 自动安装失败！${LFS_INSTALL_HELP}"
        fi

        # 初始化
        git lfs install || { error "Git LFS 初始化失败！${LFS_INSTALL_HELP}"; }
    fi
fi


progress "开始拉取/更新 pi0 模型仓库（使用 git lfs 进行下载）..."

if [ -d "${PI0_MODEL_DIR}" ]; then
    info "pi0_model 目录已存在，直接更新并切换到目标版本..."
    cd "${PI0_MODEL_DIR}"
    
    # 1. 拉取仓库最新的指针文件和提交历史（确保能找到目标版本）
    git fetch origin || error "拉取 pi0 模型仓库最新信息失败"
    
    # 2. 清理本地修改（若有，避免影响版本回退）
    git reset --hard HEAD || error "清理本地修改失败"
else
    info "pi0_model 目录不存在，开始克隆仓库..."
    # 克隆时跳过 LFS 自动下载，仅拉取指针和文本文件
    git lfs install --skip-smudge || error "Git LFS 跳过自动下载配置失败"
    git clone https://www.modelscope.cn/models/lerobot/pi0.git "${PI0_MODEL_DIR}" || error "pi0 模型仓库克隆失败"
    cd "${PI0_MODEL_DIR}" || error "进入 pi0_model 目录失败"
fi

progress "开始回退到指定版本：${PI0_MODEL_COMMIT}"
GIT_LFS_SKIP_SMUDGE=1 git reset --hard "${PI0_MODEL_COMMIT}" || error "pi0 模型版本回退失败"

progress "开始拉取/更新 pi0 模型 LFS 大文件（权重文件）..."
# 强制初始化 LFS（确保配置生效），拉取目标版本的 LFS 大文件（自动覆盖旧版本）
git lfs install --force || error "Git LFS 强制初始化失败"
git lfs pull || error "pi0 模型 LFS 大文件拉取失败"

success "pi0 模型权重下载/更新成功！"

# ===================== 全部完成 =====================
cd "${ROOT_DIR}"
success "========================================"
success "所有步骤执行完成！"
success "lerobot 代码仓：${LEROBOT_DIR}"
success "koch_test 数据集：${KOCH_TEST_DIR}"
success "pi0 模型权重：${PI0_MODEL_DIR}"
success "========================================"