#!/usr/bin/env bash
set -euo pipefail

echo "========== 0. 检查当前目录 =========="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}"

echo "[Info] 当前脚本目录: ${SCRIPT_DIR}"

if [ "$(basename "${TARGET_DIR}")" != "vggt" ]; then
    echo "[Error] 请将脚本放在 cann-recipes-embodied-intelligence/3d_vision/vggt 目录下执行"
    exit 1
fi

echo "========== 1. 准备 VGGT 官方仓库 =========="

WORKSPACE_DIR="cann_recipes" # 请用户自行配置
OFFICIAL_VGGT_DIR="${WORKSPACE_DIR}/vggt"

if [ ! -d "${OFFICIAL_VGGT_DIR}" ]; then
    git clone https://github.com/facebookresearch/vggt.git "${OFFICIAL_VGGT_DIR}"
else
    echo "[Skip] ${OFFICIAL_VGGT_DIR} 已存在"
fi

echo "========== 2. 下载 VGGT 模型权重 =========="

pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

if [ ! -f "${OFFICIAL_VGGT_DIR}/model.pt" ]; then
    hf download facebook/VGGT-1B --local-dir "${OFFICIAL_VGGT_DIR}"
else
    echo "[Skip] ${OFFICIAL_VGGT_DIR}/model.pt 已存在"
fi

echo "========== 3. 创建 ckpt 目录并复制模型权重 =========="

CKPT_DIR="${TARGET_DIR}/ckpt"
mkdir -p "${CKPT_DIR}"

if [ -f "${OFFICIAL_VGGT_DIR}/model.pt" ]; then
    cp -n "${OFFICIAL_VGGT_DIR}/model.pt" "${CKPT_DIR}/model.pt"
    echo "[OK] model.pt 已复制到 ${CKPT_DIR}/model.pt"
else
    echo "[Error] 未找到 ${OFFICIAL_VGGT_DIR}/model.pt，请检查权重是否下载成功"
    exit 1
fi

echo "========== 4. 复制 VGGT 网络结构代码到当前项目目录 =========="

mkdir -p "${TARGET_DIR}/vggt"

cp -n "${OFFICIAL_VGGT_DIR}/visual_util.py" "${TARGET_DIR}/" || true
cp -rn "${OFFICIAL_VGGT_DIR}/examples" "${TARGET_DIR}/" || true

cp -rn "${OFFICIAL_VGGT_DIR}/vggt/dependency" "${TARGET_DIR}/vggt/dependency" || true
cp -rn "${OFFICIAL_VGGT_DIR}/vggt/heads" "${TARGET_DIR}/vggt/" || true
cp -rn "${OFFICIAL_VGGT_DIR}/vggt/layers" "${TARGET_DIR}/vggt/" || true
cp -rn "${OFFICIAL_VGGT_DIR}/vggt/utils" "${TARGET_DIR}/vggt/" || true

echo "========== 5. 安装 Python 依赖 =========="

cd "${TARGET_DIR}"
pip3 install -r requirements.txt

echo "========== VGGT 环境与代码准备完成 =========="
echo "项目目录: ${TARGET_DIR}"
echo "官方 VGGT 仓库目录: ${OFFICIAL_VGGT_DIR}"
echo "权重文件: ${CKPT_DIR}/model.pt"