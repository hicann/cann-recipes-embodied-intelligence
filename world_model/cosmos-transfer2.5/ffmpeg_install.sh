#!/bin/bash
# Adapted from
# https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ====================== 脚本配置项（可根据需求修改）======================
# Python版本
PYTHON_VERSION="3.10.19"
# FFmpeg版本
FFMPEG_VERSION="4.4.2"
# 编译线程数（根据服务器CPU核心数调整，A3服务器推荐64）
MAKE_THREADS=64
# ========================================================================

# 颜色输出定义，方便查看执行状态
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印日志函数
info() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# ====================== 前置检查 ======================
info "========== 开始执行 FFmpeg + UV 环境一键安装脚本 =========="

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    error "未检测到 uv 工具，请先执行：curl -LsSf https://astral.sh/uv/install.sh | sh 安装uv"
fi

# 检查wget是否安装
if ! command -v wget &> /dev/null; then
    error "未检测到 wget 工具，请先安装 wget：yum install wget -y 或 apt install wget -y"
fi

# ====================== 创建并激活UV虚拟环境 ======================
info "创建虚拟环境（Python ${PYTHON_VERSION}）"
if [ ! -d ".venv" ]; then
    uv venv --python "${PYTHON_VERSION}" || error "虚拟环境创建失败"
else
    warn "检测到已存在 .venv 虚拟环境，跳过创建步骤"
fi

info "激活虚拟环境"
# 定义虚拟环境路径变量
VIRTUAL_ENV="$(pwd)/.venv"
# 激活环境
source "${VIRTUAL_ENV}/bin/activate" || error "虚拟环境激活失败"

# ====================== 下载并解压FFmpeg ======================
FFMPEG_SRC="ffmpeg-${FFMPEG_VERSION}"
FFMPEG_TAR="${FFMPEG_SRC}.tar.bz2"
FFMPEG_URL="https://ffmpeg.org/releases/${FFMPEG_TAR}"

if [ ! -d "${FFMPEG_SRC}" ]; then
    info "下载 FFmpeg ${FFMPEG_VERSION} 源码包"
    wget "${FFMPEG_URL}" --no-check-certificate -q --show-progress || error "FFmpeg 下载失败"

    info "解压源码包"
    tar -xvf "${FFMPEG_TAR}" || error "源码包解压失败"
else
    warn "检测到 FFmpeg 源码目录，跳过下载解压步骤"
fi

# ====================== 编译安装FFmpeg ======================
info "进入 FFmpeg 源码目录，开始编译配置"
cd "${FFMPEG_SRC}" || exit 1

# 执行configure配置，安装到虚拟环境内
./configure \
    --enable-shared \
    --enable-pic \
    --disable-static \
    --prefix="${VIRTUAL_ENV}/ffmpeg" \
    --bindir="${VIRTUAL_ENV}/bin" \
    --libdir="${VIRTUAL_ENV}/lib" \
    --incdir="${VIRTUAL_ENV}/include" || error "FFmpeg configure 配置失败"

info "开始编译 FFmpeg（线程数：${MAKE_THREADS}）"
make -j "${MAKE_THREADS}" || error "FFmpeg 编译失败"

info "安装 FFmpeg 到虚拟环境"
make install || error "FFmpeg 安装失败"

# ====================== 配置环境变量 ======================
info "配置动态链接库环境变量到激活脚本"
# 追加配置，避免重复写入
ENV_LINE="export LD_LIBRARY_PATH=\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH"
if ! grep -qxF "${ENV_LINE}" "${VIRTUAL_ENV}/bin/activate"; then
    echo "${ENV_LINE}" >> "${VIRTUAL_ENV}/bin/activate"
else
    warn "环境变量配置已存在，跳过写入"
fi

# 重新激活环境，使配置生效
info "重新激活虚拟环境，加载新配置"
source "${VIRTUAL_ENV}/bin/activate"

# ====================== 安装验证 ======================
info "========== 开始验证安装结果 =========="
# 验证动态库文件
info "检查 FFmpeg 动态链接库"
ls "${VIRTUAL_ENV}/lib/libavformat.so" "${VIRTUAL_ENV}/lib/libavcodec.so" || error "动态库文件缺失"

# 验证ffmpeg版本
info "检查 FFmpeg 版本信息"
ffmpeg -version || error "FFmpeg 命令验证失败"

info "========== 全部安装完成！=========="
info "虚拟环境路径：${VIRTUAL_ENV}"
info "后续使用时，执行以下命令激活环境："
info "cd $(pwd)/../ && source .venv/bin/activate"