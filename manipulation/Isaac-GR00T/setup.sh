#!/bin/bash
# coding=utf-8
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

# ====================== 配置项 ======================
PYTHON_VERSION="3.10.19"
FFMPEG_VERSION="4.4.2"
MAKE_THREADS=64
PROJECT_ROOT=$(pwd)
WHEELS_DIR="${PROJECT_ROOT}/wheels"
# ====================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; exit 1; }

info "========== 开始全自动环境部署 =========="

# 前置依赖检查
for cmd in uv git cmake pkg-config wget; do
    command -v $cmd &> /dev/null || error "缺少必要工具: $cmd，请先安装 (e.g., yum/apt install $cmd)"
done

# 虚拟环境准备
if [ ! -d ".venv" ]; then
    info "创建虚拟环境 (Python ${PYTHON_VERSION})..."
    uv venv --python "${PYTHON_VERSION}"
fi
source .venv/bin/activate
VIRTUAL_ENV=$(pwd)/.venv

# FFmpeg 安装
if [ -f "${VIRTUAL_ENV}/lib/libavcodec.so" ]; then
    warn "检测到 FFmpeg 已安装在虚拟环境中，跳过编译步骤。"
else
    info "开始下载并编译 FFmpeg ${FFMPEG_VERSION}..."
    FFMPEG_SRC="ffmpeg-${FFMPEG_VERSION}"
    if [ ! -d "${FFMPEG_SRC}" ]; then
        wget "https://ffmpeg.org/releases/${FFMPEG_SRC}.tar.bz2" --no-check-certificate -q --show-progress
        tar -xjf "${FFMPEG_SRC}.tar.bz2"
    fi
    
    cd "${FFMPEG_SRC}"
    ./configure \
        --enable-shared --enable-pic --disable-static \
        --prefix="${VIRTUAL_ENV}" \
        --libdir="${VIRTUAL_ENV}/lib" \
        --incdir="${VIRTUAL_ENV}/include"
    make -j "${MAKE_THREADS}" && make install
    cd "${PROJECT_ROOT}"
    info "FFmpeg 安装完成。"
fi

# 配置环境变量
export PKG_CONFIG_PATH="${VIRTUAL_ENV}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Decord 编译与打包
if [ -d "decord/python/dist" ] && [ "$(ls -A decord/python/dist/*.whl 2>/dev/null)" ]; then
    warn "检测到已存在 Decord Wheel 包，准备直接同步。"
else
    info "开始编译 Decord..."
    [ ! -d "decord" ] && git clone --recursive https://github.com/dmlc/decord --depth 1
    
    cd decord
    rm -rf build && mkdir build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_FIND_USE_PKG_CONFIG=ON
    make -j "${MAKE_THREADS}" && make install
    
    info "正在将 Decord 打包为 Wheel..."
    cd ../python
    uv pip install setuptools wheel
    python setup.py bdist_wheel
    cd "${PROJECT_ROOT}"
fi

info "========== 正在注入依赖并同步项目 =========="

WHEEL_PATH=$(ls "${PROJECT_ROOT}"/decord/python/dist/decord-*.whl | head -n 1)
mkdir -p "${WHEELS_DIR}"
cp "${WHEEL_PATH}" "${WHEELS_DIR}/"
WHEEL_NAME=$(basename "${WHEEL_PATH}")

# 使用 uv add 自动修改 pyproject.toml
info "执行 uv add 注入本地 Wheel..."
uv add "decord @ file://${WHEELS_DIR}/${WHEEL_NAME}"

# 执行同步
info "同步虚拟环境..."
uv sync

info "环境已全部就绪。"
