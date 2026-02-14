#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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
# 记录cosmos-predict2.5根目录（脚本执行的初始目录）
TARGET_ROOT=$(pwd)
# 编译线程数，根据服务器CPU核心数调整
MAKE_THREADS=64
# 依赖包版本配置
SETUPTOOLS_VER=80.9.0
NUMPY_VER=2.4.1
WHEEL_VER=0.41.2
# Decord仓库地址
DECORD_REPO=https://github.com/dmlc/decord
# 目标wheel存放目录（相对路径，适配同级目录结构，packages目录和decord目录同级，都在当前cosmos-predict2.5目录下）
WHEEL_TARGET_DIR="${TARGET_ROOT}/packages/cosmos-oss/wheels"
# 虚拟环境路径（绝对路径）
VENV_DIR="${TARGET_ROOT}/.venv"
# ======================================================================

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 清除颜色

# 日志打印函数
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

# ====================== 前置校验 ======================
info "========== 启动 Decord 编译+打包+迁移一键脚本 =========="

# 校验核心工具是否安装
check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "未检测到依赖工具：$1，请先安装后重试"
    fi
}
check_command git
check_command cmake
check_command make
check_command uv
check_command g++


# 校验Python版本是否存在
info "校验Python ${PYTHON_VERSION} 是否可用"
if ! uv python find "${PYTHON_VERSION}" &> /dev/null; then
    error "未找到Python ${PYTHON_VERSION}，请先安装该版本Python"
fi

# ====================== 创建并激活UV虚拟环境 ======================
info "创建虚拟环境（Python ${PYTHON_VERSION}）"
if [ ! -d "${VENV_DIR}" ]; then
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}" || error "虚拟环境创建失败"
else
    warn "检测到已存在 .venv 虚拟环境，跳过创建步骤"
fi

info "激活虚拟环境"
# 激活环境（source会自动设置VIRTUAL_ENV变量）
source "${VENV_DIR}/bin/activate" || error "虚拟环境激活失败"
info "当前激活的虚拟环境：${VIRTUAL_ENV}"


# ====================== 克隆源码（指定commit版本） ======================
DECORD_DIR="${TARGET_ROOT}/decord" # 绝对路径
DECORD_COMMIT="d2e56190286ae394032a8141885f76d5372bd44b" # 目标commit ID

if [[ ! -d "${DECORD_DIR}" ]]; then
    info "关闭Git SSL证书校验"
    git config --global http.sslVerify false

    info "克隆decord源码并切换到指定commit: ${DECORD_COMMIT}"
    # 保留链式写法，补充关键参数
    git clone --recursive "${DECORD_REPO}" "${DECORD_DIR}" && \
    cd "${DECORD_DIR}" && \
    git checkout "${DECORD_COMMIT}" && \
    git submodule update --init --recursive || error "Decord克隆/切换commit失败"
else
    warn "检测到decord源码目录已存在，跳过克隆步骤（如需切换版本请手动执行git checkout ${DECORD_COMMIT}）"
fi


# ====================== 编译C++核心库 ======================
BUILD_DIR="build"
if [[ ! -d "${BUILD_DIR}" ]]; then
    info "创建编译目录：${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
fi
cd "${BUILD_DIR}" || error "进入build目录失败"

info "执行CMake配置编译参数"
PKG_CONFIG_PATH="${VIRTUAL_ENV}/lib/pkgconfig" \
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
-DBUILD_SHARED_LIBS=ON \
-DCMAKE_FIND_USE_PKG_CONFIG=ON || error "CMake配置执行失败"

info "开始编译Decord核心库（线程数：${MAKE_THREADS}）"
make -j "${MAKE_THREADS}" || error "Decord编译失败"

# ====================== 安装Python绑定 ======================
info "进入Python模块目录"
cd ../python || error "进入python目录失败"

info "安装指定版本Python依赖"
uv pip install setuptools=="${SETUPTOOLS_VER}" numpy=="${NUMPY_VER}" wheel=="${WHEEL_VER}" || error "Python依赖安装失败"

info "执行setup.py安装Python接口"
python setup.py install || error "Python模块安装失败"

info "回到build目录执行库文件安装"
cd ../build || error "返回build目录失败"
make install -j "${MAKE_THREADS}" || error "动态库安装失败"

# ====================== 验证基础安装 ======================
info "验证Decord基础安装状态"
python -c "import decord; print('Decord 基础安装成功，版本：', decord.__version__)" || error "Decord导入验证失败"

# ====================== 打包为wheel文件 ======================
info "进入python目录准备打包"
cd ../python || error "进入python目录失败"

info "构建aarch64架构wheel包"
python setup.py bdist_wheel || error "wheel打包失败"

# 获取生成的wheel文件完整路径
WHEEL_FILE=$(ls ./dist/decord-*.whl 2>/dev/null | head -n 1)
if [[ -z "${WHEEL_FILE}" || ! -f "${WHEEL_FILE}" ]]; then
    error "未在dist目录下找到生成的wheel包"
fi
info "成功生成wheel包：${WHEEL_FILE}"

# ====================== 迁移wheel包到目标目录 ======================
info "创建目标wheels目录（不存在则自动创建）"
mkdir -p "${WHEEL_TARGET_DIR}"

info "迁移wheel包至项目目录"
cp "${WHEEL_FILE}" "${WHEEL_TARGET_DIR}/" || error "wheel包复制失败"

# ====================== 最终验证与收尾 ======================
info "========== 全流程执行完成 =========="
info "Decord 编译安装状态：已验证可用"
info "wheel包路径：${WHEEL_TARGET_DIR}/$(basename ${WHEEL_FILE})"
info "后续操作：修改pyproject.toml配置后执行 uv sync 即可"


# ====================== 替换decord-*.whl文件名到pyproject.toml中 ========
# 先回到cosmos-predict2.5根目录，再进入packages
cd "${TARGET_ROOT}/packages/cosmos-oss" || error "进入cosmos-oss目录失败"

# 获取当前架构
ARCH=$(uname -m)
# 配置文件路径
TOML_FILE="pyproject.toml"
# whl目录
WHL_DIR="wheels"

# 根据架构匹配对应whl包
info "检测当前架构：${ARCH}"
if [ "$ARCH" = "aarch64" ]; then
    WHL_NAME=$(ls ${WHL_DIR}/decord-*_aarch64.whl 2>/dev/null | head -n 1)
elif [ "$ARCH" = "x86_64" ]; then
    WHL_NAME=$(ls ${WHL_DIR}/decord-*_x86_64.whl 2>/dev/null | head -n 1)
else
    error "不支持的架构：${ARCH}（仅支持aarch64/x86_64）"
fi

# 校验文件存在
if [ ! -f "$WHL_NAME" ]; then
    echo "未找到对应架构的decord whl包"
    exit 1
fi

# 使用sed替换pyproject.toml中的decord path配置
# 匹配原有配置行，自动更新为当前架构的whl路径
# 添加.bak备份，兼容所有Linux发行版
sed -i.bak "/decord = { path/c\decord = { path = \"${WHL_NAME}\" }" ${TOML_FILE}
rm -f "${TOML_FILE}.bak" # 清理备份文件

echo "已自动配置decord源：${WHL_NAME}"