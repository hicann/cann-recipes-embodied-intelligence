#!/bin/bash

set -euo pipefail

PYTHON_VERSION="3.10"
FFMPEG_VERSION="4.4.2"
MAKE_THREADS=64
PROJECT_ROOT=$(pwd)
WHEELS_DIR="${PROJECT_ROOT}/wheels"
CONDA_ENV_NAME="gr00t"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; exit 1; }

install() {
    local pkg=$1
    if [ -f /etc/redhat-release ] || [ -f /etc/openEuler-release ]; then
        if [ "$pkg" = "g++" ]; then
            yum install -y gcc-c++
        else
            yum install -y "$pkg"
        fi
    else
        apt-get update
        apt-get install -y "$pkg"
    fi
}

info "========== Automatic Environment Deployment =========="

# 自动检测并安装依赖
for cmd in git cmake pkg-config wget gcc g++; do
    if ! command -v "$cmd" &> /dev/null; then
        info "Missing $cmd, installing..."
        install "$cmd"
    fi
done

command -v conda &> /dev/null || error "Conda is not installed"

info "Initializing Conda..."

CONDA_BASE=$(conda info --base)
# Disable 'set -u' temporarily as conda.sh references PS1 which may be unbound in non-interactive shells
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set -u

info "Checking Conda environment: ${CONDA_ENV_NAME}"
if ! conda env list | grep -q "^${CONDA_ENV_NAME}"; then
    info "Creating Conda environment (Python ${PYTHON_VERSION})..."
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
fi

info "Activating Conda environment: ${CONDA_ENV_NAME}"
set +u
conda activate ${CONDA_ENV_NAME}
set -u

VIRTUAL_ENV=$(conda info --base)/envs/${CONDA_ENV_NAME}
info "Conda environment path: ${VIRTUAL_ENV}"

if [ -f "${VIRTUAL_ENV}/lib/libavcodec.so" ]; then
    warn "FFmpeg is already installed, skipping compilation"
else
    info "Downloading and compiling FFmpeg ${FFMPEG_VERSION}..."
    FFMPEG_SRC="ffmpeg-${FFMPEG_VERSION}"
    if [ ! -d "${FFMPEG_SRC}" ]; then
        wget "https://ffmpeg.org/releases/${FFMPEG_SRC}.tar.bz2" -q --show-progress
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
    info "FFmpeg installation completed"
fi

info "Configuring environment variables..."
export PKG_CONFIG_PATH="${VIRTUAL_ENV}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

if [ -d "decord/python/dist" ] && [ "$(ls -A decord/python/dist/*.whl 2>/dev/null)" ]; then
    warn "Decord wheel detected, skipping compilation"
else
    info "Compiling Decord..."
    [ ! -d "decord" ] && git clone --recursive https://github.com/dmlc/decord --depth 1

    cd decord
    rm -rf build && mkdir build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_FIND_USE_PKG_CONFIG=ON \
        -DPYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
    make -j "${MAKE_THREADS}" && make install

    info "Packaging Decord wheel..."
    cd ../python
    pip install setuptools wheel --trusted-host pypi.org --trusted-host files.pythonhosted.org
    python setup.py bdist_wheel
    cd "${PROJECT_ROOT}"
    info "Decord compilation completed"
fi

info "Installing local Decord wheel..."
WHEEL_PATH=$(ls "${PROJECT_ROOT}"/decord/python/dist/decord-*.whl | head -n 1)
mkdir -p "${WHEELS_DIR}"
cp -f "${WHEEL_PATH}" "${WHEELS_DIR}/"

pip install "${WHEEL_PATH}" --trusted-host pypi.org --trusted-host files.pythonhosted.org

info "Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt \
        --trusted-host download.mindspore.cn \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org
else
    error "requirements.txt not found!"
fi

info "========== Deployment Completed =========="
info "Python version: $(python --version)"
info "Verifying NPU status..."
python -c "
try:
    import torch
    import torch_npu
    print('Ascend NPU available:', torch.npu.is_available())
    print('torch version:', torch.__version__)
    print('torch-npu version:', torch_npu.__version__)
except ImportError:
    print('torch-npu not detected, please check Ascend drivers')
"
info "Environment ready! Activate with: conda activate ${CONDA_ENV_NAME}"
