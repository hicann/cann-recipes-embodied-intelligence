#!/bin/bash
# Copyright (c) 2026 Institute of Software, Chinese Academy of Sciences (ISCAS). All rights reserved.
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

install_pi05_optional_deps() {
    progress "安装 Libero 仿真环境依赖（libero）..."

    if python_pkg_satisfies "hf-libero" ">=0.1.3,<0.2.0"; then
        info "hf-libero 版本满足要求，跳过安装 libero 额外依赖"
    else
        CMAKE_PREFIX_PATH="${CMAKE3_INSTALL_DIR}" \
        CMAKE="${CMAKE3_BIN}" \
        pinned_lerobot_pip_install -e ".[libero]" || error "Libero 依赖安装失败"
    fi

    progress "安装 Pi05 策略依赖（pi）..."

    if pi_transformers_ready && python_pkg_satisfies "scipy" ">=1.10.1,<1.15"; then
        info "检测到 PI05 所需 transformers 补丁与 scipy 版本均满足要求，跳过安装 pi 额外依赖"
    else
        CMAKE_PREFIX_PATH="${CMAKE3_INSTALL_DIR}" \
        CMAKE="${CMAKE3_BIN}" \
        pinned_lerobot_pip_install -e ".[pi]" || error "Pi05 依赖安装失败"
    fi
}

install_torchcodec_for_aarch64() {
    if [[ "$(uname -m)" == "aarch64" ]]; then
        progress "检测到 aarch64 架构，安装 torchcodec（transformers/AutoProcessor 导入所需）..."

        TORCHCODEC_LOCAL_DIR="${TORCHCODEC_LOCAL_DIR:-${ROOT_DIR}/torchcodec}"
        TORCHCODEC_BUILD_DIR="${LEROBOT_DIR}/torchcodec"
        TORCHCODEC_GIT_PRIMARY_URL="${TORCHCODEC_GIT_PRIMARY_URL:-https://gitcode.com/gh_mirrors/to/torchcodec.git}"
        TORCHCODEC_GIT_FALLBACK_URL="${TORCHCODEC_GIT_FALLBACK_URL:-https://github.com/meta-pytorch/torchcodec.git}"
        TORCHCODEC_SOURCE_DIR=""

        if python_pkg_exact_version "torchcodec" "${TORCHCODEC_VERSION}"; then
            info "torchcodec ${TORCHCODEC_VERSION} 已安装，跳过编译"
            return 0
        fi

        if python_pkg_installed "pybind11"; then
            info "pybind11 已安装，跳过安装"
        else
            info "正在安装 pybind11"
            pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pybind11 || error "pybind11 安装失败"
        fi

        FFMPEG_BIN="${CONDA_PREFIX}/bin/ffmpeg"
        PKG_CONFIG_BIN="${CONDA_PREFIX}/bin/pkg-config"
        FFMPEG_VERSION=$(${FFMPEG_BIN} -version 2>/dev/null | awk 'NR==1 {print $3}' | cut -d- -f1 || true)
        if [[ -x "${FFMPEG_BIN}" && -x "${PKG_CONFIG_BIN}" && "${FFMPEG_VERSION}" == "7.1.1" ]]; then
            info "ffmpeg 7.1.1 与 pkg-config 已安装，跳过 conda 安装"
        else
            info "通过 conda-forge 安装 torchcodec 所需 FFmpeg 与 pkg-config..."
            conda install -y -c conda-forge ffmpeg=7.1.1 pkg-config || error "FFmpeg/pkg-config 安装失败"
        fi

        export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig:${PKG_CONFIG_PATH:-}"
        if ! pkg-config --exists libavdevice libavfilter libavformat libavcodec libavutil libswresample libswscale; then
            error "未检测到 torchcodec 所需的 FFmpeg 开发库（libavdevice/libavfilter/libavformat/libavcodec/libavutil/libswresample/libswscale）。请确认 conda 环境中的 ffmpeg/pkg-config 安装成功，并检查 PKG_CONFIG_PATH。"
        fi

        if [[ -f "${TORCHCODEC_BUILD_DIR}/pyproject.toml" ]]; then
            TORCHCODEC_SOURCE_DIR="${TORCHCODEC_BUILD_DIR}"
            info "检测到当前 LeRobot 目录下已有完整 torchcodec 源码，直接复用：${TORCHCODEC_SOURCE_DIR}"
        elif [[ -d "${TORCHCODEC_BUILD_DIR}" ]]; then
            warn "检测到不完整的 torchcodec 目录：${TORCHCODEC_BUILD_DIR}，将删除后重新准备源码"
            rm -rf "${TORCHCODEC_BUILD_DIR}"
        fi

        if [[ -z "${TORCHCODEC_SOURCE_DIR}" && -f "${TORCHCODEC_LOCAL_DIR}/pyproject.toml" ]]; then
            TORCHCODEC_SOURCE_DIR="${TORCHCODEC_LOCAL_DIR}"
            info "检测到本地 torchcodec 缓存，直接复用：${TORCHCODEC_SOURCE_DIR}"
        fi

        if [[ -z "${TORCHCODEC_SOURCE_DIR}" ]]; then
            for TORCHCODEC_GIT_URL in "${TORCHCODEC_GIT_PRIMARY_URL}" "${TORCHCODEC_GIT_FALLBACK_URL}"; do
                [[ -z "${TORCHCODEC_GIT_URL}" ]] && continue
                info "尝试克隆 torchcodec 源码：${TORCHCODEC_GIT_URL}"
                if git clone --branch "${TORCHCODEC_GIT_REF}" --depth 1 "${TORCHCODEC_GIT_URL}" "${TORCHCODEC_BUILD_DIR}"; then
                    TORCHCODEC_SOURCE_DIR="${TORCHCODEC_BUILD_DIR}"
                    success "torchcodec ${TORCHCODEC_GIT_REF} 源码获取成功：${TORCHCODEC_GIT_URL}"
                    break
                fi
                warn "torchcodec 克隆失败：${TORCHCODEC_GIT_URL}"
                rm -rf "${TORCHCODEC_BUILD_DIR}"
            done
        fi

        if [[ -z "${TORCHCODEC_SOURCE_DIR}" ]]; then
            error "torchcodec 源码准备失败。可通过 TORCHCODEC_LOCAL_DIR 指定本地缓存目录，或通过 TORCHCODEC_GIT_PRIMARY_URL 指定可用的国内镜像地址。"
        fi

        if [[ -d "${TORCHCODEC_SOURCE_DIR}/.git" ]]; then
            git -C "${TORCHCODEC_SOURCE_DIR}" fetch --tags --force >/dev/null 2>&1 || true
            git -C "${TORCHCODEC_SOURCE_DIR}" checkout -f "${TORCHCODEC_GIT_REF}" >/dev/null 2>&1 || error "无法切换 torchcodec 到 ${TORCHCODEC_GIT_REF}"
        fi

        TORCHCODEC_VERSION_FILE="${TORCHCODEC_SOURCE_DIR}/version.txt"
        if [[ ! -f "${TORCHCODEC_VERSION_FILE}" ]]; then
            warn "未找到 torchcodec version.txt：${TORCHCODEC_VERSION_FILE}，跳过版本校验"
        fi
        if [[ -f "${TORCHCODEC_VERSION_FILE}" ]]; then
            TORCHCODEC_SOURCE_VERSION=$(tr -d '[:space:]' < "${TORCHCODEC_VERSION_FILE}")
        else
            TORCHCODEC_SOURCE_VERSION="${TORCHCODEC_VERSION}"
        fi
        if [[ "${TORCHCODEC_SOURCE_VERSION}" != "${TORCHCODEC_VERSION}" ]]; then
            error "torchcodec 源码版本为 ${TORCHCODEC_SOURCE_VERSION}，与要求的 ${TORCHCODEC_VERSION} 不一致。请切换到 ${TORCHCODEC_GIT_REF} 后重试。"
        fi

        cd "${TORCHCODEC_SOURCE_DIR}"
        info "正在编译并安装 torchcodec ${TORCHCODEC_VERSION}，源码目录：${TORCHCODEC_SOURCE_DIR}"
        PYBIND11_CMAKE_DIR=$(python -m pybind11 --cmakedir) || error "无法获取 pybind11 CMake 目录"
        if TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=1 \
            CMAKE="${CMAKE3_BIN}" \
            CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}:${CMAKE3_INSTALL_DIR}:${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}" \
            pip install -e . --no-build-isolation -v; then
            success "torchcodec ${TORCHCODEC_VERSION} 安装成功"
        else
            error "torchcodec 安装失败。当前推荐组合为 torchcodec ${TORCHCODEC_VERSION} + ffmpeg 7.1.1 + TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=1，请先排查编译依赖后重试"
        fi
        cd "${LEROBOT_DIR}"
    else
        info "当前架构非 aarch64（$(uname -m)），跳过 torchcodec 编译"
    fi
}

install_torch_npu_if_enabled() {
    if [[ "${SKIP_TORCH_NPU}" == false ]]; then
        if [[ "$(uname -m)" == "aarch64" ]]; then
            progress "安装固定版本 torch_npu（aarch64 架构）..."

            PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
            if [[ "${PYTORCH_VERSION}" != "${TORCH_VERSION}"* ]]; then
                error "检测到 PyTorch 版本 ${PYTORCH_VERSION}，与要求的 ${TORCH_VERSION} 不一致，请检查前面的安装步骤"
            fi

            INSTALLED_TORCH_NPU_VERSION=$(python - <<'EOF'
import importlib.metadata as md
try:
    print(md.version('torch-npu'))
except Exception:
    print('')
EOF
)
            progress "安装 torch_npu 运行时依赖..."
            for RUNTIME_DEP in "${TORCH_NPU_RUNTIME_DEPS[@]}"; do
                if python_pkg_installed "${RUNTIME_DEP}"; then
                    info "检测到 ${RUNTIME_DEP} 已安装，跳过安装"
                else
                    pip install "${RUNTIME_DEP}" || error "${RUNTIME_DEP} 安装失败"
                fi
            done

            for OPTIONAL_DEP in "${TORCH_NPU_RUNTIME_OPTIONAL_DEPS[@]}"; do
                if python - <<EOF >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("absl") else 1)
EOF
                then
                    info "检测到 ${OPTIONAL_DEP} 已存在，跳过覆盖安装"
                else
                    pip install --ignore-installed "${OPTIONAL_DEP}" || error "${OPTIONAL_DEP} 安装失败"
                fi
            done

            if [[ "${INSTALLED_TORCH_NPU_VERSION}" == "${TORCH_NPU_VERSION}" ]]; then
                info "torch_npu ${TORCH_NPU_VERSION} 已安装，跳过安装"
            else
                pip install --upgrade "torch-npu==${TORCH_NPU_VERSION}" || error "torch_npu 安装失败"
                success "torch_npu 安装成功"
            fi

            python - <<'EOF' || error "torch_npu 运行时校验失败"
import torch
import torch_npu
print('torch', torch.__version__)
print('torch_npu ok')
print('npu available', torch.npu.is_available())
EOF
        else
            warn "当前架构为 $(uname -m)，不支持安装 torch_npu（仅 aarch64），已自动跳过"
        fi
    else
        info "已跳过 torch_npu 安装"
    fi
}
