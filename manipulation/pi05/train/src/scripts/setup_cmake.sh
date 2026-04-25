#!/bin/bash
# Copyright (c) 2026 Institute of Software, Chinese Academy of Sciences (ISCAS). All rights reserved.
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

CMAKE3_INSTALL_DIR="${CMAKE3_INSTALL_DIR:-${HOME}/.local/cmake3}"
CMAKE3_BIN="${CMAKE3_INSTALL_DIR}/bin/cmake"
CMAKE3_VERSION="${CMAKE3_VERSION:-3.28.3}"
CMAKE3_TEMP_DIR="${HOME}/cmake3_temp"
CMAKE3_TAR_FILE="${CMAKE3_TEMP_DIR}/cmake-${CMAKE3_VERSION}.tar.gz"
CMAKE3_OFFICIAL_URL="${CMAKE3_OFFICIAL_URL:-https://cmake.org/files/v3.28/cmake-${CMAKE3_VERSION}.tar.gz}"
CMAKE3_MIRROR_URL="${CMAKE3_MIRROR_URL:-https://mirrors.tuna.tsinghua.edu.cn/kitware/cmake/v3.28/cmake-${CMAKE3_VERSION}.tar.gz}"

install_cmake3() {
    info "开始自动安装用户级 CMake ${CMAKE3_VERSION}..."

    if ! command -v gcc &>/dev/null || ! command -v g++ &>/dev/null; then
        error "未检测到 gcc/g++，无法编译 CMake！请联系管理员安装 gcc/g++ 后重试"
    fi

    mkdir -p "${CMAKE3_TEMP_DIR}" && cd "${CMAKE3_TEMP_DIR}" || error "创建临时目录失败"

    info "检查 CMake ${CMAKE3_VERSION} 源码包缓存..."
    validate_cmake_tarball() {
        tar -tzf "cmake-${CMAKE3_VERSION}.tar.gz" >/dev/null 2>&1
    }

    if [[ -f "${CMAKE3_TAR_FILE}" ]]; then
        if validate_cmake_tarball; then
            info "发现可用的 CMake 源码包缓存：${CMAKE3_TAR_FILE}，跳过下载"
        else
            warn "检测到损坏的 CMake 源码包缓存：${CMAKE3_TAR_FILE}，将重新下载"
            rm -f "${CMAKE3_TAR_FILE}"
        fi
    fi

    if [[ ! -f "${CMAKE3_TAR_FILE}" ]]; then
        info "下载 CMake ${CMAKE3_VERSION} 源码..."

        download_cmake() {
            local url="$1"
            if command -v wget >/dev/null 2>&1; then
                wget --progress=bar:force -c -O "cmake-${CMAKE3_VERSION}.tar.gz" "${url}"
                return $?
            elif command -v curl >/dev/null 2>&1; then
                curl -# -L -C - -o "cmake-${CMAKE3_VERSION}.tar.gz" "${url}"
                return $?
            else
                error "未检测到 wget/curl，无法下载文件！"
                return 1
            fi
        }

        CMAKE3_DOWNLOAD_URLS=("${CMAKE3_OFFICIAL_URL}" "${CMAKE3_MIRROR_URL}")
        download_success=false

        for CMAKE3_URL in "${CMAKE3_DOWNLOAD_URLS[@]}"; do
            [[ -z "${CMAKE3_URL}" ]] && continue
            info "尝试下载 CMake 源码：${CMAKE3_URL}"
            if download_cmake "${CMAKE3_URL}" && validate_cmake_tarball; then
                success "CMake 源码下载成功：${CMAKE3_URL}"
                download_success=true
                break
            fi
            warn "CMake 下载或校验失败：${CMAKE3_URL}，尝试下一个地址"
            rm -f "cmake-${CMAKE3_VERSION}.tar.gz"
        done

        if [[ "${download_success}" != true ]]; then
            error "CMake 源码下载失败！请检查网络，或手动下载后放到 ${CMAKE3_TEMP_DIR}。官方地址：${CMAKE3_OFFICIAL_URL}，国内镜像：${CMAKE3_MIRROR_URL}"
        fi
    fi

    info "解压 CMake 源码包..."
    tar -zxf "cmake-${CMAKE3_VERSION}.tar.gz" && cd "cmake-${CMAKE3_VERSION}" || error "解压失败"

    info "配置 CMake 编译参数..."
    ./bootstrap --prefix="${CMAKE3_INSTALL_DIR}" \
                --no-system-curl \
                --no-system-zlib \
                --no-system-bzip2 || error "CMake 配置失败"

    info "编译 CMake ${CMAKE3_VERSION}..."
    make -j"$(nproc)" || error "CMake 编译失败"

    info "安装 CMake ${CMAKE3_VERSION} 到 ${CMAKE3_INSTALL_DIR}..."
    make install || error "CMake 安装失败"

    cd "${HOME}" && rm -rf "${CMAKE3_TEMP_DIR}/cmake-${CMAKE3_VERSION}"

    if ! "${CMAKE3_BIN}" --version >/dev/null 2>&1; then
        error "CMake ${CMAKE3_VERSION} 安装后验证失败！"
    fi

    info "CMake ${CMAKE3_VERSION} 安装成功！"
}

setup_cmake3_if_needed() {
    progress "检查并准备 CMake ${CMAKE3_VERSION}..."

    CMAKE_VERSION=""
    SKIP_CMAKE_INSTALL=false

    if [[ -f "${CMAKE3_BIN}" ]]; then
        INSTALLED_CMAKE_VERSION=$("${CMAKE3_BIN}" --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        if [[ "${INSTALLED_CMAKE_VERSION}" == "${CMAKE3_VERSION}" ]]; then
            info "用户目录已安装 CMake ${CMAKE3_VERSION}，直接使用"
            CMAKE3_BIN="${CMAKE3_INSTALL_DIR}/bin/cmake"
            SKIP_CMAKE_INSTALL=true
        else
            info "用户目录安装的 CMake 版本为 ${INSTALLED_CMAKE_VERSION}，非目标版本 ${CMAKE3_VERSION}"
            SKIP_CMAKE_INSTALL=false
        fi
    else
        SKIP_CMAKE_INSTALL=false
    fi

    if [[ "${SKIP_CMAKE_INSTALL}" == false ]]; then
        if command -v cmake >/dev/null 2>&1; then
            CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
            MAJOR_VERSION=$(echo "${CMAKE_VERSION}" | cut -d. -f1)
        fi

        if [[ -z "${CMAKE_VERSION}" || "${MAJOR_VERSION}" -ge 4 ]]; then
            if [[ -n "${CMAKE_VERSION}" ]]; then
                warn "检测到 CMake ${CMAKE_VERSION} 不兼容，自动安装 ${CMAKE3_VERSION}..."
            else
                warn "未检测到 CMake，自动安装 ${CMAKE3_VERSION}..."
            fi
            install_cmake3
            SKIP_CMAKE_INSTALL=true
        else
            info "CMake 版本 ${CMAKE_VERSION} 兼容，无需安装"
            CMAKE3_BIN="cmake"
            SKIP_CMAKE_INSTALL=true
        fi
    fi

    export PATH="${CMAKE3_INSTALL_DIR}/bin:${PATH}"

    CURRENT_CMAKE=$(which cmake)
    CURRENT_CMAKE_VERSION=$("${CURRENT_CMAKE}" --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    info "当前生效的 CMake 路径：${CURRENT_CMAKE}，版本：${CURRENT_CMAKE_VERSION}"

    FINAL_MAJOR_VERSION=$(echo "${CURRENT_CMAKE_VERSION}" | cut -d. -f1)
    if [[ "${FINAL_MAJOR_VERSION}" -ne 3 ]]; then
        error "CMake 版本验证失败！当前版本：${CURRENT_CMAKE_VERSION}，请手动检查"
    fi

    progress "CMake 版本 ${CURRENT_CMAKE_VERSION} 兼容，继续安装 Pi05 依赖..."
}
