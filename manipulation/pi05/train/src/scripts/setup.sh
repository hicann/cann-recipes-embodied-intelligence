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

# ===================== 脚本配置与帮助信息 =====================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "一键部署 Pi05 项目环境"
    echo ""
    echo "Options:"
    echo "  -h, --help    显示此帮助信息并退出"
    echo "  --skip-torch-npu  跳过 torch_npu 安装"
    echo ""
    echo "前置要求："
    echo "  1. 已安装 conda 并配置好镜像源"
    echo "  2. 网络可访问 GitHub/GitCode/Ascend 仓库"
}

# 解析命令行参数
SKIP_TORCH_NPU=false
for arg in "$@"; do
    case $arg in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-torch-npu)
            SKIP_TORCH_NPU=true
            shift
            ;;
        *)
            error "未知参数：$arg，请使用 -h 查看帮助"
            ;;
    esac
done

# ===================== 自动获取当前路径 =====================
ROOT_DIR="$(pwd)"
LEROBOT_DIR="${ROOT_DIR}/lerobot"
CANN_RECIPES_DIR="${ROOT_DIR}/cann-recipes-embodied-intelligence"

# 固定 LeRobot 仓库版本
LEROBOT_COMMIT="58f70b6bd370864139a3795ac3497a9eae8c42d5"

# ===================== 工具函数 =====================
# 带时间戳的日志函数
info() {
    echo -e "\033[32m[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1\033[0m"
}

progress() {
    echo -e "\033[34m[$(date +'%Y-%m-%d %H:%M:%S')] [PROGRESS] $1\033[0m"
}

success() {
    echo -e "\033[32;1m[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS] $1\033[0m"
}

error() {
    echo -e "\033[31;1m[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1\033[0m" >&2
    exit 1
}

warn() {
    echo -e "\033[33m[$(date +'%Y-%m-%d %H:%M:%S')] [WARN] $1\033[0m"
}

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "命令 '$1' 未找到，请先安装"
    fi
}

# 检查文件/目录是否存在
check_path() {
    local path="$1"
    local type="$2" # file/dir
    local msg="$3"
    
    if [[ "$type" == "file" && ! -f "$path" ]]; then
        error "${msg}：文件不存在 ${path}"
    elif [[ "$type" == "dir" && ! -d "$path" ]]; then
        error "${msg}：目录不存在 ${path}"
    fi
}

# ===================== 前置检查 =====================
info "开始执行 Pi05 代码仓、模型、数据集一键下载脚本"
info "工作目录：${ROOT_DIR}"

# 检查基础命令
check_command "git"
check_command "cp"
check_command "conda"
check_command "pip"

# 检查 cann-recipes 目录是否存在
if [[ ! -d "${CANN_RECIPES_DIR}" ]]; then
    info "未检测到 cann-recipes-embodied-intelligence 目录，开始自动克隆..."
    git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git "${CANN_RECIPES_DIR}" || error "CANN Recipes 仓库克隆失败"
else
    info "检测到 cann-recipes-embodied-intelligence 目录已存在"
fi

# 检查 Pi05 源文件是否存在
check_path "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/modeling_pi05.py" "file" "Pi05 核心模型文件缺失"
check_path "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_train.py" "file" "Pi05 训练脚本缺失"

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
    info "lerobot 目录已存在，更新代码"
    git fetch --all || error "LeRobot 仓库 fetch 失败"
else
    git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}" || error "LeRobot 仓库克隆失败"
    cd "${LEROBOT_DIR}"
fi

git reset --hard "${LEROBOT_COMMIT}" || error "无法回退到指定提交：${LEROBOT_COMMIT}（请检查 commit 是否存在）"
success "LeRobot 代码仓库已锁定至指定版本"

# ===================== 步骤 2：复制 Pi05 相关文件 =====================
progress "复制 Pi05 实现文件至 LeRobot 项目中..."
# 确保目标子目录存在
mkdir -p "${LEROBOT_DIR}/src/lerobot/policies/pi05/"
mkdir -p "${LEROBOT_DIR}/src/lerobot/scripts/"
mkdir -p "${LEROBOT_DIR}/src/lerobot/utils/"
mkdir -p "${LEROBOT_DIR}/src/lerobot/configs/"

# 复制文件
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/modeling_pi05.py" "${LEROBOT_DIR}/src/lerobot/policies/pi05/" || error "复制 modeling_pi05.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_train.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 lerobot_train.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_eval.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 lerobot_eval.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_train_profiling.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 run_train_profiling.py 失败"
cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/utils.py" "${LEROBOT_DIR}/src/lerobot/utils/" || error "复制 utils.py 失败"

# 处理空目录情况
if compgen -G "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/configs/*.yaml" > /dev/null; then
    cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/configs/"*.yaml "${LEROBOT_DIR}/src/lerobot/configs/" || error "复制配置文件失败"
else
    warn "Pi05 配置文件目录为空，跳过配置文件复制"
fi

success "Pi05 相关文件复制完成"

# ===================== 步骤 3：创建并激活 Conda 虚拟环境 =====================
progress "创建 Conda 虚拟环境 'lerobot' (Python 3.10)..."

# 检查环境是否已存在
if conda info --envs | grep -q "lerobot"; then
    info "lerobot 环境已存在，跳过创建"
else
    # 指定 conda 镜像源
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --set show_channel_urls yes
    
    conda create -y -n lerobot python=3.10 || error "Conda 环境创建失败"
fi

# 使用 eval 确保在非交互式 shell 中正确激活环境
eval "$(conda shell.bash hook)"
conda activate lerobot || error "无法激活 Conda 环境 'lerobot'"

cd "${LEROBOT_DIR}"

# 安装基础依赖（指定镜像源）
progress "安装 LeRobot 基础依赖..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e . || error "LeRobot 基础安装失败"

# ===================== 步骤 4：安装 Git LFS =====================
progress "配置 Git LFS..."

# 定义标记变量：标识 Git LFS 是否安装成功
GIT_LFS_INSTALLED=false

# 先检查 Git LFS 是否已安装
if command -v git-lfs >/dev/null 2>&1; then
    info "Git LFS 已安装"
    GIT_LFS_INSTALLED=true
else
    info "Git LFS 未安装，正在尝试自动安装..."
    
    # 定义提权函数
    run_with_sudo() {
        local cmd="$*"
        if command -v sudo >/dev/null 2>&1; then
            # 有 sudo 则用 sudo 执行
            sudo $cmd
            return $?
        else
            # 无 sudo 仅警告，不退出
            warn "需要 root 权限执行命令：$cmd，但未检测到 sudo，请手动安装 Git LFS"
            return 1
        fi
    }

    # 定义不同系统的安装函数
    install_with_apt() {
        info "检测到 Debian/Ubuntu 系统，使用 apt 安装 Git LFS..."
        if ! run_with_sudo "apt install -y git-lfs" >/dev/null 2>&1; then
            info "首次安装失败，尝试更新包列表后重试..."
            if ! run_with_sudo "apt update -y" >/dev/null 2>&1; then
                warn "apt 更新失败，无法继续安装 Git LFS"
                return 1
            fi
            if ! run_with_sudo "apt install -y git-lfs" >/dev/null 2>&1; then
                warn "apt 安装 Git LFS 失败"
                return 1
            fi
        fi
        return 0
    }

    install_with_yum() {
        info "检测到 CentOS/RHEL 系统，使用 yum 安装 Git LFS..."
        if ! run_with_sudo "yum install -y git-lfs" >/dev/null 2>&1; then
            warn "yum 安装 Git LFS 失败"
            return 1
        fi
        return 0
    }

    install_with_dnf() {
        info "检测到 Fedora/RHEL 8+ 系统，使用 dnf 安装 Git LFS..."
        if ! run_with_sudo "dnf install -y git-lfs" >/dev/null 2>&1; then
            warn "dnf 安装 Git LFS 失败"
            return 1
        fi
        return 0
    }

    install_with_brew() {
        info "检测到 macOS 系统，使用 brew 安装 Git LFS..."
        # brew 安装不需要 root 权限，直接执行
        if ! brew install git-lfs >/dev/null 2>&1; then
            warn "brew 安装 Git LFS 失败"
            return 1
        fi
        return 0
    }

    # 自动检测包管理器并安装
    install_success=false
    if command -v apt >/dev/null 2>&1; then
        if install_with_apt; then
            install_success=true
        fi
    elif command -v dnf >/dev/null 2>&1; then
        if install_with_dnf; then
            install_success=true
        fi
    elif command -v yum >/dev/null 2>&1; then
        if install_with_yum; then
            install_success=true
        fi
    elif command -v brew >/dev/null 2>&1; then
        if install_with_brew; then
            install_success=true
        fi
    else
        warn "未检测到支持的包管理器（apt/yum/dnf/brew），跳过自动安装 Git LFS"
    fi

    # 二次检查安装是否成功
    if command -v git-lfs >/dev/null 2>&1; then
        info "Git LFS 安装成功"
        GIT_LFS_INSTALLED=true
    else
        warn "Git LFS 自动安装失败，脚本将继续执行，但后续依赖 Git LFS 的操作可能出错"
    fi
fi

# 初始化 Git LFS
if $GIT_LFS_INSTALLED; then
    if ! git lfs install --local; then
        warn "Git LFS 初始化失败，但脚本将继续执行"
    else
        info "Git LFS 已在 lerobot 仓库内初始化完成"
    fi
else
    warn "Git LFS 未安装，跳过初始化步骤"
fi

# ===================== 步骤 5：安装 Pi 与 Libero 可选依赖... =====================
progress "安装 Pi05 相关可选依赖..."

# 定义 CMake 3.x 安装路径
CMAKE3_INSTALL_DIR="${HOME}/.local/cmake3"
CMAKE3_BIN="${CMAKE3_INSTALL_DIR}/bin/cmake"
# 定义 CMake 源码包缓存路径和目标版本
CMAKE3_VERSION="3.28.3"
CMAKE3_TAR_FILE="${HOME}/cmake3_temp/cmake-${CMAKE3_VERSION}.tar.gz"
CMAKE3_OFFICIAL_URL="https://cmake.org/files/v3.28/cmake-${CMAKE3_VERSION}.tar.gz"

# 检查并自动安装/降级 CMake 3.x
install_cmake3() {
    info "开始自动安装用户级 CMake ${CMAKE3_VERSION}..."
    
    # 1. 准备依赖
    if ! command -v gcc &>/dev/null || ! command -v g++ &>/dev/null; then
        error "未检测到 gcc/g++，无法编译 CMake！请联系管理员安装 gcc/g++ 后重试"
    fi

    # 2. 新建临时目录，清理旧缓存
    if [[ -d ~/cmake3_temp ]]; then
        rm -rf ~/cmake3_temp
    fi
    mkdir -p ~/cmake3_temp && cd ~/cmake3_temp || error "创建临时目录失败"

    # 3. 下载源码
    info "检查 CMake ${CMAKE3_VERSION} 源码包缓存..."
    if [[ -f "${CMAKE3_TAR_FILE}" ]]; then
        info "发现已下载的 CMake 源码包：${CMAKE3_TAR_FILE}，跳过下载"
    else
        info "下载 CMake ${CMAKE3_VERSION} 源码..."
        # 带进度的下载函数
        download_cmake() {
            local url="$1"
            # 优先用 wget
            if command -v wget >/dev/null 2>&1; then
                wget --progress=bar:force -c -O cmake-${CMAKE3_VERSION}.tar.gz "${url}"
                return $?
            # 备用 curl
            elif command -v curl >/dev/null 2>&1; then
                curl -# -L -C - -o cmake-${CMAKE3_VERSION}.tar.gz "${url}"
                return $?
            else
                error "未检测到 wget/curl，无法下载文件！"
                return 1
            fi
        }

        # 仅从官方地址下载
        if ! download_cmake "${CMAKE3_OFFICIAL_URL}"; then
            error "CMake 源码下载失败！请检查网络或手动下载后放到 ~/cmake3_temp/（官方地址：${CMAKE3_OFFICIAL_URL}）"
        fi
    fi

    # 4. 解压
    info "解压 CMake 源码包..."
    tar -zxf cmake-${CMAKE3_VERSION}.tar.gz && cd cmake-${CMAKE3_VERSION} || error "解压失败"

    # 5. 配置编译
    info "配置 CMake 编译参数..."
    ./bootstrap --prefix="${CMAKE3_INSTALL_DIR}" \
                --no-system-curl \
                --no-system-zlib \
                --no-system-bzip2 || error "CMake 配置失败"

    # 6. 编译
    info "编译 CMake ${CMAKE3_VERSION}..."
    make -j$(nproc) || error "CMake 编译失败"

    # 7. 安装到用户目录
    info "安装 CMake ${CMAKE3_VERSION} 到 ${CMAKE3_INSTALL_DIR}..."
    make install || error "CMake 安装失败"

    # 8. 清理临时文件
    cd ~ && rm -rf ~/cmake3_temp/cmake-${CMAKE3_VERSION}

    # 9. 验证安装
    if ! "${CMAKE3_BIN}" --version >/dev/null 2>&1; then
        error "CMake ${CMAKE3_VERSION} 安装后验证失败！"
    fi

    info "CMake ${CMAKE3_VERSION} 安装成功！"
}

# 主逻辑：检查 CMake 版本，不兼容则自动安装 3.x
CMAKE_VERSION=""
SKIP_CMAKE_INSTALL=false

# 先检查用户目录是否已安装目标版本 CMake
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

# 如果用户目录没有目标版本，检查系统 CMake
if [[ "${SKIP_CMAKE_INSTALL}" == false ]]; then
    if command -v cmake >/dev/null 2>&1; then
        CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        MAJOR_VERSION=$(echo "${CMAKE_VERSION}" | cut -d. -f1)
    fi

    # 情况1：未安装 CMake 或版本 >=4 → 自动安装 3.x
    if [[ -z "${CMAKE_VERSION}" || "${MAJOR_VERSION}" -ge 4 ]]; then
        if [[ -n "${CMAKE_VERSION}" ]]; then
            warn "检测到 CMake ${CMAKE_VERSION} 不兼容，自动安装 ${CMAKE3_VERSION}..."
        else
            warn "未检测到 CMake，自动安装 ${CMAKE3_VERSION}..."
        fi
        install_cmake3
        SKIP_CMAKE_INSTALL=true
    # 情况2：版本是 3.x → 直接使用
    else
        info "CMake 版本 ${CMAKE_VERSION} 兼容，无需安装"
        CMAKE3_BIN="cmake"
        SKIP_CMAKE_INSTALL=true
    fi
fi

# 强制将用户级 CMake 加入 PATH
export PATH="${CMAKE3_INSTALL_DIR}/bin:${PATH}"
# 验证当前 PATH 中的 CMake 版本
CURRENT_CMAKE=$(which cmake)
CURRENT_CMAKE_VERSION=$("${CURRENT_CMAKE}" --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
info "当前生效的 CMake 路径：${CURRENT_CMAKE}，版本：${CURRENT_CMAKE_VERSION}"

# 最终验证 CMake 版本
FINAL_MAJOR_VERSION=$(echo "${CURRENT_CMAKE_VERSION}" | cut -d. -f1)
if [[ "${FINAL_MAJOR_VERSION}" -ne 3 ]]; then
    error "CMake 版本验证失败！当前版本：${CURRENT_CMAKE_VERSION}，请手动检查"
fi

progress "CMake 版本 ${CURRENT_CMAKE_VERSION} 兼容，继续安装 Pi05 依赖..."

# 关键修复：安装依赖时显式指定 CMake 路径，确保 subprocess 调用正确的 CMake
CMAKE_PREFIX_PATH="${CMAKE3_INSTALL_DIR}" \
CMAKE="${CMAKE3_BIN}" \
pip install -e ".[pi]" || error "Pi05 依赖安装失败"

progress "安装 Libero 仿真环境依赖（libero）..."

# 关键修复：安装 Libero 依赖时同样传递 CMake 环境变量
CMAKE_PREFIX_PATH="${CMAKE3_INSTALL_DIR}" \
CMAKE="${CMAKE3_BIN}" \
pip install -e ".[libero]" || error "Libero 依赖安装失败"

# ===================== 步骤 6：aarch64 架构下手动编译 torchcodec（可选） =====================
if [[ "$(uname -m)" == "aarch64" ]]; then
    progress "检测到 aarch64 架构，尝试安装 torchcodec..."
    
    # 检查是否已安装 torchcodec
    if python -c "import torchcodec" &>/dev/null; then
        info "torchcodec 已安装，跳过编译"
    else
        info "正在安装 pybind11"
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pybind11
        info "正在克隆 torchcodec 仓库..."
        git clone https://github.com/meta-pytorch/torchcodec.git || error "torchcodec 仓库克隆失败"
        cd torchcodec

        info "正在编译并安装 torchcodec..."
        # 使用 --no-build-isolation 避免重复下载 PyTorch；-v 用于调试
        if pip install -e . --no-build-isolation -v; then
            success "torchcodec 安装成功"
        else
            warn "torchcodec 安装失败。若需视频解码功能，请手动排查编译依赖（如 ffmpeg、cmake 等）"
            # 不 exit，因为 torchcodec 仅为可选优化组件
        fi
        cd "${LEROBOT_DIR}"  # 返回主目录
    fi
else
    info "当前架构非 aarch64（$(uname -m)），跳过 torchcodec 编译"
fi

success "所有依赖安装完成"

# ====================== 安装torch_npu ======================
if [[ "${SKIP_TORCH_NPU}" == false ]]; then
    if [[ "$(uname -m)" == "aarch64" ]]; then
        progress "安装 torch_npu（aarch64 架构）..."
        
        # 检查 PyTorch 版本是否匹配
        PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
        if [[ "${PYTORCH_VERSION}" != "2.6.0"* ]]; then
            warn "检测到 PyTorch 版本 ${PYTORCH_VERSION}，建议升级至 2.6.0 以匹配 torch_npu"
        fi

        if python -c "import torch_npu" &>/dev/null; then
            info "torch_npu 已安装，跳过安装"
        else
            # 断点续传下载 whl 包
            wget -c https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_28_aarch64.whl || error "torch_npu 包下载失败"
            pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_28_aarch64.whl || error "torch_npu 安装失败"
            success "torch_npu 安装成功"
        fi
    else
        warn "当前架构为 $(uname -m)，不支持安装 torch_npu（仅 aarch64），已自动跳过"
    fi
else
    info "已跳过 torch_npu 安装"
fi

# ===================== 完成提示 =====================
cd "${ROOT_DIR}"
success "========================================"
success "Pi05 项目环境部署成功！"
success "请执行以下命令激活环境："
success "  conda activate lerobot"
success "注意：脚本内激活的环境仅对脚本进程有效，你需要手动激活！"
success "========================================"