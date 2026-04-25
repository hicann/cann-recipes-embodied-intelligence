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

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "一键部署 Pi05 项目环境"
    echo ""
    echo "Options:"
    echo "  -h, --help    显示此帮助信息并退出"
    echo "  --skip-torch-npu  跳过 torch_npu 安装"
    echo "  --sync-only  仅同步 LeRobot 代码与 Pi05 相关文件，跳过环境安装"
    echo "  --lerobot-dir <path>  指定 LeRobot 目标目录"
    echo "  --lerobot-ref <ref>   指定要同步的 LeRobot git ref/commit/branch"
    echo "  --force  强制覆盖目标 LeRobot 目录中的未提交修改"
    echo ""
    echo "Environment Variables:"
    echo "  CMAKE3_MIRROR_URL  CMake 源码下载备用地址（默认清华镜像）"
    echo ""
    echo "前置要求："
    echo "  1. 已安装 conda 并配置好镜像源"
    echo "  2. 网络可访问 GitHub/GitCode/Ascend 仓库"
}

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

pinned_lerobot_pip_install() {
    pip install -c <(
        cat <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
EOF
    ) "$@"
}

check_path() {
    local path="$1"
    local type="$2"
    local msg="$3"

    if [[ "$type" == "file" && ! -f "$path" ]]; then
        error "${msg}：文件不存在 ${path}"
    elif [[ "$type" == "dir" && ! -d "$path" ]]; then
        error "${msg}：目录不存在 ${path}"
    fi
}

python_pkg_version() {
    python - "$1" <<'EOF'
import importlib.metadata as md
import sys
try:
    print(md.version(sys.argv[1]))
except Exception:
    sys.exit(1)
EOF
}

python_pkg_installed() {
    python_pkg_version "$1" >/dev/null 2>&1
}

python_pkg_exact_version() {
    local package="$1"
    local expected="$2"
    local version
    version=$(python_pkg_version "$package" 2>/dev/null || true)
    [[ -n "$version" && "$version" == "$expected" ]]
}

python_pkg_satisfies() {
    python - "$1" "$2" <<'EOF'
import importlib.metadata as md
import sys
from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:
    version = md.version(sys.argv[1])
except Exception:
    raise SystemExit(1)

specifier = SpecifierSet(sys.argv[2])
raise SystemExit(0 if Version(version) in specifier else 1)
EOF
}

python_pkg_editable_location() {
    python - "$1" <<'EOF'
import importlib.metadata as md
import json
import sys
try:
    dist = md.distribution(sys.argv[1])
except Exception:
    sys.exit(1)
text = dist.read_text('direct_url.json')
if not text:
    sys.exit(1)
url = json.loads(text).get('url', '')
if url.startswith('file://'):
    print(url[7:])
EOF
}

ensure_python_version() {
    local expected="$1"
    local current
    current=$(python - <<'EOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)
    if [[ "$current" != "$expected" ]]; then
        error "当前激活环境的 Python 版本为 ${current}，与要求的 ${expected} 不一致"
    fi
}

pi_transformers_ready() {
    python - <<'EOF' >/dev/null 2>&1
try:
    from transformers.models.siglip import check
    import sys
    sys.exit(0 if check.check_whether_transformers_replace_is_installed_correctly() else 1)
except Exception:
    raise SystemExit(1)
EOF
}

parse_args() {
    SKIP_TORCH_NPU=false
    SYNC_ONLY=false
    FORCE_SYNC=false
    CUSTOM_LEROBOT_DIR=""
    CUSTOM_LEROBOT_REF=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            --skip-torch-npu)
                SKIP_TORCH_NPU=true
                shift
                ;;
            --sync-only)
                SYNC_ONLY=true
                shift
                ;;
            --lerobot-dir)
                [[ $# -lt 2 ]] && error "--lerobot-dir 需要传入路径"
                CUSTOM_LEROBOT_DIR="$2"
                shift 2
                ;;
            --lerobot-ref)
                [[ $# -lt 2 ]] && error "--lerobot-ref 需要传入 git ref"
                CUSTOM_LEROBOT_REF="$2"
                shift 2
                ;;
            --force)
                FORCE_SYNC=true
                shift
                ;;
            *)
                error "未知参数：$1，请使用 -h 查看帮助"
                ;;
        esac
    done
}

load_modules() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "${SCRIPT_DIR}/setup_lerobot.sh"
    source "${SCRIPT_DIR}/setup_cmake.sh"
    source "${SCRIPT_DIR}/setup_deps.sh"
}

main() {
    LEROBOT_TESTED_VERSION="v0.4.4"
    LEROBOT_TESTED_COMMIT="8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
    PYTHON_VERSION="3.10"
    TORCH_VERSION="2.8.0"
    TORCHVISION_VERSION="0.23.0"
    TORCH_NPU_VERSION="2.8.0.post2"
    TORCHCODEC_VERSION="0.7.0"
    TORCHCODEC_GIT_REF="v${TORCHCODEC_VERSION}"
    TORCH_NPU_RUNTIME_DEPS=(pyyaml attrs psutil decorator cloudpickle scipy tornado ml-dtypes)
    TORCH_NPU_RUNTIME_OPTIONAL_DEPS=(absl-py)

    parse_args "$@"

    ROOT_DIR="$(pwd)"
    CANN_RECIPES_DIR="${ROOT_DIR}/cann-recipes-embodied-intelligence"
    if [[ -n "${CUSTOM_LEROBOT_DIR}" ]]; then
        LEROBOT_DIR="${CUSTOM_LEROBOT_DIR}"
    else
        LEROBOT_DIR="${ROOT_DIR}/lerobot"
    fi

    LEROBOT_COMMIT="${CUSTOM_LEROBOT_REF:-${LEROBOT_TESTED_COMMIT}}"

    load_modules

    info "开始执行 Pi05 代码仓、模型、数据集一键下载脚本"
    info "工作目录：${ROOT_DIR}"
    info "LeRobot 目标目录：${LEROBOT_DIR}"
    info "LeRobot 目标版本：${LEROBOT_TESTED_VERSION}"
    info "LeRobot 目标提交：${LEROBOT_COMMIT}"
    info "Python 版本：${PYTHON_VERSION}"
    info "PyTorch 版本：${TORCH_VERSION}"
    info "TorchVision 版本：${TORCHVISION_VERSION}"
    info "torch_npu 版本：${TORCH_NPU_VERSION}"
    info "torchcodec 版本：${TORCHCODEC_VERSION}"
    info "CMake 官方下载地址：${CMAKE3_OFFICIAL_URL}"
    info "CMake 国内镜像备用地址：${CMAKE3_MIRROR_URL}"
    if [[ "${SYNC_ONLY}" == true ]]; then
        info "当前模式：仅同步源码，不安装环境"
    fi
    if [[ "${FORCE_SYNC}" == true ]]; then
        warn "当前模式：强制覆盖 LeRobot 目标目录中的本地修改"
    fi

    check_command "git"
    check_command "cp"
    check_command "conda"
    check_command "pip"

    if [[ ! -d "${CANN_RECIPES_DIR}" ]]; then
        info "未检测到 cann-recipes-embodied-intelligence 目录，开始自动克隆..."
        git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git "${CANN_RECIPES_DIR}" || error "CANN Recipes 仓库克隆失败"
    else
        info "检测到 cann-recipes-embodied-intelligence 目录已存在"
    fi

    check_path "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/modeling_pi05.py" "file" "Pi05 核心模型文件缺失"
    check_path "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_train.py" "file" "Pi05 训练脚本缺失"

    if [[ -d "${CANN_RECIPES_DIR}" && -d "${LEROBOT_DIR}" ]]; then
        info "cann-recipes 与 lerobot 目录均已存在，将进行增量更新"
    elif [[ -d "${CANN_RECIPES_DIR}" || -d "${LEROBOT_DIR}" ]]; then
        info "部分目录已存在，将补全缺失内容"
    else
        info "未检测到目标目录，将执行全新克隆"
    fi

    sync_lerobot_repo
    copy_pi05_files

    if [[ "${SYNC_ONLY}" == true ]]; then
        success "同步完成：已更新 LeRobot 代码并复制 Pi05 相关文件"
        exit 0
    fi

    prepare_conda_env_and_lerobot_base
    setup_git_lfs
    setup_cmake3_if_needed
    install_pi05_optional_deps
    install_torchcodec_for_aarch64
    success "所有依赖安装完成"

    install_torch_npu_if_enabled

    cd "${ROOT_DIR}"
    success "========================================"
    success "Pi05 项目环境部署成功！"
    success "请执行以下命令激活环境："
    success "  conda activate lerobot"
    success "注意：脚本内激活的环境仅对脚本进程有效，你需要手动激活！"
    success "========================================"
}

main "$@"
