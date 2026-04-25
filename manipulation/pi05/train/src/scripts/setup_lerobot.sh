#!/bin/bash
# Copyright (c) 2026 Institute of Software, Chinese Academy of Sciences (ISCAS). All rights reserved.
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

# shellcheck disable=SC2034
sync_lerobot_repo() {
    progress "拉取 LeRobot 代码仓库..."
    if [[ -d "${LEROBOT_DIR}" ]]; then
        if [[ -d "${LEROBOT_DIR}/.git" ]]; then
            cd "${LEROBOT_DIR}"
            if [[ -n "$(git status --porcelain)" ]]; then
                if [[ "${FORCE_SYNC}" == true ]]; then
                    warn "检测到 ${LEROBOT_DIR} 存在未提交修改，按 --force 要求执行覆盖"
                    git reset --hard HEAD || error "LeRobot 仓库 reset 失败"
                    git clean -fd || error "LeRobot 仓库 clean 失败"
                else
                    error "目标目录 ${LEROBOT_DIR} 存在未提交修改。为避免覆盖当前工作，请先提交/备份，或通过 --lerobot-dir 指定新的同步目录；若确认要覆盖，请追加 --force"
                fi
            fi
            info "lerobot 目录已存在，更新代码"
            git fetch --all --tags || error "LeRobot 仓库 fetch 失败"
        else
            if [[ "${FORCE_SYNC}" == true ]]; then
                warn "目标目录 ${LEROBOT_DIR} 已存在但不是 git 仓库，按 --force 要求删除后重建"
                rm -rf "${LEROBOT_DIR}"
                git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}" || error "LeRobot 仓库克隆失败"
                cd "${LEROBOT_DIR}"
            else
                error "目标目录 ${LEROBOT_DIR} 已存在，但不是 git 仓库。请清理该目录或通过 --lerobot-dir 指定新的目录；若确认要覆盖，请追加 --force"
            fi
        fi
    else
        git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}" || error "LeRobot 仓库克隆失败"
        cd "${LEROBOT_DIR}"
    fi

    git checkout --detach "${LEROBOT_COMMIT}" || error "无法切换到指定提交：${LEROBOT_COMMIT}（请检查 commit/tag 是否存在）"
    git reset --hard "${LEROBOT_COMMIT}" || error "无法重置到指定提交：${LEROBOT_COMMIT}"
    git clean -fd || error "LeRobot 仓库 clean 失败"
    success "LeRobot 代码仓库已锁定至指定版本"
}

copy_pi05_files() {
    progress "复制 Pi05 实现文件至 LeRobot 项目中..."

    mkdir -p "${LEROBOT_DIR}/src/lerobot/policies/pi05/"
    mkdir -p "${LEROBOT_DIR}/src/lerobot/scripts/"
    mkdir -p "${LEROBOT_DIR}/src/lerobot/utils/"
    mkdir -p "${LEROBOT_DIR}/src/lerobot/configs/"

    cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/modeling_pi05.py" "${LEROBOT_DIR}/src/lerobot/policies/pi05/" || error "复制 modeling_pi05.py 失败"
    cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_train.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 lerobot_train.py 失败"
    cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_eval.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 lerobot_eval.py 失败"
    cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/lerobot_train_profiling.py" "${LEROBOT_DIR}/src/lerobot/scripts/" || error "复制 run_train_profiling.py 失败"
    cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/utils.py" "${LEROBOT_DIR}/src/lerobot/utils/" || error "复制 utils.py 失败"

    if compgen -G "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/configs/*.yaml" > /dev/null; then
        cp -f "${CANN_RECIPES_DIR}/manipulation/pi05/train/src/configs/"*.yaml "${LEROBOT_DIR}/src/lerobot/configs/" || error "复制配置文件失败"
    else
        warn "Pi05 配置文件目录为空，跳过配置文件复制"
    fi

    success "Pi05 相关文件复制完成"
}

prepare_conda_env_and_lerobot_base() {
    progress "创建 Conda 虚拟环境 'lerobot' (Python ${PYTHON_VERSION})..."

    if conda info --envs | grep -q "lerobot"; then
        info "lerobot 环境已存在，跳过创建"
    else
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
        conda config --set show_channel_urls yes

        conda create -y -n lerobot python=${PYTHON_VERSION} || error "Conda 环境创建失败"
    fi

    eval "$(conda shell.bash hook)"
    conda activate lerobot || error "无法激活 Conda 环境 'lerobot'"
    ensure_python_version "${PYTHON_VERSION}"

    cd "${LEROBOT_DIR}"

    progress "安装固定版本 PyTorch 栈..."
    if python_pkg_exact_version "torch" "${TORCH_VERSION}" && python_pkg_exact_version "torchvision" "${TORCHVISION_VERSION}"; then
        info "torch ${TORCH_VERSION} 与 torchvision ${TORCHVISION_VERSION} 已安装，跳过安装"
    else
        pip install --upgrade "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" || error "PyTorch 栈安装失败"
    fi

    progress "安装 LeRobot 基础依赖..."
    INSTALLED_LEROBOT_VERSION=$(python_pkg_version "lerobot" 2>/dev/null || true)
    INSTALLED_LEROBOT_EDITABLE=$(python_pkg_editable_location "lerobot" 2>/dev/null || true)
    if [[ "${INSTALLED_LEROBOT_VERSION}" == "0.4.4" && "${INSTALLED_LEROBOT_EDITABLE}" == "${LEROBOT_DIR}" ]]; then
        info "LeRobot 0.4.4 已以 editable 方式安装到 ${LEROBOT_DIR}，跳过基础安装"
    else
        pinned_lerobot_pip_install -e . || error "LeRobot 基础安装失败"
    fi
}

setup_git_lfs() {
    progress "配置 Git LFS..."

    GIT_LFS_INSTALLED=false

    if command -v git-lfs >/dev/null 2>&1; then
        info "Git LFS 已安装"
        GIT_LFS_INSTALLED=true
    else
        info "Git LFS 未安装，正在尝试自动安装..."

        run_with_sudo() {
            local cmd="$*"
            if command -v sudo >/dev/null 2>&1; then
                sudo $cmd
                return $?
            else
                warn "需要 root 权限执行命令：$cmd，但未检测到 sudo，请手动安装 Git LFS"
                return 1
            fi
        }

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
            if ! brew install git-lfs >/dev/null 2>&1; then
                warn "brew 安装 Git LFS 失败"
                return 1
            fi
            return 0
        }

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

        if command -v git-lfs >/dev/null 2>&1; then
            info "Git LFS 安装成功"
            GIT_LFS_INSTALLED=true
        else
            warn "Git LFS 自动安装失败，脚本将继续执行，但后续依赖 Git LFS 的操作可能出错"
        fi
    fi

    if $GIT_LFS_INSTALLED; then
        if ! git lfs install --local; then
            warn "Git LFS 初始化失败，但脚本将继续执行"
        else
            info "Git LFS 已在 lerobot 仓库内初始化完成"
        fi
    else
        warn "Git LFS 未安装，跳过初始化步骤"
    fi
}
