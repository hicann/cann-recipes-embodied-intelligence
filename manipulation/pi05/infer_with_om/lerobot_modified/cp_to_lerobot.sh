#!/bin/bash
# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 Shanghai Jiao Tong University
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -euo pipefail

find_root_with_src() {
    local candidate="$1"

    if [ -z "$candidate" ]; then
        return 1
    fi

    # 绝对路径化（路径不存在时保持原值）
    if [ -e "$candidate" ]; then
        candidate="$(readlink -f "$candidate")"
    fi

    while [ "$candidate" != "/" ]; do
        if [ -d "$candidate/src/lerobot" ]; then
            echo "$candidate"
            return 0
        fi
        candidate="$(dirname "$candidate")"
    done

    return 1
}

detect_lerobot_root() {
    local pip_show_output editable_root location_root import_path resolved_root

    # 优先使用 pip show 的可编辑安装路径
    pip_show_output="$(pip show lerobot 2>/dev/null || true)"
    editable_root="$(printf '%s\n' "$pip_show_output" | awk -F': ' '/^Editable project location:/ {print $2; exit}')"
    if resolved_root="$(find_root_with_src "$editable_root")"; then
        echo "$resolved_root"
        return 0
    fi

    # 其次尝试 pip show 的 Location 字段
    location_root="$(printf '%s\n' "$pip_show_output" | awk -F': ' '/^Location:/ {print $2; exit}')"
    if resolved_root="$(find_root_with_src "$location_root")"; then
        echo "$resolved_root"
        return 0
    fi
    if resolved_root="$(find_root_with_src "$location_root/lerobot")"; then
        echo "$resolved_root"
        return 0
    fi

    # 最后回退到 import lerobot 的路径
    import_path="$(python -c "import lerobot; print(list(lerobot.__path__)[0])" 2>/dev/null || true)"
    if resolved_root="$(find_root_with_src "$import_path")"; then
        echo "$resolved_root"
        return 0
    fi

    return 1
}

SCRIPT_NAME="$(basename "$0")"

SRC_FILE="${1:-}"
TARGET_SUBDIR_INPUT="${2:-}"
LEROBOT_ROOT_INPUT="${3:-}"

if [ -z "$SRC_FILE" ] || [ -z "$TARGET_SUBDIR_INPUT" ]; then
    echo "❌ 错误: 参数缺失。"
    echo "👉 用法: bash $SCRIPT_NAME <源文件路径> <目标子目录> [lerobot根目录(可选)]"
    echo "💡 示例: bash $SCRIPT_NAME export_pi05.py policies/pi05"
    exit 1
fi

# 自动探测 lerobot 根目录；若用户传了第3个参数则优先使用
if [ -n "$LEROBOT_ROOT_INPUT" ]; then
    LEROBOT_ROOT="$LEROBOT_ROOT_INPUT"
else
    LEROBOT_ROOT="$(detect_lerobot_root || true)"
fi

if [ -z "$LEROBOT_ROOT" ]; then
    echo "❌ 错误: 自动探测 lerobot 根目录失败。"
    echo "👉 请追加第3个参数手动指定，例如:"
    echo "   bash $SCRIPT_NAME export_pi05.py policies/pi05 /home/HwHiAiUser/lerobot"
    exit 1
fi

if [[ "$TARGET_SUBDIR_INPUT" == src/lerobot/* ]]; then
    TASK_SUBDIR="$TARGET_SUBDIR_INPUT"
else
    TASK_SUBDIR="src/lerobot/$TARGET_SUBDIR_INPUT"
fi

# 1. 检查参数输入是否完整
if [ -z "$SRC_FILE" ] || [ -z "$LEROBOT_ROOT" ] || [ -z "$TASK_SUBDIR" ]; then
    echo "❌ 错误: 参数缺失。"
    echo "👉 用法: bash $SCRIPT_NAME <源文件路径> <目标子目录> [lerobot根目录(可选)]"
    echo "💡 示例: bash $SCRIPT_NAME export_pi05.py policies/pi05"
    exit 1
fi

# 2. 检查源文件和根目录是否存在
if [ ! -f "$SRC_FILE" ]; then
    echo "❌ 错误: 找不到你要复制的 Python 文件 '$SRC_FILE'"
    exit 1
fi

if [ ! -d "$LEROBOT_ROOT" ]; then
    echo "❌ 错误: 找不到 lerobot 仓库根目录 '$LEROBOT_ROOT'"
    exit 1
fi

if [ ! -d "$LEROBOT_ROOT/src/lerobot" ]; then
    echo "❌ 错误: '$LEROBOT_ROOT' 看起来不是 lerobot 根目录（缺少 src/lerobot）。"
    exit 1
fi

# 3. 解析绝对路径并构建最终的目标路径
SRC_ABS=$(readlink -f "$SRC_FILE")
# 将根目录和子目录拼接起来
TARGET_DIR="${LEROBOT_ROOT}/${TASK_SUBDIR}"
echo "📁 目标路径已解析为: $TARGET_DIR"
# 4. 如果目标文件夹不存在，则自动创建
if [ ! -d "$TARGET_DIR" ]; then
    echo "📁 目标子目录不存在，正在为你自动创建: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# 5. 执行复制操作
FILENAME=$(basename "$SRC_ABS")
echo "📄 正在将 $FILENAME 复制到 $TARGET_DIR ..."

# 使用 cp 进行复制，保留原文件
cp "$SRC_ABS" "$TARGET_DIR/"

echo "✅ 复制成功！原文件已保留。"
echo "📂 新文件现在位于: $TARGET_DIR/$FILENAME"