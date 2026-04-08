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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_LOCAL_FILE="$SCRIPT_DIR/symbolic_opset14.py"
LOCAL_MODEL_FILE="${1:-$DEFAULT_LOCAL_FILE}"
PATCH_FILE="$SCRIPT_DIR/symbolic_opset14.patch"

if [ ! -f "$LOCAL_MODEL_FILE" ]; then
    echo "❌ 错误: 找不到本地文件 '$LOCAL_MODEL_FILE'"
    echo "👉 用法: bash apply_symbolic_opset14_patch.sh <你的本地 symbolic_opset14.py>"
    exit 1
fi

echo "🔍 正在当前 Python 环境中寻找 torch 库..."

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    fi
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "❌ 错误: 找不到可用的 Python 解释器。"
    exit 1
fi

SITE_PACKAGES_DIR=$("$PYTHON_BIN" -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))" 2>/dev/null)

if [ -z "$SITE_PACKAGES_DIR" ]; then
    echo "❌ 错误: 在当前环境中找不到 torch 库。请确认你已经激活了正确的 Conda 环境！"
    exit 1
fi

echo "📂 找到目标目录: $SITE_PACKAGES_DIR"

TARGET_FILE="$SITE_PACKAGES_DIR/torch/onnx/symbolic_opset14.py"

if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ 错误: 找不到目标文件 '$TARGET_FILE'"
    exit 1
fi

echo "🛠️  正在生成补丁: $PATCH_FILE"

# if diff -u \
#     --label "a/torch/onnx/symbolic_opset14.py" \
#     --label "b/torch/onnx/symbolic_opset14.py" \
#     "$TARGET_FILE" \
#     "$LOCAL_MODEL_FILE" \
#     > "$PATCH_FILE"; then
#     echo "ℹ️  本地文件和目标文件没有差异，无需应用补丁。"
#     exit 0
# else
#     DIFF_STATUS=$?
#     if [ "$DIFF_STATUS" -ne 1 ]; then
#         echo "❌ 错误: 生成补丁失败。"
#         exit "$DIFF_STATUS"
#     fi
# fi

cd "$SITE_PACKAGES_DIR" || exit 1

echo "🛠️  开始应用补丁..."

if patch -p1 < "$PATCH_FILE"; then
    echo "✅ 补丁应用成功！"
else
    echo "⚠️  补丁应用失败，或该文件已经被修改过。请查看上面的报错信息。"
    exit 1
fi