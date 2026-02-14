#!/bin/bash
# Adapted from
# https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e


# ======================== 要修改的代码文件路径定义 =========================
COSMOS_ROOT="./"
QWEN2_5_VL_FILE="${COSMOS_ROOT}/cosmos_transfer2/_src/reason1/networks/qwen2_5_vl.py"
MINIMAL_V4_DIT_FILE="${COSMOS_ROOT}/cosmos_transfer2/_src/predict2/networks/minimal_v4_dit.py"
MINIMAL_V4_LVG_DIT_CONTROL_FACE="${COSMOS_ROOT}/cosmos_transfer2/_src/transfer2/networks/minimal_v4_lvg_dit_control_vace.py"
WAN2PT1_FILE="${COSMOS_ROOT}/cosmos_transfer2/_src/predict2/networks/wan2pt1.py"
INFERENCE_FILE="${COSMOS_ROOT}/examples/inference.py"
FUSED_ADAM_FILES=(
    "${COSMOS_ROOT}/cosmos_transfer2/_src/imaginaire/utils/fused_adam.py"
    "${COSMOS_ROOT}/cosmos_transfer2/_src/predict2/utils/fused_adam_dtensor.py"
    "${COSMOS_ROOT}/cosmos_transfer2/_src/reason1/utils/fused_adam.py"
)
UTILS_FILES=(
    "${COSMOS_ROOT}/cosmos_transfer2/_src/imaginaire/utils/graph.py"
)




# ====================== 第一步：修复minimal_v4_dit.py =======================
if [ -f "${MINIMAL_V4_DIT_FILE}" ]; then
    # 1. 备份原文件
    cp "${MINIMAL_V4_DIT_FILE}" "${MINIMAL_V4_DIT_FILE}.bak"

    # 2. 直接修改原文件，删除transformer_engine版本判断的4行代码块
    sed -i '/te.__version__.*2.8.0/,/apply_rotary_pos_emb/d' "${MINIMAL_V4_DIT_FILE}"
    sed -i '/^else:/{N;/transformer_engine/d;}' "${MINIMAL_V4_DIT_FILE}"

    # 3. 删除TE导入行
    sed -i '/^import transformer_engine as te/d' "${MINIMAL_V4_DIT_FILE}"
    sed -i '/^from transformer_engine.pytorch.attention/d' "${MINIMAL_V4_DIT_FILE}"

    # 4. 替换te.pytorch.RMSNorm为RMSNorm
    sed -i 's/te.pytorch.RMSNorm/RMSNorm/g' "${MINIMAL_V4_DIT_FILE}"

    echo -e "\033[32m[INFO] Fixed ${MINIMAL_V4_DIT_FILE} syntax error successfully\033[0m"
fi




# ====================== 第二步：修复qwen2_5_vl.py =======================
if [ -f "${QWEN2_5_VL_FILE}" ]; then
    cp "${QWEN2_5_VL_FILE}" "${QWEN2_5_VL_FILE}.bak"
    sed -i '/^assert is_flash_attn_2_available()/d' "${QWEN2_5_VL_FILE}"
    echo -e "\033[32m[INFO] Fixed ${QWEN2_5_VL_FILE} syntax error successfully\033[0m"
fi


# ============ 第三步：清理transformer_engine引用 及 多余报错类 ===========
for file in "${WAN2PT1_FILE}" "${FUSED_ADAM_FILES[@]}"; do
    if [ -f "${file}" ]; then
        cp "${file}" "${file}.bak"
        sed -i '/from transformer_engine.pytorch.attention/d' "${file}"
        sed -i '/import transformer_engine/d' "${file}"
        sed -i '/class SelfAttnOp(DotProductAttention)/,/return super().forward(q_B_L_H_D,/d' "${file}"
    fi
    echo -e "\033[32m[INFO] Fixed import transformer_engine and class SelfAttnOp syntax error successfully\033[0m"
done



# ============ 第四步：修改qwen2_5_vl.py，添加torch_npu导入 ==========
if [ -f "${QWEN2_5_VL_FILE}" ]; then
    # 检查是否已存在 import torch_npu，避免重复插入
    if ! grep -q "^import torch_npu" "${QWEN2_5_VL_FILE}"; then
        # 插入import torch_npu导入代码
        sed -i '/^from torch.nn import CrossEntropyLoss/a\import torch_npu' "${QWEN2_5_VL_FILE}"
        echo -e "\033[32m[INFO] Added torch_npu imports to ${QWEN2_5_VL_FILE} successfully\033[0m"
    else
        # 已有导入时的提示
        echo "[INFO] import torch_npu already exists in ${QWEN2_5_VL_FILE}, skipping insertion"
    fi
fi

# ====================== 第五步：修改minimal_v4_lvg_dit_control_face.py =======================
if [ -f "${MINIMAL_V4_LVG_DIT_CONTROL_FACE}" ]; then
    cp "${MINIMAL_V4_LVG_DIT_CONTROL_FACE}" "${MINIMAL_V4_LVG_DIT_CONTROL_FACE}.bak"
    sed -i '/from transformer_engine.pytorch.attention/d' "${MINIMAL_V4_LVG_DIT_CONTROL_FACE}"
    sed -i '/import transformer_engine/d' "${MINIMAL_V4_LVG_DIT_CONTROL_FACE}"
    echo -e "\033[32m[INFO] Fixed import transformer_engine syntax error successfully\033[0m"
fi

# ====================== 第六步：修改UTILS_FILES =====================
for file in "${UTILS_FILES[@]}"; do
    if [ -f "${file}" ]; then
        cp "${file}" "${file}.bak"
        sed -i '/from transformer_engine/d' "${file}"
        sed -i '/import transformer_engine/d' "${file}"
    fi
    echo -e "\033[32m[INFO] Fixed import transformer_engine syntax error successfully\033[0m"
done
