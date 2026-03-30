#!/bin/bash
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
COSMOS_INIT_FILE="${COSMOS_ROOT}/cosmos_predict2/__init__.py"
COSMOS_OSS_INIT_FILE="${COSMOS_ROOT}/packages/cosmos-oss/cosmos_oss/__init__.py"
QWEN2_5_VL_FILE="${COSMOS_ROOT}/cosmos_predict2/_src/reason1/networks/qwen2_5_vl.py"
MINIMAL_V4_DIT_FILE="${COSMOS_ROOT}/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py"
WAN2PT1_FILE="${COSMOS_ROOT}/cosmos_predict2/_src/predict2/networks/wan2pt1.py"
INFERENCE_FILE="${COSMOS_ROOT}/examples/inference.py"
DISTRIBUTED_FILE="${COSMOS_ROOT}/cosmos_predict2/_src/imaginaire/utils/distributed.py"
FUSED_ADAM_FILES=(
    "${COSMOS_ROOT}/cosmos_predict2/_src/imaginaire/utils/fused_adam.py"
    "${COSMOS_ROOT}/cosmos_predict2/_src/predict2/utils/fused_adam_dtensor.py"
    "${COSMOS_ROOT}/cosmos_predict2/_src/reason1/utils/fused_adam.py"
)

# 要生成的monkey_patch代码补丁文件路径
MONKEY_PATCH_FILE="${COSMOS_ROOT}/monkey_patch_npu_cosmos_predict.py"


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


# ====================== 第二步：禁用CUDA检查 =======================
if [ -f "${COSMOS_INIT_FILE}" ]; then
    cp "${COSMOS_INIT_FILE}" "${COSMOS_INIT_FILE}.bak"
    sed -i '/^_check_cuda_extra()/d' "${COSMOS_INIT_FILE}"
    echo -e "\033[32m[INFO] Fixed ${COSMOS_INIT_FILE} syntax error successfully\033[0m"
fi

if [ -f "${COSMOS_OSS_INIT_FILE}" ]; then
    cp "${COSMOS_OSS_INIT_FILE}" "${COSMOS_OSS_INIT_FILE}.bak"
    sed -i '/^_check_cuda_extra()/d' "${COSMOS_OSS_INIT_FILE}"
    echo -e "\033[32m[INFO] Fixed ${COSMOS_OSS_INIT_FILE} syntax error successfully\033[0m"
fi


# ====================== 第三步：修复qwen2_5_vl.py =======================
if [ -f "${QWEN2_5_VL_FILE}" ]; then
    cp "${QWEN2_5_VL_FILE}" "${QWEN2_5_VL_FILE}.bak"
    sed -i '/^assert is_flash_attn_2_available()/d' "${QWEN2_5_VL_FILE}"
    echo -e "\033[32m[INFO] Fixed ${QWEN2_5_VL_FILE} syntax error successfully\033[0m"
fi


# ============ 第四步：清理transformer_engine引用 及 多余报错类 ===========
for file in "${WAN2PT1_FILE}" "${FUSED_ADAM_FILES[@]}"; do
    if [ -f "${file}" ]; then
        cp "${file}" "${file}.bak"
        sed -i '/from transformer_engine.pytorch.attention/d' "${file}"
        sed -i '/import transformer_engine/d' "${file}"
        sed -i '/class SelfAttnOp(DotProductAttention)/,/return super().forward(q_B_L_H_D,/d' "${file}"
    fi
    echo -e "\033[32m[INFO] Fixed import transformer_engine and class SelfAttnOp syntax error successfully\033[0m"
done

if [ -f "${DISTRIBUTED_FILE}" ]; then
    cp "${DISTRIBUTED_FILE}" "${DISTRIBUTED_FILE}.bak"
    sed -i '/pynvml.nvmlInit()/d' "${DISTRIBUTED_FILE}"
    sed -i '/_libcudart = ctypes.CDLL("libcudart.so")/d' "${DISTRIBUTED_FILE}"
    sed -i '/p_value = ctypes.cast/d' "${DISTRIBUTED_FILE}"
    sed -i '/_libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))/d' "${DISTRIBUTED_FILE}"
    sed -i '/_libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))/d' "${DISTRIBUTED_FILE}"
    echo -e "\033[32m[INFO] Fixed distributed syntax error successfully\033[0m"
fi


