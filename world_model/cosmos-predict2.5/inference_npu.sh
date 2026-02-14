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


# =========== 第五步：修改inference.py，添加NPU导入 + 加载补丁 ============
if [ -f "${INFERENCE_FILE}" ]; then
    cp "${INFERENCE_FILE}" "${INFERENCE_FILE}.bak"
    
    # ---------- 第一步：检查是否已存在 monkey_patch_npu_cosmos_predict 导入，避免重复 ----------
    if ! grep -q "^import monkey_patch_npu_cosmos_predict" "${INFERENCE_FILE}"; then
        # 在 import pydantic 后插入 NPU 导入（用双引号包裹字符串，避免转义问题）
        sed -i '/^import tyro/a\try:\
    import monkey_patch_npu_cosmos_predict\
    monkey_patch_npu_cosmos_predict.apply_all_patches()\
    print("[INFO] Monkey patch applied successfully!")\
except Exception as e:\
    print(f"[WARNING] Failed to apply patch: {e}")' "${INFERENCE_FILE}"
        echo -e "[INFO] Added NPU imports and patch after 'import pydantic'"
    else  # else和fi对齐
        echo -e "[INFO] monkey_patch_npu_cosmos_predict import already exists, skipping insertion"
    fi

    echo -e "\033[32m[INFO] Updated ${INFERENCE_FILE} (no duplicate imports) successfully\033[0m"
fi


# ======================= 第六步：生成NPU补丁 =======================
rm -f ${MONKEY_PATCH_FILE}
cat > ${MONKEY_PATCH_FILE} << 'EOF'
#!/usr/bin/env python3
import sys
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import math
from einops import rearrange  # 提前导入，避免后续报错
from torch.distributed.device_mesh import DeviceMesh
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache


def setup_npu_device():
    if torch.npu.is_available():
        torch.set_default_device('npu')
        torch.npu.set_device(0)
        print("\033[32m[INFO] NPU device configured successfully (device 0)\033[0m")


# 内置RoPE实现
def apply_rotary_pos_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    radians = freqs.transpose(0, 1)  # [1,S2,1,D2]
    # 核心旋转计算
    cos = torch.cos(radians)  # [1,S2,1,128]
    sin = torch.sin(radians)  # [1,S2,1,128]
    # 用torch_npu融合算子进行快速使能
    res_rot = torch_npu.npu_rotary_mul(x, cos, sin)
    return res_rot


# 重载Attention类的compute_qkv方法（去掉多余参数）
def patched_compute_qkv(self, x, context=None, rope_emb=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = self.q_proj(x)
    context = x if context is None else context
    k = self.k_proj(context)
    v = self.v_proj(context)
    q, k, v = map(
        lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
        (q, k, v),
    )

    def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        original_dtype = q.dtype
        if self.is_selfattn and rope_emb is not None:
            if self.use_wan_fp32_strategy:
                q = q.to(torch.float32)
                k = k.to(torch.float32)
            # 仅传q/k和rope_emb，去掉tensor_format和fused参数
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)
            if self.use_wan_fp32_strategy:
                q = q.to(original_dtype)
                k = k.to(original_dtype)
        return q, k, v

    q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)
    return q, k, v


# ------------------------- 修复minimal_v4_dit -------------------------
def patch_minimal_v4_dit():
    import cosmos_predict2._src.predict2.networks.minimal_v4_dit as m


    m.apply_rotary_pos_emb = apply_rotary_pos_emb

    # 1. 重载Attention类的compute_qkv方法
    m.Attention.compute_qkv = patched_compute_qkv
    
    # npu_rms_norm融合算子版本RMSNorm
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))

        def reset_parameters(self):
            torch.nn.init.ones_(self.weight)

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = torch_npu.npu_rms_norm(
                x,
                self.weight.float(),
                self.eps
            )[0]
            return result

    # 仅对核心重载逻辑做异常捕获（try/except紧邻）
    try:
        # 2. 强制Attention后端为torch（避免TE依赖）
        original_attn_init = m.Attention.__init__
        def new_attn_init(self, query_dim, context_dim=None, n_heads=8, head_dim=64, dropout=0.0, qkv_format="bshd", backend="torch", use_wan_fp32_strategy=False):
            # 强制使用torch后端，忽略传入的backend参数
            original_attn_init(self, query_dim, context_dim, n_heads, head_dim, dropout, qkv_format, "torch", use_wan_fp32_strategy)
            # 替换norm为本地实现
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.v_norm = torch.nn.Identity()
        
        m.Attention.__init__ = new_attn_init
        
        # 3. 修复MiniTrainDIT中的t_embedding_norm
        original_mini_init = m.MiniTrainDIT.__init__
        def new_mini_init(self, *args, **kwargs):
            original_mini_init(self, *args, **kwargs)
            # 替换TE的RMSNorm为本地实现
            self.t_embedding_norm = RMSNorm(self.model_channels, eps=1e-6)
        
        m.MiniTrainDIT.__init__ = new_mini_init
        
        # 4. 修复I2VCrossAttention的k_img_norm
        original_i2v_init = m.I2VCrossAttention.__init__
        def new_i2v_init(self, *args, **kwargs):
            original_i2v_init(self, *args, **kwargs)
            self.k_img_norm = RMSNorm(self.head_dim, eps=1e-6)
        
        m.I2VCrossAttention.__init__ = new_i2v_init

        if not hasattr(m, 'RMSNorm'):
            m.RMSNorm = RMSNorm
        print("\033[32m[INFO] patch_minimal_v4_dit applied successfully!\033[0m")
    except Exception as e:
        print(f"\033[33m[WARNING] patch_minimal_v4_dit failed: {e}\033[0m")
        import traceback
        traceback.print_exc()


# ---------------------- 修复qwen2_5_vl的FlashAttention2 ----------------------
def patch_qwen2_5_vl():
    """适配qwen2_5_vl到NPU"""
    try:
        import cosmos_predict2._src.reason1.networks.qwen2_5_vl as qwen2_5_vl


        original_flashAttention2_init = qwen2_5_vl.Qwen2_5_VLFlashAttention2.__init__
        
        def new_original_flashAttention2_init(self, *args, **kwargs):
            original_flashAttention2_init(self, *args, **kwargs)
            self.target_device = torch.device("npu")
            # 直接在NPU上创建张量，无设备间拷贝，效率更高
            self.atten_mask_npu = torch.triu(torch.ones(2048, 2048, device=self.target_device), diagonal=1).bool()

        qwen2_5_vl.Qwen2_5_VLFlashAttention2.__init__ = new_original_flashAttention2_init

        # 强制关闭flash_attn检查（关键：避免走flash路径）
        qwen2_5_vl.is_flash_attn_2_available = lambda: False
        
        def new_npu_flash_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            cp_mesh: DeviceMesh | None = None,
        ):
            """
            Args:
                cp_mesh (DeviceMesh, optional): Device mesh over which context parallelism is done.
            """
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

            # Because the input can be padded, the absolute sequence length depends on the max position id.
            cos, sin = position_embeddings
            query_states, key_states = qwen2_5_vl.apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
            )

            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }  # Specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = qwen2_5_vl.repeat_kv(key_states, self.num_key_value_groups)
            value_states = qwen2_5_vl.repeat_kv(value_states, self.num_key_value_groups)

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in float16 just to be sure everything works as expected.
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                qwen2_5_vl.logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)


            # 确认输入布局：为BNSD
            query_bnsd = query_states.contiguous()
            key_bnsd = key_states.contiguous()
            value_bnsd = value_states.contiguous()

            scale = 1.0 / math.sqrt(self.head_dim)
            head_num = query_bnsd.shape[1]
            attn_output_bnsd = torch_npu.npu_fusion_attention(
                                query_bnsd, key_bnsd, value_bnsd, head_num, input_layout="BNSD", 
                                pse=None,
                                atten_mask=self.atten_mask_npu,
                                scale=scale,
                                pre_tockens=2147483647,
                                next_tockens=2147483647,
                                keep_prob=1,
                                sparse_mode=2)[0]

            # 转换输出布局：BNSD 转为 BSND（恢复原格式）
            attn_output = attn_output_bnsd.permute(0, 2, 1, 3)

            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value
        
        # 替换forward方法
        if hasattr(qwen2_5_vl, 'Qwen2_5_VLFlashAttention2'):
            qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = new_npu_flash_forward
        print("\033[32m[INFO] patch_qwen2_5_vl applied successfully!\033[0m")
            
    except Exception as e:
        print(f"\033[33m[WARNING] patch_qwen2_5_vl failed: {e}\033[0m")
        import traceback
        traceback.print_exc()
        return


# 应用补丁
def apply_all_patches():
    setup_npu_device()  # 启用NPU设备初始化
    patch_minimal_v4_dit()
    patch_qwen2_5_vl()
    print("\033[32m[INFO] All NPU patches for cosmos-predict2.5 applied successfully!\033[0m")


if __name__ == "__main__":
    apply_all_patches()
EOF


# ============== 第七步：先执行Monkey Patch，再运行推理脚本 ============

# 步骤1：执行补丁脚本，完成NPU初始化和运行时Patch
uv run python ${MONKEY_PATCH_FILE}
echo -e "\033[32m[INFO] Step 7-1: Apply NPU patches and initialize NPU successfully!\033[0m"


# 步骤2：单独运行推理脚本
echo -e "\033[32m[INFO] Step 7-2: Running inference script...\033[0m"
uv run python examples/inference.py \
-i assets/base/robot_pouring.json \
-o outputs/base_video2world \
--inference-type=video2world \
--model="2B/post-trained" \
--disable_guardrails

# =========================== 清理临时文件 ===========================
rm -f ${MONKEY_PATCH_FILE}
echo -e "\033[32m[INFO] Inference completed successfully!\033[0m"
echo -e "\033[32m[INFO] Output directory: outputs/base_video2world\033[0m"
