# Adapted from
# https://github.com/NVIDIA/Isaac-GR00T
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

import math
import sys
import logging
import traceback
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch_npu

# NPU 头数限制
MAX_HEADS = 256

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NPU-Patch")


@dataclass
class FlashAttentionArgs:
    """FlashAttention NPU执行参数封装类"""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    num_heads: int
    scale: float
    input_layout: str = "BSND"
    pre_tokens: int = 65535
    next_tokens: int = 65535
    
    def __str__(self):
        return (
            f"FlashAttentionArgs(\n"
            f"  num_heads={self.num_heads}, scale={self.scale:.6f},\n"
            f"  input_layout={self.input_layout}, pre_tokens={self.pre_tokens},\n"
            f"  next_tokens={self.next_tokens}, device={self.q.device}\n"
            f")"
        )


@dataclass
class FlashAttentionForwardParams:
    module: torch.nn.Module
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    dropout: float = 0.0
    scaling: Optional[float] = None
    kwargs: dict = None


def patched_flash_attention_forward_for_packing(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None, ** kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    NPU适配的FlashAttention前向传播函数（支持head分片）
    
    Args:
        module: 调用该函数的Module实例
        query: 查询张量，shape [bsz, q_len, num_heads, head_dim]
        key: 键张量，shape [bsz, k_len, num_heads, head_dim]
        value: 值张量，shape [bsz, v_len, num_heads, head_dim]
        attention_mask: 注意力掩码，可选
        dropout: dropout概率，默认0.0
        scaling: 缩放因子，默认使用1/sqrt(head_dim)
        kwargs: 其他关键字参数
    
    Returns:
        Tuple[Tensor, None]: 注意力输出张量和None（保持原接口）
    """
    # 封装函数输入参数
    forward_params = FlashAttentionForwardParams(
        module=module,
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        dropout=dropout,
        scaling=scaling,
        kwargs=kwargs
    )
    
    # 数据类型转换和基础参数计算
    q, k, v = forward_params.query.bfloat16(), forward_params.key.bfloat16(), forward_params.value.bfloat16()
    bsz, q_len, num_heads, head_dim = q.shape
    scale = forward_params.scaling or (1.0 / math.sqrt(head_dim))
    
    # 封装FlashAttention执行参数
    fa_args = FlashAttentionArgs(
        q=q,
        k=k,
        v=v,
        num_heads=num_heads,
        scale=scale
    )
    
    outputs = []

    try:
        # 按MAX_HEADS分片执行NPU FlashAttention
        for i in range(0, fa_args.num_heads, MAX_HEADS):
            end = min(i + MAX_HEADS, fa_args.num_heads)
            curr_heads = end - i
            
            out = torch_npu.npu_prompt_flash_attention(
                query=fa_args.q[:, :, i:end, :].contiguous(),
                key=fa_args.k[:, :, i:end, :].contiguous(),
                value=fa_args.v[:, :, i:end, :].contiguous(),
                num_heads=curr_heads,
                input_layout=fa_args.input_layout,
                scale_value=fa_args.scale,
                pre_tokens=fa_args.pre_tokens,
                next_tokens=fa_args.next_tokens,
                atten_mask=None,
                sparse_mode=0,
            )
            outputs.append(out)

        # 拼接分片结果并返回
        final_output = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
        return final_output, None

    except torch.cuda.OutOfMemoryError:
        logger.error(
            "[NPU OOM] Memory limit reached. \n"
            "  Device: %s \n"
            "  Total Heads: %d \n"
            "  Max Heads per Chunk: %d",
            fa_args.q.device, fa_args.num_heads, MAX_HEADS
        )
        raise
    except RuntimeError as e:
        error_trace = traceback.format_exc()
        logger.error(
            "\n%s \n"
            "[NPU Patch RuntimeError] Operator failed! \n"
            "Context: \n"
            "  - FlashAttention Args: %s \n"
            "  - Query Tensor Shape: %s \n"
            "Traceback: \n%s \n%s",
            '=' * 60,
            fa_args,
            list(fa_args.q.shape),
            error_trace,
            '=' * 60 
        )
        raise e
    except Exception as e:
        logger.error(
            "[NPU Patch Unexpected Error] \n"
            "  Error Type: %s \n"
            "  Error Message: %s",
            type(e).__name__, e
        )
        raise e


def apply_patch():
    """Dynamically scan and replace Attention function in SigLIP2 modules."""
    # 检查NPU环境
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        logger.warning("[NPU-Patch] NPU device not found. Patch may fail.")

    targets = ['flash_attention_forward_for_packing', '_flash_attention_forward']
    patched_count = 0
    
    for name, mod in sys.modules.items():
        if 'modeling_siglip2' in name:
            for attr in targets:
                if hasattr(mod, attr):
                    setattr(mod, attr, patched_flash_attention_forward_for_packing)
                    patched_count += 1
                    logger.debug("[NPU-Patch] Replaced %s in %s", attr, name)

    if patched_count > 0:
        logger.info("[NPU-Patch] Applied %d patches.", patched_count)
    else:
        logger.warning("[NPU-Patch] Target module not found. Import modeling_siglip2 first!")

apply_patch()