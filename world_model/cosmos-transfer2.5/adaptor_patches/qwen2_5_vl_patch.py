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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.utils
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


try:
    from torch.distributed.tensor import Shard
except ImportError:
    print("torch.distributed.tensor is not available. DeepSeek model will not work.")

try:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
except ImportError:
    print("transformer version too old, please upgrade to latest version, qwen model is not supported")
    Qwen2_5_VLConfig = dict
    Qwen2_5_VLVisionConfig = dict

from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
)

from cosmos_transfer2._src.reason1.networks.qwen2_5_vl import (
    Qwen2_5_VLFlashAttention2,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

original_is_flash_attn_2_available = transformers.utils.is_flash_attn_2_available


def _is_flash_attn_2_available_patched():
    """Patched version that always returns False to bypass flash_attn_2 check."""
    return False


def patch_forward(
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
        query_states, key_states = apply_multimodal_rotary_pos_emb(
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
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

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

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None
        # key_stats: 1, seq_len, 16, 128
        # query_states: 1, seq_len, 16, 128
        if cp_mesh is not None:
            raise NotImplementedError(
                "CP is not supported for flash attention2, _flash_attention_forward will produce wrong output if query_states is sharded"
            )

        # 转换输入：BSND为BNSD
        query_bnsd = query_states.permute(0, 2, 1, 3).contiguous()
        key_bnsd = key_states.permute(0, 2, 1, 3).contiguous()
        value_bnsd = value_states.permute(0, 2, 1, 3).contiguous()

        # 调用原生注意力接口
        attn_output_bnsd = F.scaled_dot_product_attention(
            query_bnsd, 
            key_bnsd, 
            value_bnsd, 
            attn_mask=None, 
            dropout_p=0.0,
            is_causal=True
        )

        # 由BNSD转换回BSND
        attn_output = attn_output_bnsd.permute(0, 2, 1, 3).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


Qwen2_5_VLFlashAttention2.forward = patch_forward
transformers.utils.is_flash_attn_2_available = _is_flash_attn_2_available_patched
