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

from typing import Optional
from dataclasses import dataclass

import torch

from gr00t.model.modules.dit import BasicTransformerBlock


@dataclass
class TransformerForwardInput:
    hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    temb: Optional[torch.LongTensor] = None


def patched_forward(
    self,
    hidden_states: torch.Tensor,
    **kwargs, 
) -> torch.Tensor:
    inputs = TransformerForwardInput(hidden_states=hidden_states, **kwargs)
    
    hidden_states = inputs.hidden_states
    attention_mask = inputs.attention_mask
    encoder_hidden_states = inputs.encoder_hidden_states
    encoder_attention_mask = inputs.encoder_attention_mask
    temb = inputs.temb

    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, temb)
    else:
        norm_hidden_states = self.norm1(hidden_states)

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    # 适配昇腾mask
    _attn_mask = encoder_attention_mask if encoder_hidden_states is not None else attention_mask
    if _attn_mask is not None:
        # [B, S] -> [B, 1, Sq, S]
        seq_len = norm_hidden_states.shape[1] 
        _attn_mask = _attn_mask.view(_attn_mask.shape[0], 1, 1, _attn_mask.shape[1])
        _attn_mask = _attn_mask.expand(-1, 1, seq_len, -1)  

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=_attn_mask,
    )
    
    if self.final_dropout:
        attn_output = self.final_dropout(attn_output)

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    norm_hidden_states = self.norm3(hidden_states)
    ff_output = self.ff(norm_hidden_states)

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)
        
    return hidden_states

BasicTransformerBlock.forward = patched_forward