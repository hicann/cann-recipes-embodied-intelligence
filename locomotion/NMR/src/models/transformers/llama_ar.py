# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
# Copyright (c) 2022, Andrej Karpathy.
# Copyright (c) 2025, Shanghai AI Laboratory. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Portions are based on nanoGPT under the MIT License and adapted from
# VankouF/MotionMillion-Codes under Apache-2.0.
#
# This file is not Apache-2.0-only. Use and redistribution of third-party
# portions are subject to their upstream licenses. NMR modifications are
# licensed under Apache-2.0. See THIRD_PARTY_LICENSES.md.
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
"""Full definition of a LLaMA Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from mmengine.model import BaseModel
from mmengine.registry import MODELS


@dataclass
class LLaMAHFConfig:
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    condition_dim: int = 512


@MODELS.register_module(name='LLaMAHF_AR')
class LlamaHfAr(BaseModel):

    def __init__(self, 
                 block_size: int = 4096, 
                 vocab_size: int = 32000, 
                 n_layer: int = 32, 
                 n_head: int = 32, 
                 n_embd: int = 4096,
                 condition_dim=512, 
                 **kwargs) -> None:
        '''
        end_token_idx: vocab size - 2
        pad_token_idx: vocab size - 1
        '''
        super().__init__(**kwargs)
        config = LLaMAHFConfig(
            block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, 
            n_head=n_head, n_embd=n_embd, condition_dim=condition_dim)
        if config.vocab_size is None:
            raise ValueError('vocab_size must not be None')
        if config.block_size is None:
            raise ValueError('block_size must not be None')
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )


    @torch.no_grad()
    def sample(self, condition_feature, condition_mask, if_categorial=False, sample_cnt=30):
        '''
        Retuen: 
            - generated_idx: Tensor, [B, T]
            - generated_length: Tensor, int, [B, ], the length before motion_end_idx (inclusive)
        '''
        # support batch inference
        batch_size, seq_len, _ = condition_feature.shape
        device = condition_feature.device
        generated_idx = []
        # right padding -> left padding
        flip_condition_mask = condition_mask.flip(1).bool()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) # B, L
        condition_feature[flip_condition_mask] = condition_feature[condition_mask.bool()]
        position_ids[flip_condition_mask] = position_ids[condition_mask.bool()]
        pre_embeddings = condition_feature
        past_key_values = [None] * len(self.transformer.h) # type: ignore
        for _ in range(sample_cnt):
            idx, past_key_values = self.forward_sample(
                pre_embeddings, flip_condition_mask, past_key_values, position_ids, if_categorial) # B
            position_ids = position_ids[:, -1:] + 1
            idx = idx.squeeze(-1)
            generated_idx.append(idx)
            pre_embeddings = self.transformer.wte(idx).unsqueeze(1) # B, 1, C # type: ignore
        generated_idx = torch.stack(generated_idx, dim=1) # B, T
        return generated_idx


    def forward_sample(self, pre_embeddings: Tensor, condition_mask: Tensor, 
                       past_key_values, position_ids, if_categorial=False):
        '''
        end_flag: Tensor, bool, [B, ]
        pre_embeddings: Tensor, [B, L, C]
        condition_mask: Tensor, [B, N, C]
        '''
        x = pre_embeddings.clone()
        new_past_key_values = []
        for i, block in enumerate(self.transformer.h): # type: ignore
            x, present_kv = block(x, condition_mask, past_key_values[i], position_ids)
            new_past_key_values.append(present_kv)
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        if if_categorial:
            idx = Categorical(probs).sample().unsqueeze(-1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
        return idx, new_past_key_values


    def forward(self, idx: Tensor, condition_features, condition_masks): # type: ignore
        batch_size, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
            )
         
        # B, T -> B, T, C
        x = self.transformer.wte(idx) # type: ignore
        condition_length = condition_features.shape[1]
        expanded_mask = condition_masks.unsqueeze(-1).expand(-1, -1, x.shape[-1]) # B, L -> B, L, C
        result = torch.where(expanded_mask == 1, condition_features, x[:, :condition_length, :])
        x = torch.cat((result, x[:, condition_length:]), dim=1)

        for block in self.transformer.h: # type: ignore
            x, _ = block(x, condition_masks)
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)  # (b, t, vocab_size)
        return logits


class Block(nn.Module):

    def __init__(self, config: LLaMAHFConfig) -> None: # , use_qkNorm=False, use_moe=False) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = LengthCausalSelfAttention(config) # , use_qkNorm)
        self.rms_2 = RMSNorm(config.n_embd)
        
        self.mlp = MLP(config)


    def forward(
        self,
        x: Tensor,
        y_mask: Tensor,
        past_key_values: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        attn_output, current_key_values = self.attn(self.rms_1(x), y_mask, past_key_values, position_ids)
        x = x + attn_output
        x = x + self.mlp(self.rms_2(x))
        
        return x, current_key_values


class LengthCausalSelfAttention(nn.Module):

    def __init__(self, config: LLaMAHFConfig) -> None: # , use_qkNorm=False) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError('n_embd must be divisible by n_head')

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None 


    def forward(
        self,
        x: Tensor,
        y_mask: Tensor,
        past_key_values: Optional[Tensor],
        position_ids: Optional[Tensor] = None,
    ):
        batch_size, seq_len, embd_dim = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = embd_dim // self.n_head
        k = k.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2).contiguous()  # (B, nh, T, hs)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )
        q = apply_rope(q, self.rope_cache, position_ids)
        k = apply_rope(k, self.rope_cache, position_ids)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        current_key_values = (k, v)

        # create attention mask
        total_steps = k.shape[2]
        attn_mask = torch.ones(total_steps, total_steps, dtype=torch.bool, device=x.device)
        attn_mask = torch.tril(attn_mask)
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)

        text_mask = y_mask.unsqueeze(2) * y_mask.unsqueeze(1)
        # Clamp valid text length to avoid negative padding when using cache.
        valid_len = min(y_mask.shape[1], total_steps)
        text_mask = text_mask[:, :valid_len, :valid_len]
        if valid_len < total_steps:
            pad = total_steps - valid_len
            text_mask = F.pad(
                text_mask,
                (0, pad, 0, pad),
                mode='constant',
                value=0,
            )
        attn_mask = torch.logical_or(attn_mask, text_mask)
        attn_mask = attn_mask[:, -q.size(2):, :]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False,
        )
        
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd_dim)

        y = self.c_proj(y)

        return y, current_key_values


class MLP(nn.Module):

    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        hidden_multiple = 256
        # ensure n_hidden is multiple of hidden_multiple
        n_hidden = ((n_hidden - 1) // hidden_multiple) * hidden_multiple + hidden_multiple

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)


    def forward(self, x: Tensor) -> Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """


    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim


    def forward(self, x: Tensor) -> Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Compute cache. Because polar only takes float32 or float64, we need to cast
    # when working with 16 bit floats (float16 or bfloat16)
    dtypes_requiring_casting = [torch.float16, torch.bfloat16, torch.int8]
    working_dtype = (
        torch.float32 if dtype in dtypes_requiring_casting else dtype
    )
    complex_dtype = (
        torch.complex32 if dtype in dtypes_requiring_casting else torch.complex64
    )
    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)
    return cache


def apply_rope(x: Tensor, rope_cache: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
    x = x.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
    batch_size, seq_len, num_heads, head_dim = x.size()

    # 统一 rope_cache 到 (batch_size, seq_len, num_heads, c)
    if position_ids is None:
        # rope_cache: (seq_len, c).
        rope_cache_ = rope_cache[:seq_len]
        rope_cache_ = rope_cache_.unsqueeze(0).unsqueeze(2)
        # (1, seq_len, 1, c) -> (batch, seq_len, num_heads, c)
        rope_cache_ = rope_cache_.expand(batch_size, seq_len, num_heads, -1)
    else:
        # rope_cache: (seq_len, c), position_ids: (batch, seq_len)
        rope_cache_ = rope_cache[position_ids]  # (batch, seq_len, c)
        rope_cache_ = rope_cache_.unsqueeze(2).expand(
            batch_size, seq_len, num_heads, -1
        )  # (batch, seq_len, num_heads, c)

    # 处理复数旋转
    # view_as_complex 只支持 float32/float64
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (batch, seq_len, num_heads, c/2)
    x_out = torch.view_as_real(xc * rope_cache_).flatten(3)  # (batch, seq_len, num_heads, head_dim)
    return x_out.transpose(1, 2).type_as(x)  # (batch, num_heads, seq_len, head_dim)
