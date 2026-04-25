# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch_npu
from einops import rearrange
from torch import Tensor, nn

from module.dit_cache.cache_method import cache_manager


def npu_fia(q, k, v, scale):
    attn_mask = None

    batch_size, num_head, seq_len, head_dim = q.shape

    out = torch_npu.npu_fused_infer_attention_score(
        q, k, v, num_heads=num_head, input_layout="BNSD", scale=scale, atten_mask=attn_mask
    )[0]
    return out


def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    batch_size, num_head, seq_len, head_dim = q.shape
    x = npu_fia(q, k, v, scale=(1 / math.sqrt(head_dim)))
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, 
    dtype=torch.float32, device=t.device) / half)


    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return torch_npu.npu_fast_gelu(x)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        return torch_npu.npu_rms_norm(x, self.scale, epsilon=self.eps)[0]


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


def first_block_forward(
    self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    enable_separate_cfg = cache_manager.enable_separate_cfg
    if not hasattr(self, 'cfg_step'):
        self.cfg_step = 0
    is_cond = True
    if enable_separate_cfg:
        is_cond = (self.cfg_step == 0)
        self.cfg_step = 1 - self.cfg_step
    if cache_manager.cache_method.cache_name == "TeaCache":
        judge_input = img_modulated
        args = {
            "latent": img,
            "judge_input": judge_input,
            "is_cond": is_cond
        }
        should_calc, img = cache_manager.cache_method.pre_cache_process(args)

        if not should_calc:
            cache_manager.cache_method.post_cache_update(img)
            cache_manager.cache_method.last_is_cond = is_cond
            return img, txt
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    q = torch.cat((txt_q, img_q), dim=2)
    k = torch.cat((txt_k, img_k), dim=2)
    v = torch.cat((txt_v, img_v), dim=2)

    attn = attention(q, k, v, pe=pe)
    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
    if cache_manager.cache_method.cache_name == "FBCache":
        args = {
            "latent": img,
            "judge_input": img.clone(),
            "is_cond": is_cond
        }
        should_calc, img = cache_manager.cache_method.pre_cache_process(args)
    
    return img, txt


def _double_block_full_compute(
    block_self,
    double_block_dict: dict,
    pe, cache_dic, current
):
    img = double_block_dict['img']
    txt = double_block_dict['txt']
    img_mod1 = double_block_dict['img_mod1']
    img_mod2 = double_block_dict['img_mod2']
    txt_mod1 = double_block_dict['txt_mod1']
    txt_mod2 = double_block_dict['txt_mod2']
    current['module'] = 'img_attn'
    img_modulated = block_self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = block_self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=block_self.num_heads)
    img_q, img_k = block_self.img_attn.norm(img_q, img_k, img_v)
    txt_modulated = block_self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = block_self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=block_self.num_heads)
    txt_q, txt_k = block_self.txt_attn.norm(txt_q, txt_k, txt_v)
    q = torch.cat((txt_q, img_q), dim=2)
    k = torch.cat((txt_k, img_k), dim=2)
    v = torch.cat((txt_v, img_v), dim=2)
    attn = attention(q, k, v, pe=pe)
    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]
    img_attn_out = block_self.img_attn.proj(img_attn)
    img = img + img_mod1.gate * img_attn_out
    cache_manager.cache_method.derivative_approximation(cache_dic, current, img_attn_out)

    current['module'] = 'img_mlp'
    img_mlp_in = (1 + img_mod2.scale) * block_self.img_norm2(img) + img_mod2.shift
    img_mlp_out = block_self.img_mlp(img_mlp_in)
    img = img + img_mod2.gate * img_mlp_out
    cache_manager.cache_method.derivative_approximation(cache_dic, current, img_mlp_out)

    current['module'] = 'txt_attn'
    txt_attn_out = block_self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod1.gate * txt_attn_out
    cache_manager.cache_method.derivative_approximation(cache_dic, current, txt_attn_out)

    current['module'] = 'txt_mlp'
    txt_mlp_in = (1 + txt_mod2.scale) * block_self.txt_norm2(txt) + txt_mod2.shift
    txt_mlp_out = block_self.txt_mlp(txt_mlp_in)
    txt = txt + txt_mod2.gate * txt_mlp_out
    cache_manager.cache_method.derivative_approximation(cache_dic, current, txt_mlp_out)
    return img, txt


def _double_block_taylor_compute(
    block_self,
    double_block_dict: dict,
    cache_dic, current
):
    img = double_block_dict['img']
    txt = double_block_dict['txt']
    img_mod1 = double_block_dict['img_mod1']
    img_mod2 = double_block_dict['img_mod2']
    txt_mod1 = double_block_dict['txt_mod1']
    txt_mod2 = double_block_dict['txt_mod2']
    current['module'] = 'img_attn'
    img_attn_out = cache_manager.cache_method.taylor_formula(cache_dic, current)
    img = img + img_mod1.gate * img_attn_out

    current['module'] = 'img_mlp'
    img_mlp_out = cache_manager.cache_method.taylor_formula(cache_dic, current)
    img = img + img_mod2.gate * img_mlp_out 

    current['module'] = 'txt_attn'
    txt_attn_out = cache_manager.cache_method.taylor_formula(cache_dic, current)
    txt = txt + txt_mod1.gate * txt_attn_out

    current['module'] = 'txt_mlp'
    txt_mlp_out = cache_manager.cache_method.taylor_formula(cache_dic, current)
    txt = txt + txt_mod2.gate * txt_mlp_out
    return img, txt


def double_block_forward(
    self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_dic = cache_manager.cache_method.cache_dic
    current = cache_manager.cache_method.current
    double_stream_layers = cache_manager.cache_method.double_stream_layers
    current_layer_idx = cache_manager.cache_method.layer_counter - 1
    current_layer_idx = current_layer_idx % double_stream_layers
    current.update({
        "stream": "double_stream",
        "layer": current_layer_idx,
        "step": cache_manager.cache_method.num_steps
    })
    cache_manager.cache_method.update_layer_counter()

    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    double_block_dict = {
        'img': img,
        'txt': txt,
        'img_mod1': img_mod1,
        'img_mod2': img_mod2,
        'txt_mod1': txt_mod1,
        'txt_mod2': txt_mod2
    }
    if current.get('type') == 'full':
        img, txt = _double_block_full_compute(
            self, double_block_dict, pe, cache_dic, current
        )
    elif current.get('type') == 'taylor_cache':
        img, txt = _double_block_taylor_compute(
            self, double_block_dict, cache_dic, current
        )
    else:
        raise ValueError(f"unsupported current type: {current.get('type')}!, Only full/taylor_cache are allowed")
    return img, txt


def _single_block_full_compute(
    block_self,
    single_block_dict: dict, 
    pe, cache_dic, current
):
    x = single_block_dict['x']
    mod = single_block_dict['mod']
    x_mod = (1 + mod.scale) * block_self.pre_norm(x) + mod.shift
    qkv, mlp = torch.split(block_self.linear1(x_mod), [3 * block_self.hidden_size, block_self.mlp_hidden_dim], dim=-1)

    q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=block_self.num_heads)
    q, k = block_self.norm(q, k, v)
    attn = attention(q, k, v, pe=pe)
    current['module'] = 'total'
    output = block_self.linear2(torch.cat((attn, block_self.mlp_act(mlp)), 2))
    cache_manager.cache_method.derivative_approximation(cache_dic, current, output)
    x = x + mod.gate * output
    return x


def _single_block_taylor_compute(
    block_self,
    single_block_dict: dict, 
    cache_dic, current
):
    x = single_block_dict['x']
    mod = single_block_dict['mod']
    current['module'] = 'total'
    output = cache_manager.cache_method.taylor_formula(cache_dic, current)
    x = x + mod.gate * output
    return x


def single_block_forward(
    self, x: Tensor, vec: Tensor, pe: Tensor
) -> Tensor:
    cache_dic = cache_manager.cache_method.cache_dic
    current = cache_manager.cache_method.current
    double_stream_layers = cache_manager.cache_method.double_stream_layers
    current_layer_idx = cache_manager.cache_method.layer_counter - 1
    single_layer_idx = (current_layer_idx - double_stream_layers) % cache_manager.cache_method.single_stream_layers
    current.update({
        "stream": "single_stream",
        "layer": single_layer_idx,
        "step": cache_manager.cache_method.num_steps
    })
    cache_manager.cache_method.update_layer_counter()
    mod, _ = self.modulation(vec)
    single_block_dict = {
        'x': x,
        'mod': mod,
    }
    if current.get('type') == 'full':
        x = _single_block_full_compute(
            self, single_block_dict, pe, cache_dic, current
        )
    elif current.get('type') == 'taylor_cache':
        x = _single_block_taylor_compute(
            self, single_block_dict, cache_dic, current
        )
    else:
        raise ValueError(f"unsupported current type: {current.get('type')}!, Only full/taylor_cache are allowed")
    return x
