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
from typing import Any, List, Literal, Optional, Tuple, Union

import megatron.core.parallel_state as parallel_state
import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange
from torch.distributed import ProcessGroup, get_process_group_ranks
from torchvision import transforms
import torch_npu

from cosmos_transfer2._src.imaginaire.utils.graph import create_cuda_graph
from cosmos_transfer2._src.predict2.conditioner import DataType
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import (
    Attention,
    FinalLayer,
    PatchEmbed,
    SACConfig,
    TimestepEmbedding,
    Timesteps,
    replace_selfattn_op_with_sparse_attn_op,
)
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import Block as BaseBlock
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT as BaseMiniTrainDIT

from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import (
    I2VCrossAttention,
    MiniTrainDIT,
    Attention,
    Block,
)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def I2VCrossAttention__init__(self, *args, img_latent_dim: int = 1024, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
        inner_dim = self.head_dim * self.n_heads
        self.k_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.v_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.q_img = nn.Linear(self._query_dim, inner_dim, bias=False)  # NEW: separate query for image attention
        self.q_img_norm = RMSNorm(self.head_dim, eps=1e-6)  # NEW: dedicated normalization for q_img
        self.k_img_norm = RMSNorm(self.head_dim, eps=1e-6)


def MiniTrainDIT__init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "transformer_engine",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        extra_image_context_dim: Optional[int] = None,  # Main flag of whether user reference image
        img_context_deep_proj: bool = False,  # work when extra_image_context_dim is not None
        share_q_in_i2v_cross_attn: bool = False,  # work when extra_image_context_dim is not None
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        sac_config: SACConfig = SACConfig(),
        n_dense_blocks: int = -1,
        gna_parameters=None,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        # Initialize the grandparent class (whatever the parent inherits from)
        super(BaseMiniTrainDIT, self).__init__()

        # Store parameters in the same order as parent class
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.extra_image_context_dim = extra_image_context_dim
        # NEW: Our additional parameters
        self.img_context_deep_proj = img_context_deep_proj
        self.share_q_in_i2v_cross_attn = share_q_in_i2v_cross_attn
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        # Component building (same order as parent)
        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels

        # Create blocks with our modified Block class
        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                    image_context_dim=None if extra_image_context_dim is None else model_channels,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        # Create image context projection with deep support
        if extra_image_context_dim is not None:
            if img_context_deep_proj:
                # Deep MLP projection
                self.img_context_proj = nn.Sequential(
                    nn.Linear(extra_image_context_dim, extra_image_context_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(extra_image_context_dim, model_channels, bias=False),
                    nn.LayerNorm(model_channels),
                )
            else:
                # Simple projection
                self.img_context_proj = nn.Sequential(
                    nn.Linear(
                        extra_image_context_dim, model_channels, bias=True
                    ),  # help distinguish between image and video context
                    nn.GELU(),
                )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

        # Replace self-attention with sparse attention if specified
        if n_dense_blocks != -1:
            self = replace_selfattn_op_with_sparse_attn_op(self, n_dense_blocks, gna_parameters=gna_parameters)

        self._is_context_parallel_enabled = False


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


def patch_compute_qkv(self, x, context=None, rope_emb=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                if self.use_wan_fp32_strategy:  # wan will force q and k to fp32 before rotary pos emb
                    q = q.to(torch.float32)
                    k = k.to(torch.float32)
                q = apply_rotary_pos_emb(q, rope_emb)
                k = apply_rotary_pos_emb(k, rope_emb)
                if self.use_wan_fp32_strategy:
                    q = q.to(original_dtype)
                    k = k.to(original_dtype)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v


I2VCrossAttention.__init__ = I2VCrossAttention__init__
MiniTrainDIT.__init__ = MiniTrainDIT__init__
Attention.compute_qkv = patch_compute_qkv
