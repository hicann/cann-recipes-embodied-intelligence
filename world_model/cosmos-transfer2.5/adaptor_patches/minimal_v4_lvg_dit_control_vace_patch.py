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

from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import (
    MiniTrainDITImageContext,
    MinimalV4LVGControlVaceDiT,
    Block,
    ControlAwareDiTBlock,
    ControlEncoderDiTBlock
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


def MiniTrainDITImageContext__init__(
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
                    share_q_in_i2v_cross_attn=share_q_in_i2v_cross_attn,
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


def MinimalV4LVGControlVaceDiT__init__(
        self,
        *args,
        crossattn_emb_channels: int = 1024,
        mlp_ratio: float = 4.0,
        vace_has_mask: bool = False,
        vace_block_every_n: int = 2,
        condition_strategy: Literal["spaced", "first_n"] = "spaced",
        num_max_modalities: int = 8,
        use_input_hint_block: bool = False,
        spatial_compression_factor: int = 8,
        num_control_branches: int = 1,
        separate_embedders: bool = False,
        use_after_proj_for_multi_branch: bool = True,
        timestep_scale: float = 1.0,  # Add timestep scaling for rectified flow
        use_cuda_graphs: bool = False,
        **kwargs,
    ):
        """
        vace_block_every_n: create one control block every n base model blocks
        vace_has_mask: if true, control branch latent is [inactive, reactive, mask] as in VACE paper. Otherwise, just the latent of the control input
        condition_strategy: How the control blocks correspond to the base model blocks. "first_n" conditions first n base model blocks.
            "spaced" conditions every vace_block_every_n base model block. E.g. vace_block_every_n=2, condition_strategy="spaced" means control block 0
            controls base block 0 and 2, control block 1 controls base block 2, etc.
        use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.
        """

        assert "in_channels" in kwargs, "in_channels must be provided"

        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        nf = kwargs["model_channels"]
        hint_nf = kwargs.pop("hint_nf", [nf, nf, nf, nf, nf, nf, nf, nf])
        self.num_control_branches = num_control_branches
        self.use_after_proj_for_multi_branch = use_after_proj_for_multi_branch
        self.timestep_scale = timestep_scale  # Store timestep scale for rectified flow
        self.use_cuda_graphs = use_cuda_graphs
        self.cuda_graphs = {}
        self.controlnet_cuda_graphs = {}
        self.cfg_parallel = False
        super(MiniTrainDITImageContext, self).__init__(
            *args,
            crossattn_emb_channels=crossattn_emb_channels,
            mlp_ratio=mlp_ratio,
            **kwargs,
        )

        self.crossattn_emb_channels = crossattn_emb_channels
        self.mlp_ratio = mlp_ratio

        # if vace_has_mask, the control latent is 16 + 64 (for mask)
        self.vace_has_mask = vace_has_mask
        self.num_max_modalities = num_max_modalities
        in_channels = self.in_channels - 1  # subtract the condition mask
        self.vace_in_channels = (in_channels + spatial_compression_factor**2) if vace_has_mask else in_channels
        self.vace_in_channels *= num_max_modalities
        self.vace_in_channels += 1  # adding the condition mask back

        # for finding corresponding control block with base model block.
        self.condition_strategy = condition_strategy
        if self.condition_strategy == "spaced":
            # base block k uses the 2k'th element in the hint list, {0:0, 2:1, 4:2, ...}, as in VACE paper
            self.control_layers = [i for i in range(0, self.num_blocks, vace_block_every_n)]
            self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}
        elif self.condition_strategy == "first_n":
            # condition first n base model blocks, where n is number of control blocks
            self.control_layers = list(range(0, self.num_blocks // vace_block_every_n))
            self.control_layers_mapping = {i: i for i in range(len(self.control_layers))}
        else:
            raise ValueError(f"Invalid condition strategy: {self.condition_strategy}")
        assert 0 in self.control_layers

        # Input hint block
        self.use_input_hint_block = use_input_hint_block
        if use_input_hint_block:
            assert self.num_control_branches == 1, "input hint block is not supported for multi-branch"
            input_hint_block = []
            nonlinearity = nn.SiLU()
            for i in range(len(hint_nf) - 1):
                input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
            self.input_hint_block = nn.Sequential(*input_hint_block)

        # -------- Base model --------

        # Base model blocks. Overwrite them to enable accepting the control branch modulations ("hints").
        # Shape remains the same as the base model so we can load pretrained weights.
        self.blocks = nn.ModuleList(
            [
                ControlAwareDiTBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    block_id=self.control_layers_mapping[i] if i in self.control_layers else None,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                )
                for i in range(self.num_blocks)
            ]
        )

        # -------- Control branch --------
        self.separate_embedders = separate_embedders
        if separate_embedders:
            self.t_embedder_for_control_branch = nn.Sequential(
                Timesteps(self.model_channels),
                TimestepEmbedding(self.model_channels, self.model_channels, use_adaln_lora=self.use_adaln_lora),
            )
            self.t_embedding_norm_for_control_branch = RMSNorm(self.model_channels, eps=1e-6)

        self.build_patch_embed_vace()

        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                setattr(
                    self,
                    f"control_blocks_{nc}",
                    nn.ModuleList(
                        [
                            ControlEncoderDiTBlock(
                                x_dim=self.model_channels,
                                context_dim=self.crossattn_emb_channels,
                                num_heads=self.num_heads,
                                mlp_ratio=self.mlp_ratio,
                                use_adaln_lora=self.use_adaln_lora,
                                adaln_lora_dim=self.adaln_lora_dim,
                                backend=self.atten_backend,
                                image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                                block_id=i,
                                hint_dim=hint_nf[-1] if use_input_hint_block else None,
                                use_after_proj=not use_after_proj_for_multi_branch,
                                use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                                use_cuda_graphs=self.use_cuda_graphs,
                            )
                            for i in self.control_layers
                        ]
                    ),
                )
            if use_after_proj_for_multi_branch:
                self.after_proj = nn.ModuleList(
                    [
                        nn.Linear(self.model_channels * self.num_control_branches, self.model_channels)
                        for _ in range(len(self.control_layers))
                    ]
                )
        else:
            self.control_blocks = nn.ModuleList(
                [
                    ControlEncoderDiTBlock(
                        x_dim=self.model_channels,
                        context_dim=self.crossattn_emb_channels,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        use_adaln_lora=self.use_adaln_lora,
                        adaln_lora_dim=self.adaln_lora_dim,
                        backend=self.atten_backend,
                        image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                        block_id=i,
                        hint_dim=hint_nf[-1] if use_input_hint_block else None,
                        use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                        use_cuda_graphs=self.use_cuda_graphs,
                    )
                    for i in self.control_layers
                ]
            )

        self.init_weights()
        sac_config = kwargs.get("sac_config", SACConfig())
        self.enable_selective_checkpoint(sac_config, self.blocks)
        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                self.enable_selective_checkpoint(sac_config, getattr(self, f"control_blocks_{nc}"))
        else:
            self.enable_selective_checkpoint(sac_config, self.control_blocks)


MiniTrainDITImageContext.__init__ = MiniTrainDITImageContext__init__
MinimalV4LVGControlVaceDiT.__init__ = MinimalV4LVGControlVaceDiT__init__
