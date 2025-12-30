#!/usr/bin/env python
# Adapted from
# lerobot/lerobot/common/policies/pi0/paligemma_with_expert.py
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Union
import math

import torch
import torch.version
import torch_npu
from torch_npu.contrib import transfer_to_npu
from pytest import Cache
from torch import nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from lerobot.common.policies.pi0.flex_attention import flex_attention_forward


def apply_rope(query_states, key_states, cos, sin):
    n_q = query_states.shape[2]
    n_k = key_states.shape[2]
    
    merged_states = torch.cat([query_states, key_states], dim=2)  # 维度为[B, S, n_q + n_k, D]
    
    merged_rot = torch_npu.npu_rotary_mul(merged_states, cos, sin)
    
    q_rot, k_rot = merged_rot.split([n_q, n_k], dim=2)

    return q_rot, k_rot


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()
        
        self.models = [self.paligemma.language_model.model, self.gemma_expert.model]
        self.ones_add = torch.ones(1, dtype=torch.bfloat16, device="npu")
        
        self.num_att_heads_forward = self.config.paligemma_config.text_config.num_attention_heads
        self.num_key_value_heads_forward = self.config.paligemma_config.text_config.num_key_value_heads
        self.head_dim = self.models[0].layers[0].self_attn.head_dim
        self.scale_value = 1.0 / math.sqrt(self.head_dim)

        d_half = self.head_dim // 2
        self.timescale = 10_000 ** ((2.0 / (d_half * 2)) * torch.arange(d_half, dtype=torch.float32, device="npu"))

    # 将q/k/v权重融合为单个线性层qkv
    @torch.no_grad()
    def fuse_qkv_weights(self):
        """
        将每层 self_attn 的 q/k/v 权重 concat 成单个 qkv 线性层
        """
        for model in self.models:  # paligemma + gemma_expert
            for layer in model.layers:      
                attn = layer.self_attn
                w_q = attn.q_proj.weight
                w_k = attn.k_proj.weight
                w_v = attn.v_proj.weight

                w_fused = torch.cat([w_q, w_k, w_v], dim=0).bfloat16()

                # 创建 qkv 线性层
                attn.qkv = nn.Linear(
                    w_fused.shape[1],
                    w_fused.shape[0],
                    bias=False,
                    device=w_q.device,
                ).to(torch.bfloat16)

                # 将拼接后的权重赋值给 qkv 权重              
                attn.qkv.weight.data.copy_(w_fused)

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.model.embed_tokens(tokens)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        attention_mask = torch.logical_not(attention_mask).to(dtype=torch.int8, memory_format=torch.contiguous_format)
        attention_mask = attention_mask[:, None, :, :]
        
        radians = position_ids[..., None].to(torch.float32) / self.timescale[None, None, :]
        radians = radians[..., None, :]

        cos = torch.cos(radians)
        sin = torch.sin(radians)

        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        head_dim = self.head_dim
        batch_size = next(h.shape[0] for h in inputs_embeds if h is not None)
        
        num_q_heads = self.models[0].layers[0].self_attn.config.num_attention_heads
        num_kv_heads = self.models[0].layers[0].self_attn.config.num_key_value_heads
        
        q_out = num_q_heads * head_dim
        kv_out = num_kv_heads * head_dim

        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []

            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = self.models[i].layers[layer_idx]
                hidden_states = hidden_states.to(dtype=torch.bfloat16)

                hidden_states = torch_npu.npu_rms_norm(
                    hidden_states,
                    layer.input_layernorm.weight.add(self.ones_add),
                    1e-6
                )[0]

                hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)
                
                qkv = layer.self_attn.qkv(hidden_states)

                q_proj, k_proj, v_proj = qkv.split([q_out, kv_out, kv_out], dim=-1)

                q_proj = q_proj.view(hidden_shape)
                k_proj = k_proj.view(hidden_shape)
                v_proj = v_proj.view(hidden_shape)

                query_states.append(q_proj)
                key_states.append(k_proj)
                value_states.append(v_proj)

            # 拼接 and RoPE (复用计算好的 cos 和 sin)
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states, key_states = apply_rope(query_states, key_states, cos, sin)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_output = att_output.to(dtype=torch.bfloat16)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = self.models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    out_emb, _, after_first_residual = torch_npu.npu_add_rms_norm(
                        layer.self_attn.o_proj(att_output[:, start:end]),
                        hidden_states.to(torch.bfloat16),
                        layer.post_attention_layernorm.weight.add(self.ones_add),
                        1e-6
                    )

                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout (by default 0.0)

                    # second residual
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = self.models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        elif self.config.attention_implementation == "flex":
            attention_interface = flex_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        att_output = torch_npu.npu_prompt_flash_attention(
            query_states,
            key_states.contiguous(),
            value_states.contiguous(),
            num_heads=self.num_att_heads_forward,
            input_layout="BSND",
            scale_value=self.scale_value,
            pre_tokens=65535,
            next_tokens=65535,
            atten_mask=attention_mask,
            num_key_value_heads=self.num_key_value_heads_forward
        )

        att_output = att_output.reshape(batch_size, -1, self.num_att_heads_forward * head_dim)

        return att_output
