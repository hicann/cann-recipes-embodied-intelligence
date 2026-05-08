# Adapted from
# https://github.com/NVIDIA/Isaac-GR00T
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2026, PSI-lab USC.  All rights reserved.
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

import importlib
import time
import types
from dataclasses import dataclass

import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torch_npu.contrib import transfer_to_npu
from transformers.feature_extraction_utils import BatchFeature

import gr00t.model.gr00t_n1d6.gr00t_n1d6 as original_module
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead, Gr00tN1d6

_compiler_config = CompilerConfig()
_compiler_config.experimental_config.frozen_parameter = True
_compiler_config.experimental_config.tiling_schedule_optimize = True

_npu_backend = tng.get_npu_backend(compiler_config=_compiler_config)


@dataclass
class DiffusionLooperArgs:
    """Arguments container for diffusion loop execution."""
    actions: torch.Tensor
    vl_embeds: torch.Tensor
    state_features: torch.Tensor
    embodiment_id: torch.Tensor
    backbone_output: object
    dt: float


def _diffusion_looper(self, args: DiffusionLooperArgs):
    """
    Diffusion loop extracted for torch.compile optimization.

    This method is extracted from the original get_action_with_features
    to enable torch.compile with fullgraph=True on NPU.
    """
    batch_size = args.vl_embeds.shape[0]

    for t in range(self.num_inference_timesteps):
        t_cont = t / float(self.num_inference_timesteps)
        t_discretized = int(t_cont * self.num_timestep_buckets)
        timesteps_tensor = torch.full(
            (batch_size,), t_discretized, device=args.vl_embeds.device
        )

        action_features = self.action_encoder(args.actions, timesteps_tensor, args.embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], device=args.vl_embeds.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((args.state_features, action_features), dim=1)

        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=args.vl_embeds,
                timestep=timesteps_tensor,
                image_mask=args.backbone_output.image_mask,
                backbone_attention_mask=args.backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=args.vl_embeds,
                timestep=timesteps_tensor,
            )

        pred = self.action_decoder(model_output, args.embodiment_id)
        pred_velocity = pred[:, -self.action_horizon:]
        args.actions = args.actions + args.dt * pred_velocity

    return args.actions


def _get_action_with_features_patched(
    self,
    backbone_features: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    backbone_output,
):
    """
    Patched get_action_with_features using compiled diffusion looper.

    Changes from original:
    - Added NPU synchronization and timing
    - Uses compiled_diffusion_looper instead of inline loop
    """
    torch.npu.synchronize()
    time_start_0 = time.time()
    vl_embeds = backbone_features
    batch_size = vl_embeds.shape[0]

    actions = torch.randn(
        batch_size, self.config.action_horizon, self.config.max_action_dim,
        dtype=vl_embeds.dtype, device=vl_embeds.device
    )
    dt = 1.0 / self.num_inference_timesteps

    args = DiffusionLooperArgs(
        actions=actions,
        vl_embeds=vl_embeds,
        state_features=state_features,
        embodiment_id=embodiment_id,
        backbone_output=backbone_output,
        dt=dt
    )

    actions = self.compiled_diffusion_looper(args)

    result = BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )

    return result


def _get_action_patched(self, backbone_output, action_input):
    """
    Patched get_action method.

    Changes from original:
    - Added backbone_output.to(self.device) call
    """
    backbone_output.to(self.device)

    features = self._encode_features(backbone_output, action_input)

    result = self.get_action_with_features(
        backbone_features=features.backbone_features,
        state_features=features.state_features,
        embodiment_id=action_input.embodiment_id,
        backbone_output=backbone_output,
    )

    return result


def _patch_action_head_init(original_init):
    """
    Wrap Gr00tN1d6ActionHead.__init__ to add compiled diffusion looper.
    """
    def patched_init(self, config):
        original_init(self, config)

        # Add _diffusion_looper as a bound method, then compile it
        # This ensures self is properly passed when calling compiled_diffusion_looper
        self._diffusion_looper = types.MethodType(_diffusion_looper, self)

        # Add compiled diffusion looper for NPU optimization
        self.compiled_diffusion_looper = torch.compile(
            self._diffusion_looper,
            backend=_npu_backend,
            dynamic=False,
            fullgraph=True
        )

    return patched_init


def apply_patch():
    """
    Apply all monkey patches to gr00t_n1d6 module.

    This patches:
    - Gr00tN1d6ActionHead.__init__ (adds compiled diffusion looper)
    - Gr00tN1d6ActionHead.get_action_with_features (use compiled looper)
    - Gr00tN1d6ActionHead.get_action (add device transfer)
    """
    # Patch __init__ to add compiled diffusion looper (method binding happens in __init__)
    original_init = Gr00tN1d6ActionHead.__init__
    Gr00tN1d6ActionHead.__init__ = _patch_action_head_init(original_init)

    # Patch get_action_with_features
    Gr00tN1d6ActionHead.get_action_with_features = _get_action_with_features_patched

    # Patch get_action
    Gr00tN1d6ActionHead.get_action = _get_action_patched


apply_patch()
