# Adapted from
# Isaac-GR00T/adaptor_patches/gr00t_n1d6_patch.py
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

import torch

from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from transformers.feature_extraction_utils import BatchFeature


@torch.no_grad()
def patched_get_action_with_features(
    self,
    backbone_features: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    backbone_output: BatchFeature,
) -> BatchFeature:
    vl_embeds = backbone_features

    batch_size = vl_embeds.shape[0]
    actions = torch.randn(
        size=(batch_size, self.config.action_horizon, self.action_dim),
        dtype=vl_embeds.dtype,
        device=vl_embeds.device, 
    )

    dt = 1.0 / self.num_inference_timesteps

    for t in range(self.num_inference_timesteps):
        t_cont = t / float(self.num_inference_timesteps)
        t_discretized = int(t_cont * self.num_timestep_buckets)

        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=vl_embeds.device
        )
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=vl_embeds.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
            
        pred = self.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -self.action_horizon:]

        actions = actions + dt * pred_velocity

    return BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )


Gr00tN1d6ActionHead.get_action_with_features = patched_get_action_with_features