# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
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
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module(name="RetargetTransformerPredMotionNoSMPLVQ")
class RetargetTransformerPredMotionNoSMPLVQ(BaseModel):
    def __init__(
        self, transformer_cfg: Dict, smplx_vqvae_cfg: Dict, n_embd: int = 512, **kwargs
    ):
        super().__init__(**kwargs)
        self.transformer = MODELS.build(transformer_cfg)
        self.motion_encoder = nn.Sequential(
            nn.Linear(512, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )

        self.smplx_vqvae = MODELS.build(smplx_vqvae_cfg)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(n_embd, n_embd, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(n_embd, n_embd, 3, 1, 1),
        )

        # 217 是宇树g1机器人运动特征维度，本项目针对宇树g1机器人进行设计，因此此处使用硬编码
        # 如需适配其他机器人请修改此处
        self.projector = nn.Linear(n_embd, 217)

    def forward_loss(self, cond_embd, cond_mask, gt_motion):
        """
        motion_tokens: (B, T)
        label_tokens: (B, T)
        condition_features: list[tensor], b, n, c
        condition_masks: list[tensor], b, n
        """
        feat = self.transformer(cond_embd, cond_mask)  # B, T, C
        up_sample_feat = self.upsample(feat.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # B, T*2, C
        pred_motion = self.projector(up_sample_feat)

        new_mask = F.interpolate(
            cond_mask[None].float(), scale_factor=2, mode="nearest"
        )[0]
        loss_raw = F.smooth_l1_loss(pred_motion, gt_motion, reduction="none")
        loss = (loss_raw * new_mask[..., None]).sum() / (
            new_mask.sum() * loss_raw.shape[-1]
        )

        return dict(loss=loss)

    def forward_predict(
        self, cond_embd: torch.Tensor, cond_mask: torch.Tensor
    ):  # mix condition type
        feat = self.transformer(cond_embd, cond_mask)
        up_sample_feat = self.upsample(feat.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # B, T*2, C
        pred_motions = self.projector(up_sample_feat)
        new_mask = F.interpolate(
            cond_mask[None].float(), scale_factor=2, mode="nearest"
        )[0]
        pred_motion_lengths = new_mask.sum(dim=1)

        _pred_motions = []
        for pred_motion, length in zip(pred_motions, pred_motion_lengths):
            _pred_motions.append(pred_motion[: length.int()])
        return pred_motions, pred_motion_lengths

    def forward(
        self,
        motion: torch.Tensor = None,
        smplx_motion: torch.Tensor = None,
        motion_length: torch.Tensor = None,
        mode="loss",
        **kwargs
    ):
        """
        gt_motion: tensor, B, T, C
        smplx_motion: tensor, B, T, C
        motion_length: tensor, B
        condition_type: str, 'text', 'visual', 'audio', 'trajectory'
        condition_data:
            if condition_type == 'text': list[str], len B
            if condition_type == 'visual': tensor, B, N, C, H, W
            if condition_type == 'audio': tensor, B, T_audio, C
            if condition_type == 'trajectory': tensor, B, T, 2
        """

        batch_size, seq_len, _ = smplx_motion.shape
        smplx_motion_in = self.smplx_vqvae.preprocess(smplx_motion)
        smplx_motion_embd = self.smplx_vqvae.encoder(
            smplx_motion_in, motion_length=motion_length
        )
        smplx_motion_embd = smplx_motion_embd.permute(0, 2, 1)
        smplx_motion_embd = self.motion_encoder(smplx_motion_embd)
        masks = (
            torch.arange(seq_len // 2, device=smplx_motion_embd.device)[None].repeat(
                batch_size, 1
            )
            < motion_length[:, None] // 2
        )

        if mode == "loss":
            return self.forward_loss(smplx_motion_embd, masks, motion)
        else:
            return self.forward_predict(smplx_motion_embd, masks)
