# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
# Copyright (c) 2025 Snap Inc. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Snap-Non-Commercial AND Apache-2.0
#
# Portions are adapted from snap-research/SnapMoGen under the Snap Inc.
# Non-Commercial License:
# https://github.com/snap-research/SnapMoGen/blob/main/LICENSE
# Upstream source file:
# https://github.com/snap-research/SnapMoGen/blob/main/model/cnn_networks.py
#
# This file is not Apache-2.0-only. Use and redistribution of the SnapMoGen
# portions are subject to the upstream non-commercial license. NMR
# modifications are licensed under Apache-2.0. See THIRD_PARTY_LICENSES.md.
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
import torch.nn as nn
import torch.nn.init as init
from mmengine.registry import MODELS

from .resnet import Resnet1D

# Ref: https://github.com/snap-research/SnapMoGen/blob/main/model/cnn_networks.py#L69


@MODELS.register_module()
class EncoderAttn(nn.Module):

    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        self._down_scale = 2
        kernel_size = stride_t * 2
        padding = stride_t // 2

        self.embed = nn.Sequential(
            nn.Conv1d(input_emb_width, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.stages = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv_res": nn.Sequential(
                            nn.Conv1d(
                                width,
                                width,
                                kernel_size=kernel_size,
                                stride=stride_t,
                                padding=padding,
                            ),
                            Resnet1D(
                                width,
                                depth,
                                dilation_growth_rate,
                                activation=activation,
                                norm=norm,
                            ),
                        ),
                        "attn": AttnBlock(width),
                    }
                )
                for _ in range(down_t)
            ]
        )

        self.outproj = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        self.apply(init_weights)

    def forward(self, x, motion_length):
        x = self.embed(x)
        cur_len = motion_length
        for stage in self.stages:
            x = stage["conv_res"](x)
            cur_len = torch.div(cur_len, self._down_scale, rounding_mode="floor")
            x = stage["attn"](x, cur_len)
        return self.outproj(x)


@MODELS.register_module()
class DecoderAttn(nn.Module):

    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        self._up_scale = 2

        self.embed = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU(),
        )

        self.stages = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "res_up": nn.Sequential(
                            Resnet1D(
                                width,
                                depth,
                                dilation_growth_rate,
                                reverse_dilation=True,
                                activation=activation,
                                norm=norm,
                            ),
                            nn.Upsample(scale_factor=self._up_scale, mode="nearest"),
                            nn.Conv1d(width, width, 3, 1, 1),
                        ),
                        "attn": AttnBlock(width),
                    }
                )
                for _ in range(down_t)
            ]
        )

        self.outproj = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1),
        )
        self.apply(init_weights)

    def forward(self, x, motion_length):
        x = self.embed(x)
        cur_len = motion_length
        for stage in self.stages:
            x = stage["res_up"](x)
            cur_len = cur_len * self._up_scale
            x = stage["attn"](x, cur_len)

        return self.outproj(x)


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.attn_block = nn.MultiheadAttention(
            in_channels,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, m_lens):
        seq = x.transpose(1, 2)
        key_mask = length_to_mask(m_lens, seq.shape[1])
        normed = self.norm(seq)

        attn_out, _ = self.attn_block(
            normed,
            normed,
            normed,
            key_padding_mask=~key_mask,
        )

        seq = seq + attn_out
        return seq.transpose(1, 2)


class MultiInputIdentity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, m_lens=None):
        return x


def length_to_mask(lengths: torch.Tensor, max_length=None):
    """
    - lengths: (B,)
    - return: (B, max_length)
    """
    max_length = lengths.max() if max_length is None else max_length
    steps = torch.arange(max_length, device=lengths.device)
    return steps.unsqueeze(0) < lengths.unsqueeze(1)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
