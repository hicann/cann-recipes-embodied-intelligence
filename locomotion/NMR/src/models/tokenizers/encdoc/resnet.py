# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, Shanghai AI Laboratory. All rights reserved.
#
# Portions are adapted from VankouF/MotionMillion-Codes under Apache-2.0.
# Use and redistribution of those portions remain subject to the upstream
# Apache-2.0 license. NMR modifications are licensed under Apache-2.0.
# See THIRD_PARTY_LICENSES.md.
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
import torch.nn.functional as F


def _build_norm(norm_name, channels):
    if norm_name == 'LN':
        return nn.LayerNorm(channels)
    if norm_name == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    if norm_name == 'BN':
        return nn.BatchNorm1d(num_features=channels, eps=1e-6, affine=True)
    return nn.Identity()


def _build_activation(name):
    if name == 'relu':
        return nn.ReLU()
    if name == 'silu':
        return Nonlinearity()
    if name == 'gelu':
        return nn.GELU()
    raise NotImplementedError(f'Unsupported activation: {name}')


def _apply_norm_1d(x, norm_layer, norm_name):
    if norm_name == 'LN':
        return norm_layer(x.transpose(-2, -1)).transpose(-2, -1)
    return norm_layer(x)


class ResConv1DBlock(nn.Module):

    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.0, kernel_size=3):
        super().__init__()
        self.norm_name = norm
        padding = dilation * (kernel_size - 1) // 2

        self.norm1 = _build_norm(norm, n_in)
        self.norm2 = _build_norm(norm, n_state)
        self.activation1 = _build_activation(activation)
        self.activation2 = _build_activation(activation)

        self.conv1 = nn.Conv1d(
            n_in,
            n_state,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.activation1(_apply_norm_1d(x, self.norm1, self.norm_name))
        x = self.conv1(x)
        x = self.activation2(_apply_norm_1d(x, self.norm2, self.norm_name))
        x = self.dropout(self.conv2(x))
        return x + residual


class Nonlinearity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
class CausalResConv1DBlock(nn.Module):

    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        self.norm_name = norm
        self.norm1 = _build_norm(norm, n_in)
        self.norm2 = _build_norm(norm, n_state)
        self.activation1 = _build_activation(activation)
        self.activation2 = _build_activation(activation)

        self.left_padding = (3 - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        x = self.activation1(_apply_norm_1d(x, self.norm1, self.norm_name))
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv1(x)
        x = self.activation2(_apply_norm_1d(x, self.norm2, self.norm_name))
        x = self.conv2(x)
        return x + residual


class Resnet1D(nn.Module):

    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation='relu',
        norm=None,
        kernel_size=3,
    ):
        super().__init__()

        stage_blocks = [
            ResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate**depth,
                activation=activation,
                norm=norm,
                kernel_size=kernel_size,
            )
            for depth in range(n_depth)
        ]
        stage_blocks = list(reversed(stage_blocks)) if reverse_dilation else stage_blocks
        self.blocks = nn.ModuleList(stage_blocks)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResnet1D(nn.Module):

    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation='relu',
        norm=None,
    ):
        super().__init__()

        stage_blocks = [
            CausalResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate**depth,
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            stage_blocks.reverse()
        self.blocks = nn.ModuleList(stage_blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
