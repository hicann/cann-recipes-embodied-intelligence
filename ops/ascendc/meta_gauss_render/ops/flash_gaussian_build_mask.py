# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from typing_extensions import Literal

import torch
import torch_npu
from torch.autograd import Function
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F

import meta_gauss_render._C


class FlashGaussianBuildMask(Function):
    @staticmethod
    def forward(ctx,
                means2d: torch.Tensor,
                opacity: torch.Tensor,
                conics: torch.Tensor,
                covars2d: torch.Tensor,
                depths: torch.Tensor,
                cnt: torch.Tensor,
                tile_grid: torch.Tensor,
                image_width,
                image_height,
                tile_size=64):

        if opacity is None:
            raise ValueError("Opacity must be Tensor while using FlashGS.")

        tile_sum, tile_offset, tile_depths, gauss_index = meta_gauss_render._C.flash_gaussian_build_mask(
            means2d,
            opacity,
            conics,
            covars2d,
            depths,
            cnt,
            tile_grid,
            float(image_width),
            float(image_height),
            tile_size
        )
        return tile_sum, tile_offset, tile_depths, gauss_index

flash_gaussian_build_mask = FlashGaussianBuildMask.apply