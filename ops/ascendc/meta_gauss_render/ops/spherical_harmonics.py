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

import torch
from torch.autograd import Function
import torch.nn.functional as F

import meta_gauss_render._C


class SphericalHarmonicsForward(Function):
    @staticmethod
    def forward(ctx,
                degrees_to_use: int,
                dirs: torch.Tensor,
                coeffs: torch.Tensor):
        if degrees_to_use > 4 or degrees_to_use < 0:
            raise ValueError("Spherical harmonics order should be 0 ~ 4, but got degrees which is not supported.")
        ctx.save_for_backward(dirs, coeffs)
        ctx.degree = degrees_to_use
        output = meta_gauss_render._C.spherical_harmonics_forward(
                dirs,
                coeffs,
                degrees_to_use
        )
        return output

    @staticmethod
    def backward(ctx, *args):
        v_colors = args[0]
        dirs, coeffs = ctx.saved_tensors
        degree = ctx.degree
        v_dirs, v_coeffs = meta_gauss_render._C.spherical_harmonics_bwd(
                dirs,
                coeffs,
                v_colors,
                degree
        )
        return None, v_dirs, v_coeffs

spherical_harmonics = SphericalHarmonicsForward.apply

