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

from typing import List, Optional, Tuple

import torch

def _init_op_api_so_path(so_path: str) -> None: ...

def projection_three_dims_gaussian_fused(
    means: torch.Tensor,
    covars: torch.Tensor = None,
    viewmats: torch.Tensor = None,
    Ks: torch.Tensor = None,
    width: int = 0,
    height: int = 0,
    eps: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model = 'pinhole'
) -> torch.Tensor: ...

def gaussian_build_mask(
    means,
    radii,
    tile_grid,
    image_width,
    image_height,
    tile_size
) -> torch.Tensor:...

def gaussian_sort(
    lb_sched,
    depths,
    gs_ids,
    sorted_offset,
    max_tile_gauss
) -> torch.Tensor:...

def calc_render(
    means,
    conic0s,
    conic1s,
    conic2s,
    opacities,
    colors,
    depths,
    tile_coords,
    offsets,
    sorted_gs_ids
) -> torch.Tensor: ...

def spherical_harmonics(
    degrees_to_use,
    dirs,
    coeffs
) -> torch.Tensor:...

def get_render_schedule(nums_tensor: torch.Tensor, num_bins: int) -> torch.Tensor: ...

def flash_gaussian_build_mask(
            means2d,
            opacity,
            conics,
            covars2d,
            depths,
            cnt,
            tile_grid,
            image_width,
            image_height,
            tile_size=64
) -> torch.Tensor:...

def gaussian_filter(
    means,
    colors,
    det,
    opacities,
    means2d,
    depths,
    radius,
    conics,
    covars2d,
    compensations,
    width,
    height,
    near_plane,
    far_plane,
) -> torch.Tensor:...

__all__ = [
    "projection_three_dims_gaussian_fused",
    "calc_render",
    "gaussian_sort",
    "get_render_schedule",
    "spherical_harmonics",
    "flash_gaussian_build_mask",
    "gaussian_filter"
]
