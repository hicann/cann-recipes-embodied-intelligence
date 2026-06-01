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

import meta_gauss_render._C


class GaussianSort(Function):
    @staticmethod
    def forward(
        ctx,
        lb_sched: torch.Tensor,
        gaussian_cnt: torch.Tensor,
        depths: torch.Tensor,
        gs_ids: torch.Tensor,
        sorted_offset: torch.Tensor,
        max_tile_gauss: int,
    ):
        sorted_gs_ids = meta_gauss_render._C.gaussian_sort(
            lb_sched, gaussian_cnt, depths, gs_ids, sorted_offset, max_tile_gauss
        )
        return sorted_gs_ids


gaussian_sort = GaussianSort.apply
