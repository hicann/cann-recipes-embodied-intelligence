# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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
import torch.nn as nn
import torch
from quant.vggt_linear import LinearW8A8


def mark_ignore(module, ignore_quantize):
    for child in module.children():
        if isinstance(child, torch.nn.Linear):
            if (child.in_features != 4096): 
                child.ignore_quantize = ignore_quantize
        else:
            mark_ignore(child, ignore_quantize)  


def set_ignore_quantize(model, ignore_quantize=True):
    for module in [model.aggregator.patch_embed,
                model.aggregator.frame_blocks,
                model.aggregator.global_blocks,
                model.camera_head,
                model.point_head,
                model.depth_head,
                model.track_head]:
        mark_ignore(module, ignore_quantize)


def replace_linear_in_vggt(module, device):
    if isinstance(module, nn.Linear):
        if device is not None:
            module = module.to(device)

        if getattr(module, 'ignore_quantize', False):
            return module

        new_layer = LinearW8A8(module)
        return new_layer
    else:
        for name, child in module.named_children():
            new_child = replace_linear_in_vggt(child, device)
            if new_child is not child:
                setattr(module, name, new_child)
        if device is not None:
            module.to(device=device)
        return module
