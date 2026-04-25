# coding=utf-8
# Adapted from
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/hy3dgen/texgen/custom_rasterizer/custom_rasterizer/render.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This code is based on Tencent-Hunyuan's Hunyuan3D-2 library and the Hunyuan3D-2
# implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to Hunyuan3D-2 used by Tencent-Hunyuan team that trained the model.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from torch.utils.cpp_extension import load
import torch


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    extension = load(name="rasterize_image",
                     sources=["./hy3dgen/texgen/custom_rasterizer/rasterizer.cpp"])

    assert (pos.device == tri.device)  
    cpu_pos, cpu_tri, cpu_clamp_depth = pos.cpu(), tri.cpu(), clamp_depth.cpu()
    findices, barycentric = extension.rasterize_image(cpu_pos[0], cpu_tri, cpu_clamp_depth, resolution[1],
                                                      resolution[0], 1e-6, use_depth_prior)

    findices, barycentric = findices.to(pos.device), barycentric.to(pos.device)
    
    return findices, barycentric


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
