# Adapted from
# Isaac-GR00T/adaptor_patches/gr00t_policy_patch.py
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

import logging

import numpy as np
import torch

from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Gr00tPatch")

_original_init = getattr(Gr00tPolicy, '__init__')
_original_get_action = getattr(Gr00tPolicy, '_get_action', None)


def patched_init(self, embodiment_tag, model_path, *, device, strict=True):
    # 调用原始初始化方法
    _original_init(self, embodiment_tag, model_path, device=device, strict=strict)
    
    # 迁移至 NPU
    logger.info(f"Moving model to NPU with bfloat16 (strict={strict})")
    self.model.to(torch.bfloat16).to("npu")
    self.model.eval()


def patched_get_action(self, observation, options=None):
    if _original_get_action is None:
        raise RuntimeError("Original _get_action method not found in Gr00tPolicy")
    
    casted_action, info = _original_get_action(self, observation, options)
    
    # 使用日志工具记录动作结果
    logger.info("Generated Actions Details:")
    for key, value in casted_action.items():
        logger.debug("Action '%s' full tensor: %s", key, value)
        logger.info(f"Action '{key}': shape={value.shape} | Sample: {value.flatten()}")
        
    return casted_action, info


setattr(Gr00tPolicy, '__init__', patched_init)
setattr(Gr00tPolicy, '_get_action', patched_get_action)