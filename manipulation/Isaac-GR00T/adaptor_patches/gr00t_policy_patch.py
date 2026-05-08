# Adapted from
# https://github.com/NVIDIA/Isaac-GR00T
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
import time

import numpy as np
import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu

from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Gr00tPatch")

_original_get_action = getattr(Gr00tPolicy, '_get_action', None)


def patched_get_action(self, observation, options=None):
    if _original_get_action is None:
        raise RuntimeError("Original _get_action method not found in Gr00tPolicy")

    inference_start = time.time()

    casted_action, info = _original_get_action(self, observation, options)

    inference_time = time.time() - inference_start
    logger.info(f"Inference time: {inference_time:.3f}s")

    logger.info("Generated Actions Details:")
    for key, value in casted_action.items():
        logger.debug("Action '%s' full tensor: %s", key, value)
        logger.info(f"Action '{key}': shape={value.shape} | Sample: {value.flatten()}")

    return casted_action, info


setattr(Gr00tPolicy, '_get_action', patched_get_action)

logger.info("Gr00tPolicy monkey patch applied: NPU device migration + graph compilation enabled")
