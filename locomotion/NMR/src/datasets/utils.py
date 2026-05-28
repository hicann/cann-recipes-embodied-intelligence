# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
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

import torch
from torch.nn.utils.rnn import pad_sequence
from mmengine.registry import FUNCTIONS


@FUNCTIONS.register_module()
def motion_collate_fn(batch):
    motion = [item["motion"] for item in batch]  # [motion: T, C]
    motion_length = [item["motion_length"] for item in batch]

    motion = pad_sequence(motion, batch_first=True)
    mean = torch.stack([item["mean"] for item in batch], dim=0)
    std = torch.stack([item["std"] for item in batch], dim=0)

    return dict(
        std=std,
        mean=mean,
        motion=motion,
        motion_length=torch.tensor(motion_length),
    )


@FUNCTIONS.register_module()
def motion_collate_fn_no_translation(batch):
    motion = [item["motion"] for item in batch]  # [motion: T, C]
    motion_length = [item["motion_length"] for item in batch]

    motion = pad_sequence(motion, batch_first=True)
    mean = torch.stack([item["mean"] for item in batch], dim=0)
    std = torch.stack([item["std"] for item in batch], dim=0)

    return dict(
        std=std,
        mean=mean,
        motion=motion[..., 3:],
        motion_length=torch.tensor(motion_length),
    )
