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

import random
import torch
from mmengine.registry import DATASETS, FUNCTIONS
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import joblib


@DATASETS.register_module()
class RetargetDataset(Dataset):

    def __init__(
        self,
        split_file="",
        window_size=-1,
        unit_length=4,
        g1_mean_path="",
        g1_std_path="",
        smplx_mean_path="",
        smplx_std_path="",
        min_motion_length=60,
        max_motion_length=300,
    ):
        super().__init__()
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.window_size = window_size
        self.unit_length = unit_length

        raw_motions = joblib.load(split_file)
        self.motions = self._filter_by_min_length(raw_motions)
        if len(self.motions) == 0:
            raise ValueError(
                f"RetargetDataset: 过滤后无有效样本，请检查 min_motion_length="
                f"{self.min_motion_length} 或数据源 {split_file}"
            )
        self.g1_mean = torch.from_numpy(np.load(g1_mean_path))
        self.g1_std = torch.from_numpy(np.load(g1_std_path))
        self.smplx_mean = torch.from_numpy(np.load(smplx_mean_path))
        self.smplx_std = torch.from_numpy(np.load(smplx_std_path))

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        item = self.motions[index]
        if isinstance(item, dict):
            smplx_motion = item["smplx_motion"]
            g1_motion = item["g1_motion"]
            source_path = item.get("source_path", "")
        elif isinstance(item, tuple) and len(item) == 3:
            smplx_motion, g1_motion, source_path = item
        else:
            # fallback: 兼容其它格式，source_path设为''
            smplx_motion, g1_motion = item[:2]
            source_path = item[2] if len(item) > 2 else ""
        smplx_motion = torch.from_numpy(smplx_motion)
        g1_motion = torch.from_numpy(g1_motion)

        g1_motion = (g1_motion - self.g1_mean) / self.g1_std
        smplx_motion = (smplx_motion - self.smplx_mean) / self.smplx_std

        time_steps = g1_motion.shape[0]
        if self.window_size > 0:
            motion_length = self.window_size
            if time_steps < self.window_size:
                raise ValueError(
                    f"time_steps ({time_steps}) must be >= window_size ({self.window_size})"
                )
        else:
            motion_length = (time_steps // self.unit_length) * self.unit_length
            if motion_length > self.max_motion_length:
                motion_length = self.max_motion_length

        if motion_length <= 0:
            raise ValueError(
                f"motion_length must be positive, got {motion_length} "
                f"(time_steps={time_steps}, unit_length={self.unit_length})"
            )

        if time_steps < motion_length:
            raise ValueError(
                f"time_steps ({time_steps}) must be >= motion_length ({motion_length})"
            )
        idx = random.randint(0, time_steps - motion_length)
        smplx_motion = smplx_motion[idx:idx + motion_length]
        g1_motion = g1_motion[idx:idx + motion_length]

        return dict(
            g1_motion=g1_motion,
            smplx_motion=smplx_motion,
            motion_length=motion_length,
            mean=self.g1_mean,
            std=self.g1_std,
            smplx_mean=self.smplx_mean,
            smplx_std=self.smplx_std,
            source_path=source_path,
        )

    def _filter_by_min_length(self, motions):
        if self.min_motion_length <= 0:
            return motions

        filtered = []
        for item in motions:
            if isinstance(item, dict):
                g1_motion = item["g1_motion"]
                if g1_motion.shape[0] >= self.min_motion_length:
                    filtered.append(item)
            elif isinstance(item, tuple) and len(item) == 3:
                smplx_motion, g1_motion, source_path = item
                if g1_motion.shape[0] >= self.min_motion_length:
                    filtered.append((smplx_motion, g1_motion, source_path))
            else:
                # 兼容其它格式，直接保留
                filtered.append(item)
        return filtered


@FUNCTIONS.register_module()
def retarget_collate_fn(batch):
    g1_motion = [item["g1_motion"] for item in batch]  # [motion: T, C]
    smplx_motion = [item["smplx_motion"] for item in batch]
    motion_length = [item["motion_length"] for item in batch]

    g1_motion = pad_sequence(g1_motion, batch_first=True)
    smplx_motion = pad_sequence(smplx_motion, batch_first=True)
    mean = torch.stack([item["mean"] for item in batch], dim=0)
    std = torch.stack([item["std"] for item in batch], dim=0)
    smplx_mean = torch.stack([item["smplx_mean"] for item in batch], dim=0)
    smplx_std = torch.stack([item["smplx_std"] for item in batch], dim=0)

    source_paths = [item["source_path"] for item in batch]

    return dict(
        std=std,
        mean=mean,
        motion=g1_motion,
        smplx_motion=smplx_motion,
        motion_length=torch.tensor(motion_length),
        smplx_mean=smplx_mean,
        smplx_std=smplx_std,
        source_paths=source_paths,
    )
