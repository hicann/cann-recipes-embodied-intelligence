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
DEFAULT_HOOK_INTERVAL = 200
CHECKPOINT_EPOCH_INTERVAL = 1
MAX_EPOCHS = 12
VAL_INTERVAL = 1
AUTO_SCALE_BASE_BATCH = 16
RANDOM_SEED = 3407  # 固定随机种子，保证实验可复现。3407 为常用 benchmark/公开项目默认种子，无特殊含义。

# Runtime hook settings for timing, logging, scheduler updates and checkpointing.
default_hooks = {
    'timer': {'type': 'IterTimerHook'},
    'logger': {'type': 'LoggerHook', 'interval': DEFAULT_HOOK_INTERVAL},
    'param_scheduler': {'type': 'ParamSchedulerHook'},
    'checkpoint': {
        'type': 'CheckpointHook',
        'interval': CHECKPOINT_EPOCH_INTERVAL,
    },
    'sampler_seed': {'type': 'DistSamplerSeedHook'},
}

env_cfg = {
    'cudnn_benchmark': False,
    'mp_cfg': {
        # 'fork' 方式在 CUDA/NPU 并行下可能导致死锁或上下文问题，使用前请确认兼容性。
        'mp_start_method': 'spawn',
        'opencv_num_threads': 0,
    },
    'dist_cfg': {'backend': 'nccl'},
}

log_processor = {
    'type': 'LogProcessor',
    'window_size': 50,
    'by_epoch': True,
}

LOG_LEVEL = 'INFO'
LOAD_FROM = None
RESUME = False

# Keep mmengine runtime keys while using UPPER_CASE constants.
log_level = LOG_LEVEL
load_from = LOAD_FROM
resume = RESUME

train_cfg = {
    'type': 'EpochBasedTrainLoop',
    'max_epochs': MAX_EPOCHS,
    'val_interval': VAL_INTERVAL,
}
val_cfg = {'type': 'ValLoop'}
test_cfg = {'type': 'TestLoop'}

auto_scale_lr = {
    'enable': False,
    'base_batch_size': AUTO_SCALE_BASE_BATCH,
}

visualizer = {
    'type': 'Visualizer',
    'vis_backends': [{'type': 'TensorboardVisBackend'}],
}

randomness = {'seed': RANDOM_SEED}
