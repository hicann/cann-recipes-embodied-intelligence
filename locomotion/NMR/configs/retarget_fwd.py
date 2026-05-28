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
_base_ = ["default_runtime.py"]
FIND_UNUSED_PARAMETERS = True
find_unused_parameters = FIND_UNUSED_PARAMETERS
CODE_DIM = 512
NB_CODE = 1920

MODEL_SIZE = "70M"
TRAIN_SPLIT_FILE = "data/train_posttrain.pkl"
VAL_SPLIT_FILE = "data/test_posttrain.pkl"
TEST_SPLIT_FILE = "data/test_posttrain.pkl"

SMPLX_MEAN_PATH = "data/smplx_data/smplx_mean.npy"
SMPLX_STD_PATH = "data/smplx_data/smplx_std.npy"
G1_MEAN_PATH = "data/gmr_data/gmr_mean.npy"
G1_STD_PATH = "data/gmr_data/gmr_std.npy"


LOAD_FROM = None
load_from = LOAD_FROM

smplx_vqvae_cfg = dict(
    encoder_cfg=dict(
        activation="relu",
        depth=3,
        dilation_growth_rate=3,
        down_t=2,
        input_emb_width=140,
        norm=None,
        output_emb_width=512,
        stride_t=2,
        type="EncoderAttn",
        width=512,
    ),
    type="VQVAE",
)

llama_configs = {
    "70M": dict(n_layer=8, n_head=8, n_embd=512),
    "111M": dict(n_layer=12, n_head=12, n_embd=768),
    "343M": dict(n_layer=24, n_head=16, n_embd=1024),
    "775M": dict(n_layer=36, n_head=20, n_embd=1280),
}

# model
model = dict(
    init_cfg=None,
    type="RetargetTransformerPredMotionNoSMPLVQ",
    transformer_cfg=dict(
        type="LlamaHfFwd",
        block_size=1024,  # max seq length
        vocab_size=llama_configs[MODEL_SIZE]["n_embd"],  # 不预测token, 直接出feature
        **llama_configs[MODEL_SIZE],
    ),
    n_embd=llama_configs[MODEL_SIZE]["n_embd"],
    smplx_vqvae_cfg=smplx_vqvae_cfg,
)

# =============================================
# dataset
# =============================================
train_dataloader = dict(
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="retarget_collate_fn"),
    dataset=dict(
        type="RetargetDataset",
        split_file=TRAIN_SPLIT_FILE,
        unit_length=4,
        window_size=-1,
        g1_mean_path=G1_MEAN_PATH,
        g1_std_path=G1_STD_PATH,
        smplx_mean_path=SMPLX_MEAN_PATH,
        smplx_std_path=SMPLX_STD_PATH,
        min_motion_length=60,
        max_motion_length=300,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="retarget_collate_fn"),
    dataset=dict(
        type="RetargetDataset",
        split_file=VAL_SPLIT_FILE,
        unit_length=4, window_size=-1,
        smplx_mean_path=SMPLX_MEAN_PATH,
        smplx_std_path=SMPLX_STD_PATH,
        g1_mean_path=G1_MEAN_PATH,
        g1_std_path=G1_STD_PATH,
        max_motion_length=300,
        min_motion_length=60,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="retarget_collate_fn"),
    dataset=dict(
        type="RetargetDataset",
        split_file=TEST_SPLIT_FILE,
        g1_mean_path=G1_MEAN_PATH,
        smplx_mean_path=SMPLX_MEAN_PATH,
        unit_length=4, window_size=-1,
        g1_std_path=G1_STD_PATH,
        smplx_std_path=SMPLX_STD_PATH,
        max_motion_length=300,
        min_motion_length=60,
    ),
)

val_evaluator = dict(
    type="HumanoidReconsMetric",
)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float32",
    optimizer=dict(type="AdamW", lr=2e-5, betas=(0.9, 0.99), weight_decay=0.0),
)

# training schedule
param_scheduler = [
    dict(type="LinearLR", by_epoch=False, start_factor=1e-5, begin=0, end=200),
    dict(type="CosineAnnealingLR", by_epoch=True, T_max=50, eta_min=1e-5),
]
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=5)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", by_epoch=True, max_keep_ckpts=2, interval=50
    ),
    logger=dict(type="LoggerHook", interval=5),
)
