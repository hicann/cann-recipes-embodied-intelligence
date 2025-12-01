#!/bin/bash
# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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


# 检查传入参数数量
if [ $# -lt 2 ]; then
    echo "Usage: bash run_pi0_inference.sh [dataset] [checkpoint] [warmup] [iters] 
          [use_noise_from_file] [save_noise] [use_fixed_noise] [save_actions]"
    exit 1
fi

# 设置环境变量
export DEVICE=npu  # 用昇腾服务器时设置为 npu

# 读取传入变量
DATASET=$1            # 第一个参数：数据集名称
CHECKPOINT=$2         # 第二个参数：模型检查点
WARMUP=${3:-10}       # 第三个参数：预热次数，默认值为 10
ITERS=${4:-100}       # 第四个参数：迭代次数，默认值为 100
BATCHSIZE=${5:-1}     # 第五个参数：批处理大小，默认值为 1
EPISODESIDX=${6:-25}  # 第六个参数：测试第几个episode，默认值为 25
SAMPLEIDX=${7:-0}     # 第七个参数：每个episode中要获取的样本索引，默认值为 0

# 打印参数信息
echo "Running inference script with the following parameters:"
echo "Dataset: $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "Warmup: $WARMUP"
echo "Iterations: $ITERS"
echo "BatchSize: $BATCHSIZE"
echo "EpisodesIdx: $EPISODESIDX"
echo "SampleIdx: $SAMPLEIDX"

# 运行 Python 脚本
python test_pi0_on_ascend.py \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --batch_size "$BATCHSIZE" \
    --episodes_idx "$EPISODESIDX" \
    --target_sample_idx "$SAMPLEIDX"

# 检查脚本运行结果
if [ $? -eq 0 ]; then
    echo "脚本成功运行!"
else
    echo "脚本运行失败!"
fi
