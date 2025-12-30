#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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


# 标准库
import argparse
import logging
import os
import sys
from pathlib import Path

# 第三方库
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 应用程序自定义模块
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig

# 配置选项
torch.backends.cudnn.benchmark = True


logging.basicConfig(
    format="[%(levelname)s] %(asctime)s  %(filename)s:%(lineno)d  - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


# 添加一个自定义的布尔解析方法
def str2bool(v):
    if isinstance(v, str):
        return v.lower() in ('true', '1', 't', 'y', 'yes')
    return bool(v)


def parse_args():
    parser = argparse.ArgumentParser(description="Pi0 inference on Ascend")
    parser.add_argument("--dataset", default="koch_test", help="LeRobot dataset repo/id")
    parser.add_argument("--checkpoint", default="pi0_model", help="path to pi0 checkpoint")
    parser.add_argument("--device", default=os.getenv("DEVICE", "npu"), help="npu for Ascend")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="benchmark iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for inference")
    parser.add_argument("--episodes_idx", type=int, default=25, help="episode index to test")
    parser.add_argument("--target_sample_idx", type=int, default=0, help="target sample index in each episode")
    return parser.parse_args()


# 加载数据集并获取指定样本
def load_data(device, dataset, batch_size, target_sample_idx):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
    )

    # 获取当前 episode 的第 target_sample_idx 个样本
    try:
        # 创建一个迭代器并跳到第 target_sample_idx 个样本
        data_iter = iter(dataloader)
        for _ in range(target_sample_idx):
            next(data_iter)  # 跳过前 target_sample_idx-1 个样本
        batch = next(data_iter)  # 获取第 target_sample_idx 个样本
    except StopIteration:
        # 如果样本不足 target_sample_idx+1 个，取第一个样本作为默认值
        LOG.warning(f"Episode 的数据集中样本少于 {target_sample_idx + 1} 个，默认取第一个样本")
        try:
            batch = next(iter(dataloader))  # 取第一个样本
        except StopIteration:
            LOG.error("Episode 的数据集为空，无法获取样本")

    # To device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device, dtype=torch.float32)
    
    return batch


# 主函数，执行pi0模型的推理并保存结果
# Adapted from lerobot/lerobot/common/policies/pi0/conversion_scripts/benchmark.py
def main(args):
    device = torch.device(args.device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_repo_id = os.path.join(script_dir, args.dataset)

    dataset = LeRobotDataset(dataset_repo_id, episodes=[args.episodes_idx])
    
    # Load dataset
    batch_size = args.batch_size  # 批处理大小
    target_sample_idx = args.target_sample_idx  # 每个episode中要获取的样本索引
    batch = load_data(device, dataset, batch_size, target_sample_idx)

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)

    cfg.pretrained_path = args.checkpoint
    policy = make_policy(cfg, device, ds_meta=dataset.meta)  # 创建策略对象

    policy.model.update_qkv_weights()  # 更新 qkv 融合后的权重

    LOG.info(f"Starting warmup ({args.warmup} iters) ...")
    # Warmup
    for _ in range(args.warmup):
        torch.cuda.synchronize()
        policy.select_action(batch)  # 只输出轨迹的最后一组机械臂关节角度系列
        policy.reset()
        torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    LOG.info(f"Benchmark ({args.iters} iters) ...")

    # 创建一个列表来存储每次推理的50组动作数据
    actions_list = []
    for _ in range(args.iters):
        action = policy.select_action_all(batch)
        # 保存单次推理得到的50组关节角度序列到列表中
        actions_list.append(action.cpu().numpy())  # 转为 NumPy 数组 并存储在列表中
        policy.reset()

    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_per_iter = elapsed_time_ms / args.iters

    # 打印单次推理平均延迟
    LOG.info(f"Average latency: {avg_time_per_iter:.3f} ms")

    actions_np = np.array(actions_list)

    # 打印推理动作序列结果的形状，为(args.iters, 50, 1, joints_num)
    LOG.info(f"Shape of all action sequences: {actions_np.shape}")
    # 打印单次推理的动作序列作为示例
    LOG.info(f"Selected action sequences: {actions_np[0][0]}")


if __name__ == "__main__":
    with torch.inference_mode():
        main(parse_args())