#!/usr/bin/env python3
# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 Shanghai Jiao Tong University
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import logging

logging.basicConfig(level=logging.INFO)

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Compare file1 and file2 tensors")
    parser.add_argument("--file1", required=True, help="Path to file1 tensor file")
    parser.add_argument("--file2", required=True, help="Path to file2 tensor file")
    return parser.parse_args()


def main():
    args = parse_args()
    file1_path = args.file1
    file2_path = args.file2

    logging.info(f"file1: {file1_path}")
    logging.info(f"file2: {file2_path}")

    file2_tensor = torch.load(file2_path, map_location="cpu", weights_only=False)
    file1_tensor = torch.load(file1_path, map_location="cpu", weights_only=False)

    # 转为 float32，防止类型不一致
    file1_tensor = file1_tensor.float()
    file2_tensor = file2_tensor.float()

    logging.info(f"{file1_tensor.shape}, {file2_tensor.shape}")

    # torch.set_printoptions(threshold=float('inf'))
    logging.info(f"file1_tensor: {file1_tensor}")
    logging.info(f"file2 tensor: {file2_tensor}")

    # 检查形状一致
    if file1_tensor.shape != file2_tensor.shape:
        raise ValueError(f"Shape mismatch: file1={file1_tensor.shape}, file2={file2_tensor.shape}")

    # 计算绝对和相对差异
    abs_diff = torch.abs(file2_tensor - file1_tensor)
    denominator = torch.where(file2_tensor == 0, torch.ones_like(file2_tensor), torch.abs(file2_tensor))
    relative_diff = abs_diff / denominator
    relative_percent = relative_diff * 100

    # 计算统计量
    max_abs_diff = abs_diff.max()
    max_abs_idx_flat = torch.argmax(abs_diff)
    max_abs_file2 = file2_tensor.reshape(-1)[max_abs_idx_flat].item()
    max_abs_file1 = file1_tensor.reshape(-1)[max_abs_idx_flat].item()

    max_rel_diff = relative_percent.max()
    max_rel_idx_flat = torch.argmax(relative_percent)
    max_rel_file2 = file2_tensor.reshape(-1)[max_rel_idx_flat].item()
    max_rel_file1 = file1_tensor.reshape(-1)[max_rel_idx_flat].item()

    mean_abs_diff = abs_diff.mean().item()
    mean_rel_percent = relative_percent.mean().item()

    # ========== 新增：每行向量的余弦相似度 ==========
    if file1_tensor.ndim < 2:
        logging.warning("\n警告: 张量维度 < 2，无法计算逐行余弦相似度。")
    else:
        cos_sim = F.cosine_similarity(file1_tensor, file2_tensor, dim=-1, eps=1e-8)
        mean_cos = cos_sim.mean().item()
        min_cos = cos_sim.min().item()
        max_cos = cos_sim.max().item()
        cos_std = cos_sim.std().item()

    # 打印结果
    logging.info("\n===== 差异统计结果 =====")
    logging.info(f"张量形状: {file2_tensor.shape}")
    logging.info(f"绝对差异最大值: {max_abs_diff.item():.6e}")
    logging.info(f"  对应的 file2 值: {max_abs_file2:.6e}")
    logging.info(f"  对应的 file1 值: {max_abs_file1:.6e}")
    logging.info(f"绝对差异均值: {mean_abs_diff:.6e}")
    logging.info(f"相对差异最大百分比: {max_rel_diff.item():.6f}%")
    logging.info(f"  对应的 file2 值: {max_rel_file2:.6e}")
    logging.info(f"  对应的 file1 值: {max_rel_file1:.6e}")
    logging.info(f"相对差异平均百分比: {mean_rel_percent:.6f}%")

    # 余弦相似度结果
    if file1_tensor.ndim >= 2:
        logging.info("\n===== 余弦相似度统计 =====")
        logging.info(f"平均余弦相似度: {mean_cos:.6f}")
        logging.info(f"最小余弦相似度: {min_cos:.6f}")
        logging.info(f"最大余弦相似度: {max_cos:.6f}")
        logging.info(f"标准差: {cos_std:.6f}")

    # 差异分布
    logging.info("\n差异分布统计 (abs diff):")
    for p in [50, 90, 99, 99.9]:
        val = torch.quantile(abs_diff, p / 100.0).item()
        logging.info(f"  {p:>5.1f} 百分位: {val:.6e}")


if __name__ == "__main__":
    main()
