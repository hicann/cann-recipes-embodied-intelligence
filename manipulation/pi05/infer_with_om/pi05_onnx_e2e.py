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

import logging
import os
import random
import time

import numpy as np
import onnxruntime as ort
import torch

logging.basicConfig(level=logging.INFO)

# ================= 配置区 =================
ONNX_PART1_PATH = "output/onnx_models/pi05/pi05-part1.onnx"
ONNX_PART2_PATH = "output/onnx_models/pi05/pi05-part2.onnx"
DEBUG_DATA_PATH = "output/debug_obs/start_obs_0.pt"

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38
NUM_IMAGES = 3
IMAGE_TOKENS_PER_IMG = 256
SEED = 42
# ==========================================


def set_seed(seed: int = 42):
    """
    固定所有可能的随机源
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"[INFO] 随机种子已固定为: {seed}")


def make_masks_logic(lang_masks, device="cpu"):
    """
    根据输入的语言掩码，生成模型需要的 prefix_att_masks 和 4D 掩码
    """
    batch_size = lang_masks.shape[0]
    prefix_att_masks = torch.zeros((batch_size, 968), dtype=torch.bool, device=device)

    image_masks = torch.full((batch_size, NUM_IMAGES * IMAGE_TOKENS_PER_IMG), 1, 
                            dtype=lang_masks.dtype, device=device)
    prefix_pad_masks = torch.cat([image_masks, lang_masks], dim=1) 
    
    att_masks_32 = prefix_att_masks.to(torch.int32)
    cumsum = torch.cumsum(att_masks_32, dim=1).to(torch.bool)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
    
    prefix_att_2d_masks = (att_2d_masks & pad_2d_masks)[:, None, :, :].to(torch.bool)
    prefix_att_2d_masks_4d = torch.where(prefix_att_2d_masks, 0.0, OPENPI_ATTENTION_MASK_VALUE)
    
    return prefix_att_masks, prefix_att_2d_masks_4d


def main():
    # 1. 固定随机种子
    set_seed(SEED)

    # 加载模型
    providers = ['CPUExecutionProvider']
    session1 = ort.InferenceSession(ONNX_PART1_PATH, providers=providers)
    session2 = ort.InferenceSession(ONNX_PART2_PATH, providers=providers)

    # 2. 加载原始数据
    data = torch.load(DEBUG_DATA_PATH, map_location="cpu")
    lang_masks = data['observation.language.attention_mask']

    # 3. 动态生成掩码变量
    logging.info("Generating attention masks...")
    p_att_masks, p_att_4d = make_masks_logic(lang_masks)

    # 4. 构建 Part 1 输入
    onnx_inputs_p1 = {
        "observation.state": data['observation.state'].numpy().astype(np.float32),
        "observation.images.image": data['observation.images.image'].numpy().astype(np.float32),
        "observation.images.image2": data['observation.images.image2'].numpy().astype(np.float32),
        "observation.language.tokens": data['observation.language.tokens'].numpy().astype(np.int64),
        "observation.language.attention_mask": lang_masks.numpy().astype(np.bool_),
        "prefix_att_masks": p_att_masks.numpy().astype(np.bool_),
        "prefix_att_2d_masks_4d": p_att_4d.numpy().astype(np.float32),
    }

    # 5. 推理 Part 1
    logging.info("Running ONNX Part 1...")
    outputs_p1 = session1.run(None, onnx_inputs_p1)
    past_kv = outputs_p1[0].astype(np.float16)
    prefix_pad_masks_out = outputs_p1[1].astype(np.bool_) 
    torch.save(torch.from_numpy(past_kv), "past_kv_onnx.pt")
    # 6. 推理 Part 2 (迭代去噪)
    num_steps = 10
    
    current_noise = np.zeros((1, 50, 32), dtype=np.float16)

    logging.info(f"Running ONNX Part 2 Denosing ({num_steps} steps)...")
    start_time = time.time()

    for i in range(num_steps):
        t_val = 1.0 - (i / num_steps)
        time_tensor = np.array([t_val], dtype=np.float16)
        
        onnx_inputs_p2 = {
            "past_kv_tensor": past_kv,
            "prefix_pad_masks": prefix_pad_masks_out.astype(np.bool_),
            "time": time_tensor,
            "noise": current_noise
        }
        
        outputs_p2 = session2.run(None, onnx_inputs_p2)
        current_noise = outputs_p2[0]

    elapsed = time.time() - start_time
    
    # 7. 结果保存
    final_action = current_noise[..., :7].astype(np.float32)
    logging.info(f"\n Final action values (first step):\n{final_action}")
    logging.info(f"\n✅ Done! Time: {elapsed:.4f}s. Action Shape: {final_action.shape}")
    
    torch.save(torch.from_numpy(final_action), "onnx_baseline_action.pt")

if __name__ == "__main__":
    main()