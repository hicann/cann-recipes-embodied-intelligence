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
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import acl
import acllite_utils as utils
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import resource_list

logging.basicConfig(level=logging.INFO)


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple


@dataclass
class PI05SampleInputs:
    state: torch.Tensor
    image1: torch.Tensor
    image2: torch.Tensor
    tokens: np.ndarray
    masks: np.ndarray
    prefix_att_2d_masks_4d: torch.Tensor

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f"[INFO] 随机种子已固定为: {seed}")


class AclLiteResource:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None

    def init(self):
        logging.info("[INFO] Initializing ACL...")
        acl.init()
        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)
        self.context, ret = acl.rt.create_context(self.device_id)
        self.stream, ret = acl.rt.create_stream()
        logging.info("[INFO] ACL Initialized.")

    def __del__(self):
        logging.info("[INFO] Releasing ACL...")
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        logging.info("[INFO] ACL Released.")


class PI05Ascend:
    def __init__(self, paligemma_model_path: str, gemma_model_path: str, config):
        self.paligemma_model_path = paligemma_model_path
        self.gemma_model_path = gemma_model_path
        self.model = PI05Helper(config)
        self.config = config
        self.num_images = 3
        self.image_tokens_per_img = 256

    def init(self):
        logging.info("Init PI05Ascend Model.")
        self.model.init(self.paligemma_model_path, self.gemma_model_path)
        return 0

    def select_action(self, state, image1, image2, tokens, masks):
        # 推理出一个 Chunk 的动作
        actions = self.predict_action_chunk(state, image1, image2, tokens, masks)[:, : self.config.n_action_steps]
        # 截断维度
        return actions
      
    def predict_action_chunk(self, state, image1, image2, tokens, masks):
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.bool)

        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if masks.ndim == 1:
            masks = masks.unsqueeze(0)

        prefix_att_masks = torch.zeros((1, 968), dtype=torch.bool)

        def _prepare_attention_masks_4d(prefix_att_masks):
            image_masks = torch.full(
                (1, self.num_images * self.image_tokens_per_img),
                1,
                dtype=masks.dtype,
                device=masks.device
            )
            start_idx = 2 * self.image_tokens_per_img
            end_idx = 3 * self.image_tokens_per_img
            image_masks[:, start_idx:end_idx] = 0

            prefix_pad_masks = torch.cat([image_masks, masks], dim=1)
            
            def make_att_2d_masks(pad_masks, att_masks): 
                if att_masks.ndim != 2:
                    raise ValueError(att_masks.ndim)
                if pad_masks.ndim != 2:
                    raise ValueError(pad_masks.ndim)

                att_masks = att_masks.to(torch.int32)
                cumsum = torch.cumsum(att_masks, dim=1)
                cumsum = cumsum.to(torch.bool)
                att_masks = att_masks.to(torch.bool)
                att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
                pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
                return att_2d_masks & pad_2d_masks
            
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_att_2d_masks = prefix_att_2d_masks[:, None, :, :]
            prefix_att_2d_masks = prefix_att_2d_masks.to(dtype=torch.bool)
            return torch.where(prefix_att_2d_masks, 0.0, OPENPI_ATTENTION_MASK_VALUE)

        prefix_att_2d_masks_4d = _prepare_attention_masks_4d(prefix_att_masks)
        
        sample_inputs = PI05SampleInputs(
            state=state,
            image1=image1,
            image2=image2,
            tokens=tokens.numpy(),
            masks=masks.numpy(),
            prefix_att_2d_masks_4d=prefix_att_2d_masks_4d,
        )

        actions = self.model.sample_actions(sample_inputs)
        
        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions


class PI05Helper(nn.Module):
    def __init__(self, config):
        self.config = config
        super().__init__()

    def init(self, paligemma_model_path: str, gemma_model_path: str):
        logging.info("Loading PI05 OM models...")
        self.paligemma_model = AclLiteModel(paligemma_model_path)
        self.gemma_model = AclLiteModel(gemma_model_path)

    def sample_noise(self, shape, device):
        return torch.zeros(shape, dtype=torch.float32, device=device)
    
    @torch.no_grad()
    def sample_actions(self, sample_inputs: PI05SampleInputs) -> Tensor:
        num_steps = self.config.num_inference_steps
        device = torch.device('cpu')

        actions_shape = (1, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device).numpy().astype(np.float16)

        state = sample_inputs.state.numpy().astype(np.float32)
        image1 = sample_inputs.image1.to(torch.float32).numpy()
        image2 = sample_inputs.image2.to(torch.float32).numpy()
        tokens = sample_inputs.tokens
        masks = sample_inputs.masks
        
        prefix_att_masks = np.zeros((1, 968), dtype=np.bool_) 
        prefix_att_2d_masks_4d = sample_inputs.prefix_att_2d_masks_4d.numpy().astype(np.float32)
        
        input_list = [state, image1, image2, tokens, masks, prefix_att_masks, prefix_att_2d_masks_4d]
        output = self.paligemma_model.execute(input_list)
        output[0] = output[0].astype(np.float16)  # past_kv
        output[1] = output[1].astype(np.bool_)    # prefix_pad_masks_out
        time_tensor = np.array([1.0], dtype=np.float16)
        input_list2 = [output[0], output[1], time_tensor, noise]

        torch.save(torch.from_numpy(input_list2[0]), "past_kv_om.pt")

        for _ in range(self.config.num_inference_steps):
            output2 = self.gemma_model.execute(input_list2)
            time_tensor -= np.array([0.1], dtype=np.float16)
            input_list2 = [output[0], output[1], time_tensor, output2[0]]

        actions = torch.from_numpy(output2[0]).unsqueeze(1).to(dtype=torch.float32)
        return actions[0]


def main():
    set_seed(42)
    
    acl_resource = AclLiteResource()
    acl_resource.init()

    config = SimpleNamespace(
        n_obs_steps=1,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.empty_camera_0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        image_features=["observation.images.image", "observation.images.image2", "observation.images.empty_camera_0"],
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        device="cpu",
        use_amp=False,
        push_to_hub=True,
        repo_id="pepijn223/pi05_libero_31_9_non_q",
        private=None,
        tags=None,
        license=None,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16",
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=32,
        max_action_dim=32,
        num_inference_steps=10,
        time_sampling_beta_alpha=1.5,
        time_sampling_beta_beta=1.0,
        time_sampling_scale=0.999,
        time_sampling_offset=0.001,
        min_period=0.004,
        max_period=4.0,
        image_resolution=(224, 224),
        empty_cameras=1,
        tokenizer_max_length=200,
        normalization_mapping={
            "ACTION": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
        },
        gradient_checkpointing=True,
        compile_model=False,
        compile_mode="max-autotune",
        optimizer_lr=2.5e-05,
        optimizer_betas=(0.9, 0.95),
        optimizer_eps=1e-08,
        optimizer_weight_decay=0.01,
        optimizer_grad_clip_norm=1.0,
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=6000,
        scheduler_decay_lr=2.5e-06,
    )
    
    # 请确认你的 OM 模型路径
    paligemma_model_path = "output/om_models/pi05/pi05-part1.om"
    gemma_model_path = "output/om_models/pi05/pi05-part2_linux_x86_64.om"
    
    pi05 = PI05Ascend(paligemma_model_path, gemma_model_path, config)
    pi05.init()

    task_id = 0
    logging.info("Loading observation tensors...")
    # 加载与 PyTorch 测试一样的输入数据
    data = torch.load(f"input_data/start_obs_0.pt", map_location="cpu") 
    
    image1 = data['observation.images.image']
    image2 = data['observation.images.image2']
    state = data['observation.state']
    tokens = data['observation.language.tokens']
    masks = data['observation.language.attention_mask']

    logging.info("Running Ascend OM inference...")
    start_time = time.time()
    
    action = pi05.select_action(state, image1, image2, tokens, masks)
    
    elapsed = time.time() - start_time
    logging.info(f"\n✅ Inference completed in {elapsed:.4f} seconds")
    logging.info(f"Action shape: {action.shape}")
    logging.info(f"Action : {action}")
    
    output_path = f"om_baseline_action.pt"
    torch.save(action, output_path)
    logging.info(f"Saved OM action to {output_path}")

if __name__ == "__main__":
    main()