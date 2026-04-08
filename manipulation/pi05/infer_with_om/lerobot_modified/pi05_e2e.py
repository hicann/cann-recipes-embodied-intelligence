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

import random
import logging
logging.basicConfig(level=logging.INFO)
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import acl
import acllite_utils as utils
import constants as const
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import resource_list


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"
    

@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple

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
        self.run_mode = None

    def init(self):
        logging.info("[INFO] Initializing ACL...")
        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)
        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)
        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)
        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)
        logging.info("[INFO] ACL Initialized.")

    def __del__(self):
        logging.info("[INFO] Releasing ACL...")
        resource_list.destroy()
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        logging.info("[INFO] ACL Released.")


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


class PI05Ascend:
    def __init__(
        self, 
        paligemma_model_path: str, 
        gemma_model_path: str, 
        config, dataset_stats: dict[str, Tensor] | None = None
    ):
        self.paligemma_model_path = paligemma_model_path
        self.gemma_model_path = gemma_model_path
        self.model = PI05Helper(config, dataset_stats)
        self.config = config
        self.dataset_stats = dataset_stats
        self._dvpp = None
        self._action_queue = deque()
        
        self.num_images = 3
        self.image_tokens_per_img = 256
        

    def init(self):
        logging.info("Init PI05Ascend Model.")

        self._dvpp = AclLiteImageProc()

        self.model.init(self.paligemma_model_path, self.gemma_model_path)

        self.num_images = 3 # need to match the number of images used in training
        self.image_tokens_per_img = 256
        
        return const.SUCCESS
        
    def reset(self):
        self._action_queue.clear()

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        # device = next(self.parameters()).device
        device = torch.device(type='cpu')

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: 
            # Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: 
            # Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks
    
    def select_action(self, state, image1, image2, tokens, masks):
        single_time_cost = time.time()

        if len(self._action_queue) == 0:  
            actions = self.predict_action_chunk(state, image1, image2, tokens, masks)[:, : self.config.n_action_steps]
            # logging.info(f"Actions shape: {actions.shape}")
            # return actions
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
      
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
            start_idx = 2 * self.image_tokens_per_img  # 512
            end_idx = 3 * self.image_tokens_per_img    # 768
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
        
        tokens = tokens.numpy().astype(np.int64)
        masks = masks.numpy().astype(np.bool_)
        
        image1 = image1.to(torch.float32).numpy()
        image2 = image2.to(torch.float32).numpy()
        
        actions = self.model.sample_actions(
            state, image1, image2, tokens, masks, prefix_att_2d_masks_4d
        )
        
        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]
        
        return actions

   

class PI05Helper(nn.Module):
    """
    Core model for the pi05 model inference
    """
    def __init__(self, config, dataset_stats: dict[str, Tensor] | None = None, *args, **kwargs):
        self.config = config
        self.dataset_stats = dataset_stats
        super().__init__()

    def init(self, paligemma_model_path: str, gemma_model_path: str):
        logging.info("Loading PI05 models...")
        self.paligemma_model = AclLiteModel(paligemma_model_path)
        self.gemma_model = AclLiteModel(gemma_model_path)


    def sample_noise(self, shape, device):
        # return torch.normal(
        #     mean=0.0,
        #     std=1.0,
        #     size=shape,
        #     dtype=torch.float32,
        #     device=device,
        # )
        return torch.zeros(
            size=shape,
            dtype=torch.float32,
            device=device,
        )
    
    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(self, state, image1, image2, tokens, masks, prefix_att_2d_masks_4d) -> Tensor:
        num_steps = self.config.num_inference_steps
        device = torch.device('cpu')

        actions_shape = (1, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device).numpy().astype(np.float16)
        
        prefix_att_masks = np.zeros((1, 968), dtype=np.bool_) 
        prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.numpy().astype(np.float32)
        
        input_list = [state, image1, image2, tokens, masks, prefix_att_masks, prefix_att_2d_masks_4d]
        output = self.paligemma_model.execute(input_list)
        output[0] = output[0].astype(np.float16)  # past_kv
        output[1] = output[1].astype(np.bool_)    # prefix_pad_masks_out
        time_tensor = np.array([1.0], dtype=np.float16)
        input_list2 = [output[0], output[1], time_tensor, noise]


        for _ in range(self.config.num_inference_steps):
            output2 = self.gemma_model.execute(input_list2)
            time_tensor -= np.array([0.1], dtype=np.float16)
            input_list2 = [output[0], output[1], time_tensor, output2[0]]

        actions = torch.from_numpy(output2[0]).unsqueeze(1).to(dtype=torch.float32)
        return actions[0]
   

def model_init(model1_path, model2_path, config):
    logging.info("Init PI05 Model.")
    pi05_model = PI05Ascend(model1_path, model2_path, config, dataset_stats=None)
    ret = pi05_model.init()
    utils.check_ret("PI05Ascend.init", ret)
    return pi05_model


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
    paligemma_model_path = "output/om_models/pi05/pi05-part1.om"
    gemma_model_path = "output/om_models/pi05/pi05-part2_linux_x86_64.om"

    pi05 = model_init(paligemma_model_path, gemma_model_path, config)

    data = torch.load("input_data/start_obs_0.pt", map_location="cpu")
    
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
    logging.info(f"Action: {action}")

    output_path = f"om_baseline_action_0.pt"
    torch.save(action, output_path)
    logging.info(f"Saved OM action to {output_path}")

if __name__ == "__main__":
    main()