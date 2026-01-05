# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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
from collections import deque
import logging
import os
import time

import numpy as np
import torch
from torch import Tensor

import acl
import acllite_utils as utils
import constants as const
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import resource_list
from lerobot.policies.normalize import Unnormalize
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

logger = logging.getLogger(__name__)


class AclLiteResource:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None

    def init(self):
        logger.info("Initializing ACL...")
        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)
        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)
        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)
        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)
        logger.info("ACL Initialized.")

    def __del__(self):
        logger.info("Releasing ACL...")
        resource_list.destroy()
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        logger.info("ACL Released.")


class Pi0(object):
    """
    class for Pi0 model
    """

    def __init__(
        self,
        vlm_model_path,
        action_expert_model_path,
        config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        self._vlm_model_path = vlm_model_path
        self._action_expert_model_path = action_expert_model_path
        self._action_expert_model = None
        self._vlm_model = None
        self._dvpp = None
        self._action_queue = deque()

        # Accept config as dict or object with attributes.
        def _get(cfg, key):
            return cfg[key] if isinstance(cfg, dict) else getattr(cfg, key)

        features = _get(config, "output_features")
        norm_map = _get(config, "normalization_mapping")
        stats = dataset_stats if dataset_stats is not None else _get(config, "stats")
        if stats is None:
            raise ValueError("Unnormalize stats are required; provide mean/std via CLI or config")

        self.unnormalize_outputs = Unnormalize(features, norm_map, stats)

    def init(self):
        """
        init Pi0 model
        """
        # 初始化dvpp
        self._dvpp = AclLiteImageProc()

        logger.info("Load Paligemma model")
        self._vlm_model = AclLiteModel(self._vlm_model_path)

        logger.info("Load Gemma model")
        self._action_expert_model = AclLiteModel(self._action_expert_model_path)

        return const.SUCCESS

    def interface(self, state, image, lang_tokens, lang_masks=None):
        """
        According to the input , generate the output;
         1. get the lang token and mask.
         2. use peligemma to get the kv_tensor, prefix pad mask
         3. use gemma to get the final output
        """
        if lang_tokens is None:
            raise ValueError("lang_tokens must be provided for language inputs")

        if isinstance(lang_tokens, np.ndarray):
            lang_tokens_t = torch.from_numpy(lang_tokens)
        else:
            lang_tokens_t = lang_tokens

        if lang_masks is None:
            lang_masks_t = (lang_tokens_t != 0)
        else:
            lang_masks_t = lang_masks if isinstance(lang_masks, torch.Tensor) else torch.from_numpy(lang_masks)

        lang_tokens_np = lang_tokens_t.cpu().numpy().astype(np.int64)
        lang_masks_np = lang_masks_t.cpu().numpy().astype(np.bool_)

        part1_input_list = [state.astype(np.float32), image.astype(np.float32), lang_tokens_np, lang_masks_np]
        # measure the time cost of part1
        start = time.time()
        # for i in range(10):
        part1_output = self._vlm_model.execute(input_list=part1_input_list)
        # print("Part1 output", part1_output)
        logger.info("[TIMER] VLM interface time: %.3f s", (time.time() - start) / 10)

        kv_tensor = part1_output[0]
        prefix_pad_masks = part1_output[1]
        dummy_time = np.array([1.0], dtype=np.float16)
        action_shape = (1, 50, 32)
        dummy_noise = np.zeros(action_shape, dtype=np.float16)

        state = state.astype(np.float16)
        if isinstance(prefix_pad_masks, torch.Tensor):
            prefix_pad_masks = prefix_pad_masks.detach().cpu().numpy()
        prefix_pad_masks = prefix_pad_masks.astype(np.bool_)

        kv_tensor = kv_tensor.astype(np.float16)
        part2_input_list = [
            state,
            lang_tokens_np,
            lang_masks_np,
            kv_tensor,
            prefix_pad_masks,
            dummy_time,
            dummy_noise,
        ]
        output = []
        start = time.time()
        for _i in range(10):
            output = self._action_expert_model.execute(input_list=part2_input_list)
            dummy_time -= np.array([0.1], dtype=np.float16)

            part2_input_list = [
                state,
                lang_tokens_np,
                lang_masks_np,
                kv_tensor,
                prefix_pad_masks,
                dummy_time,
                output[0],
            ]

        actions = self._to_action_14(output[0], target_dim=14)
        tensor_actions = torch.from_numpy(actions)
        final_actions = self.unnormalize_outputs({"action": tensor_actions})["action"]
        return final_actions

    def select_action(self, state, image, lang_tokens, lang_masks=None):
        """
        select action according to the state and image, store in action queue
        """
        if not self._action_queue:
            logger.info("Generate new action sequence")
            action_array = self.interface(state, image, lang_tokens, lang_masks)
            for i in range(action_array[0].shape[0]):
                self._action_queue.append(action_array[0][i])
        return self._action_queue.popleft()

    def _to_action_14(self, actions: np.ndarray, target_dim: int = 14, indices: list[int] | None = None) -> np.ndarray:
        """
        将动作序列从 (..., D) 转成 (..., 14)。
        接受 list/np.ndarray；支持 (1,50,32) 或 (50,32) 等，统一返回 (1,50,14)。
        """
        arr = np.asarray(actions)
        if arr.ndim == 3:
            batch_size, time_steps, feature_dim = arr.shape
        elif arr.ndim == 2:
            # (T, D) -> (1, T, D)
            time_steps, feature_dim = arr.shape
            batch_size = 1
            arr = arr[None, ...]
        else:
            raise ValueError(f"Expect actions of shape (B,T,D) or (T,D), got {arr.shape}")

        if indices is not None:
            idx = np.asarray(indices, dtype=int)
            if idx.shape[0] != target_dim:
                raise ValueError(f"indices length {idx.shape[0]} != {target_dim}")
            out = arr[:, :, idx]
        else:
            if feature_dim < target_dim:
                raise ValueError(f"Action dim {feature_dim} < target {target_dim}")
            out = arr[:, :, :target_dim]

        return np.ascontiguousarray(out)


def model_init(vlm_model_path, action_expert_model_path, config):
    """
    init pi0 resource and act model
    """

    logger.info("Load Pi0 model")

    pi0 = Pi0(vlm_model_path, action_expert_model_path, config)
    ret = pi0.init()
    utils.check_ret("Pi0 init", ret)
    return pi0


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run pi0 OM inference end-to-end")
    parser.add_argument("--vlm-model-path", default="pi0_vlm.om", help="Path to part1 OM file (PaliGemma)")
    parser.add_argument(
        "--action-expert-model-path",
        default="pi0_action_expert.om",
        help="Path to part2 OM file (Gemma)",
    )
    parser.add_argument("--mean-path", default=None, help="Path to action mean .pt file")
    parser.add_argument("--std-path", default=None, help="Path to action std .pt file")
    args = parser.parse_args()

    acl_resource = AclLiteResource()
    acl_resource.init()

    config = {
        'output_features': {
            'action': PolicyFeature(
                type=FeatureType.ACTION,
                shape=(14,))
        },
        'normalization_mapping': {
            'VISUAL': NormalizationMode.IDENTITY,
            'STATE': NormalizationMode.MEAN_STD,
            'ACTION': NormalizationMode.MEAN_STD,
        },
        'stats': None,
    }

    # Override stats with user-provided mean/std tensors when both paths are given.
    if args.mean_path or args.std_path:
        if not (args.mean_path and args.std_path):
            raise ValueError("Both --mean-path and --std-path must be provided together.")
        if not os.path.exists(args.mean_path):
            raise FileNotFoundError(f"Mean file not found: {args.mean_path}")
        if not os.path.exists(args.std_path):
            raise FileNotFoundError(f"Std file not found: {args.std_path}")

        loaded_mean = torch.load(args.mean_path)
        loaded_std = torch.load(args.std_path)
        config['stats'] = {
            'action': {
                'mean': loaded_mean,
                'std': loaded_std,
            }
        }
    else:
        # Default action normalization stats if no paths are supplied.
        default_mean = torch.tensor([
            -0.0054, -0.4803, 1.0102, -0.0042, -0.5298, 1.1214, 0.5875,
            0.0196, -0.3138, 0.4702, -0.0231, 0.7722, 0.0375, 0.5962
        ], dtype=torch.float32)
        default_std = torch.tensor([
            0.0037, 0.5198, 0.1978, 0.0163, 0.3605, 0.5996, 0.4241,
            0.1111, 0.4944, 0.4435, 0.1452, 0.2956, 0.2278, 0.3861
        ], dtype=torch.float32)
        config['stats'] = {
            'action': {
                'mean': default_mean,
                'std': default_std,
            }
        }
    # add your model paths here
    # model init params: vlm_model_path, action_expert_model_path, config
    model = model_init(args.vlm_model_path, args.action_expert_model_path, config)
    # def interface(self, state, image, tokens_ids):
    state_shape = (1, 14)
    state = np.zeros(state_shape, dtype=np.float32)

    image_shape = (1, 3, 480, 640)
    image = np.zeros(image_shape, dtype=np.float32)
    vocab_size = 32000
    max_len = 48
    lang_tokens = torch.randint(low=1, high=vocab_size, size=(1, max_len), dtype=torch.long)
    lang_masks = torch.ones_like(lang_tokens, dtype=torch.bool)

    action = model.select_action(state, image, lang_tokens, lang_masks)
    logger.info("Selected action: %s", action)

if __name__ == "__main__":
    main()
