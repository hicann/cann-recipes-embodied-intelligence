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

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""
import os
import sys
import json
import logging
import threading
import time
import einops
import gymnasium as gym
import numpy as np
import torch
import acl
import logging
import acllite_utils as utils
import constants as const

from collections.abc import Callable
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from termcolor import colored
from torch import Tensor, nnrvs()
from tqdm import trange
from scipy import stats
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import resource_list

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.normalize import Unnormalize
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

logger = logging.getLogger(__name__)

_CLI_MEAN_PATH: str | None = None
_CLI_STD_PATH: str | None = None
_CLI_VLM_PATH: str | None = None
_CLI_ACTION_PATH: str | None = None


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
        logger.info("ACL initialized.")

    def __del__(self):
        logger.info("Releasing ACL...")
        resource_list.destroy()
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        logger.info("ACL released.")

TIME_SUM = 0.0
INFER_TIMES = 0


def _preparse_mean_std_paths_from_argv() -> None:
    """Extract CLI paths (mean/std, vlm/action) from sys.argv before Draccus parsing."""

    global _CLI_MEAN_PATH, _CLI_STD_PATH, _CLI_VLM_PATH, _CLI_ACTION_PATH

    argv = list(sys.argv)
    keep: list[str] = [argv[0]]
    i = 1
    keys_mean = {"--mean-path", "--mean_path"}
    keys_std = {"--std-path", "--std_path"}
    keys_vlm = {"--vlm-model-path", "--vlm_model_path"}
    keys_action = {"--action-expert-model-path", "--action_expert_model_path"}

    def _maybe_capture(key_set: set[str], current: str, next_val: str | None) -> tuple[bool, str | None]:
        if current in key_set:
            return True, next_val
        for key in key_set:
            prefix = f"{key}="
            if current.startswith(prefix):
                return True, current.split("=", 1)[1]
        return False, None

    while i < len(argv):
        arg = argv[i]
        handled = False

        handled, val = _maybe_capture(keys_mean, arg, argv[i + 1] if i + 1 < len(argv) else None)
        if handled:
            if val is None:
                raise ValueError("--mean-path requires a value")
            _CLI_MEAN_PATH = val
            if not arg.startswith("--mean-path=") and not arg.startswith("--mean_path="):
                i += 2
            else:
                i += 1
            continue

        handled, val = _maybe_capture(keys_std, arg, argv[i + 1] if i + 1 < len(argv) else None)
        if handled:
            if val is None:
                raise ValueError("--std-path requires a value")
            _CLI_STD_PATH = val
            if not arg.startswith("--std-path=") and not arg.startswith("--std_path="):
                i += 2
            else:
                i += 1
            continue

        handled, val = _maybe_capture(keys_vlm, arg, argv[i + 1] if i + 1 < len(argv) else None)
        if handled:
            if val is None:
                raise ValueError("--vlm-model-path requires a value")
            _CLI_VLM_PATH = val
            if not arg.startswith("--vlm-model-path=") and not arg.startswith("--vlm_model_path="):
                i += 2
            else:
                i += 1
            continue

        handled, val = _maybe_capture(keys_action, arg, argv[i + 1] if i + 1 < len(argv) else None)
        if handled:
            if val is None:
                raise ValueError("--action-expert-model-path requires a value")
            _CLI_ACTION_PATH = val
            if not arg.startswith("--action-expert-model-path=") and not arg.startswith("--action_expert_model_path="):
                i += 2
            else:
                i += 1
            continue

        keep.append(arg)
        i += 1

    sys.argv = keep


class Pi0(object):
    """
    class for Pi0 model
    """
    def __init__(self, vlm_model_path, action_expert_model_path, 
                config, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
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
        pi0_stats = dataset_stats if dataset_stats is not None else _get(config, "stats")
        if pi0_stats is None:
            raise ValueError("Unnormalize stats are required; provide mean/std via CLI or config")

        self.unnormalize_outputs = Unnormalize(features, norm_map, pi0_stats)

    def init(self):
        """
        init Pi0 model
        """
        # 初始化dvpp
        self._dvpp = AclLiteImageProc()

        logger.info("Load VLM model")
        self._vlm_model = AclLiteModel(self._vlm_model_path)

        logger.info("Load Action Expert model")
        self._action_expert_model = AclLiteModel(self._action_expert_model_path)

        return const.SUCCESS

    def reset(self):
        """
        reset the model status
        """
        self._action_queue.clear()
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
        part1_output = self._vlm_model.execute(input_list=part1_input_list)
        logger.info("[TIMER] VLM interface time: %.3f s", time.time() - start)

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
        logger.info("[TIMER] Action Expert interface time: %.3f s", (time.time() - start) / 10)
        actions = self._to_action_14(output[0], target_dim=14)
        tensor_actions = torch.from_numpy(actions)
        final_actions = self.unnormalize_outputs({"action": tensor_actions})["action"]
        return final_actions

    def select_action(self, state, image, tokens_ids):
        """
        select action according to the state and image, store in action queue
        """

        def build_lang_inputs(token_ids, max_len=48, pad_token_id=0):
            # token_ids: List[int]，原始的 token 序列
            # max_len: 固定的序列长度
            # pad_token_id: padding 用的 id，默认是 0

            # 1. 截断或者补齐
            tokens = token_ids[:max_len]
            if len(tokens) < max_len:
                tokens += [pad_token_id] * (max_len - len(tokens))

            # 2. 转 tensor
            # lang_tokens = torch.tensor([tokens], dtype=torch.long, device="cuda:0")   # shape: [1, max_len]
            lang_tokens = torch.tensor([tokens], dtype=torch.long, device="cpu")  # shape: [1, max_len]

            # 3. 生成 mask
            lang_masks = (lang_tokens != pad_token_id)               # True 表示有效 token

            return lang_tokens, lang_masks

        def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
            token_ids = [2, 23262, 235298, 35368, 108]  # 这里的token_ids是aloha trans cube任务的token_ids
            lang_tokens, lang_masks = build_lang_inputs(token_ids, max_len=48)
            return lang_tokens, lang_masks

        lang_tokens, lang_masks = prepare_language(self, 1)
        if not self._action_queue:
            logger.info("Generate new action sequence")
            action_array = self.interface(state, image, lang_tokens, lang_masks)
            for i in range(action_array[0].shape[0]):
                self._action_queue.append(action_array[0][i])
            # self._action_queue.append(action_array[0][1])
        return self._action_queue.popleft()

    @staticmethod
    def build_lang_inputs(token_ids, max_len=48, pad_token_id=0):
        # token_ids: List[int]，原始的 token 序列
        # max_len: 固定的序列长度
        # pad_token_id: padding 用的 id，默认是 0
        # 1. 截断或者补齐
        tokens = token_ids[:max_len]
        if len(tokens) < max_len:
            tokens += [pad_token_id] * (max_len - len(tokens))
        # 2. 转 tensor
        lang_tokens = torch.tensor([tokens], dtype=torch.long)   # shape: [1, max_len]
        # 3. 生成 mask
        lang_masks = (lang_tokens != pad_token_id)               # True 表示有效 token
        return lang_tokens, lang_masks

    def _cqw_prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """
        准备语言输入
        此处任务为固定任务：Transfer the red cube
        所以直接返回固定的 token_ids
        并且获得对应的lang_tokens 和 lang_masks
        """
        token_ids = [2, 23262, 235298, 35368, 108]  # 模拟 tokenizer 输出
        lang_tokens, lang_masks =self.build_lang_inputs(token_ids, max_len=48)
        return lang_tokens, lang_masks

    def _to_action_14(self, actions: np.ndarray, target_dim: int = 14, indices: list[int] | None = None) -> np.ndarray:
        """
        将动作序列从 (..., D) 转成 (..., 14)。
        接受 list/np.ndarray；支持 (1,50,32) 或 (50,32) 等，统一返回 (1,50,14)。
        """
        arr = np.asarray(actions)
        if arr.ndim == 3:
            B, T, D = arr.shape
        elif arr.ndim == 2:
            # (T, D) -> (1, T, D)
            T, D = arr.shape
            B = 1
            arr = arr[None, ...]
        else:
            raise ValueError(f"Expect actions of shape (B,T,D) or (T,D), got {arr.shape}")

        if indices is not None:
            idx = np.asarray(indices, dtype=int)
            assert idx.shape[0] == target_dim, f"indices length {idx.shape[0]} != {target_dim}"
            out = arr[:, :, idx]
        else:
            if D < target_dim:
                raise ValueError(f"Action dim {D} < target {target_dim}")
            out = arr[:, :, :target_dim]

        return np.ascontiguousarray(out)



def model_init(model1_path, model2_path, config):
    """
    init pi0 resource and act model
    """

    pi0 = Pi0(model1_path, model2_path, config)
    ret = pi0.init()
    utils.check_ret("Pi0 init", ret)
    return pi0


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    model: Pi0 | None = None
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done):
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        img_array = observation["observation.images.top"].cpu().numpy()
        state_array = observation["observation.state"].cpu().numpy()

        action = model.select_action(state_array, img_array, None)
        action = np.expand_dims(action, axis=0)

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
        if "final_info" in info:
            successes = [info["is_success"] if info is not None else False for info in info["final_info"]]
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        done = terminated | truncated | done

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    model: Pi0 | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        model.reset()
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env,
            policy,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            model=model
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data["action"].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            "action": rollout_data["action"][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            "next.done": rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            "next.reward": rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data["observation"]:
            ep_dict[key] = rollout_data["observation"][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    # initialize aclite resources in eval main
    acl_resource = AclLiteResource()
    acl_resource.init()

    # 构建与 ref.py 一致的 Unnormalize 配置，并内置默认 mean/std
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

    # 当未提供数据集统计信息时，沿用 ref.py 的默认动作分布参数
    default_mean = torch.tensor([
        -0.0054, -0.4803, 1.0102, -0.0042, -0.5298, 1.1214, 0.5875,
        0.0196, -0.3138, 0.4702, -0.0231, 0.7722, 0.0375, 0.5962
    ], dtype=torch.float32)
    default_std = torch.tensor([
        0.0037, 0.5198, 0.1978, 0.0163, 0.3605, 0.5996, 0.4241,
        0.1111, 0.4944, 0.4435, 0.1452, 0.2956, 0.2278, 0.3861
    ], dtype=torch.float32)

    if _CLI_MEAN_PATH or _CLI_STD_PATH:
        if not (_CLI_MEAN_PATH and _CLI_STD_PATH):
            raise ValueError("Both --mean-path and --std-path must be provided together")
        mean_path = Path(_CLI_MEAN_PATH).expanduser()
        std_path = Path(_CLI_STD_PATH).expanduser()
        if not mean_path.exists():
            raise FileNotFoundError(f"Mean file not found: {mean_path}")
        if not std_path.exists():
            raise FileNotFoundError(f"Std file not found: {std_path}")
        loaded_mean = torch.load(mean_path)
        loaded_std = torch.load(std_path)
        config['stats'] = {
            'action': {
                'mean': loaded_mean,
                'std': loaded_std,
            }
        }
        logger.info("Loaded unnormalize stats from CLI: mean=%s std=%s", mean_path, std_path)
    else:
        config['stats'] = {
            'action': {
                'mean': default_mean,
                'std': default_std,
            }
        }

    if not _CLI_VLM_PATH or not _CLI_ACTION_PATH:
        raise ValueError("You must provide --vlm-model-path and --action-expert-model-path")

    Pi0_vlm_path = Path(_CLI_VLM_PATH).expanduser()
    Pi0_action_expert_path = Path(_CLI_ACTION_PATH).expanduser()
    if not Pi0_vlm_path.exists():
        raise FileNotFoundError(f"VLM OM file not found: {Pi0_vlm_path}")
    if not Pi0_action_expert_path.exists():
        raise FileNotFoundError(f"Action expert OM file not found: {Pi0_action_expert_path}")

    model = model_init(str(Pi0_vlm_path), str(Pi0_action_expert_path), config)
    try:
        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
        )
    except FileNotFoundError as exc:
        logging.warning(
            "model.safetensors not found; constructing policy from scratch (weights not loaded). Error: %s",
            exc,
        )
        cfg.policy.pretrained_path = None
        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
        )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy(
            env,
            policy,
            cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            model=model
        )

    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    logging.info("End of eval")


def main():
    _preparse_mean_std_paths_from_argv()
    init_logging()
    eval_main()

if __name__ == "__main__":
    main()
