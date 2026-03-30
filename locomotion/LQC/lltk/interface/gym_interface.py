# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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

import time
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
import yaml


class GymWrapper(gym.Env):
    def __init__(self, impl, normalize_ob=False, debug=False):
        self._core = impl
        self._normalize_ob = normalize_ob
        self._debug = debug
        self._ob, self._ob_dim = None, None
        self._ob_mean, self._ob_std = None, None
        self._initialized = False
        self._reward_summary = defaultdict(lambda: 0)

    def _init(self):
        self._core.init()
        self._initialized = True
        self._ob_dim = self._core.getObDim()
        self._ob = np.zeros(self._ob_dim, dtype=np.float32)
        if self._normalize_ob:
            self._ob_mean = np.zeros(self._ob_dim, dtype=np.float32)
            self._ob_std = np.ones(self._ob_dim, dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, [self._core.getActionDim()])

    @property
    def observation_space(self):
        return gym.spaces.Box(-10, 10, [self._core.getObDim()])

    def step(self, action: np.ndarray):
        reward = self._core.step(action.squeeze())
        done = self._core.isTerminal()
        timeout = self._core.isTimeOut()
        ob = self._observe(self._normalize_ob)
        info = {}
        if self._debug:
            reward_info = self._core.getRewardInfo()
            for k, v in reward_info.items():
                self._reward_summary[k] += v
            info = {
                'timeout': timeout,
                'raw_ob': self._ob,
                'reward_info': reward_info,
            }
        return ob, reward, done, timeout, info

    def reset(self, *, seed=None, options=None):
        if not self._initialized:
            if seed is None:
                seed = time.time_ns()
            super().reset(seed=seed)
            self._core.seed(seed)
            self._init()
        if seed is not None:
            super().reset(seed=seed)
            self._core.seed(seed)
        self._core.reset()
        return self._observe(self._normalize_ob), {}

    def load_scaling(self, scaling_dict):
        if not self._normalize_ob:
            raise RuntimeError("Observation normalization is off")
        self._ob_mean = scaling_dict['mean']
        self._ob_std = np.sqrt(scaling_dict['var'] + 1e-8)

    def _observe(self, normalize=True):
        self._core.observe(self._ob)
        ob = self._ob.copy()
        if normalize:
            ob = (ob - self._ob_mean) / self._ob_std
        return ob

    def get_reward_dict(self):
        return self._reward_summary

    def __getattr__(self, item):
        return getattr(self._core, item)


class VGymWrapper(gym.experimental.VectorEnv):
    Single: Any
    single_obj: Any

    def __init__(self, venv_cls, cfg: dict):
        from lltk.task_registry import RSC_PATH

        self._core = venv_cls(RSC_PATH, yaml.safe_dump(cfg))
        self._core.reset()
        self._num_envs = self._core.getNumOfEnvs()
        self._ob_dim = self._core.getObDim()
        self._ac_dim = self._core.getActionDim()
        self._observation = np.zeros([self.num_envs, self._ob_dim], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._timeout = np.zeros(self.num_envs, dtype=bool)
        self._ob_count = 0.0
        self._ob_mean = np.zeros(self._ob_dim, dtype=np.float32)
        self._ob_var = np.zeros(self._ob_dim, dtype=np.float32)
        self._reward_summary = defaultdict(lambda: 0)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, [self._num_envs, self._ac_dim])

    @property
    def observation_space(self):
        return gym.spaces.Box(-10, 10, [self._num_envs, self._ob_dim])

    @property
    def single_action_space(self):
        return gym.spaces.Box(-1, 1, [self._ac_dim])

    @property
    def single_observation_space(self):
        return gym.spaces.Box(-10, 10, [self._ob_dim])

    def step(self, action):
        self._core.step(action, self._reward, self._done, self._timeout)
        observation = self._observe()
        info = {'reward_info': self._get_reward_dict()}

        if self._done.any():
            reset_indices = np.where(self._done)[0]
            final_observation = observation[reset_indices]
            self._core.resetByIdx(reset_indices)
            observation[reset_indices] = self._observe(reset_indices)
            info['reset_indices'] = reset_indices
            info['final_observation'] = final_observation
        return (
            observation,
            self._reward.copy(),
            np.logical_xor(self._done, self._timeout),
            self._timeout.copy(),
            info,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._core.seed(seed)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._core.reset()
        return self._observe(), {}

    def close(self):
        self._core.close()

    def _get_reward_dict(self) -> dict:
        summary = self._core.getRewardDict()
        self._core.clearRewardStats()
        return summary

    def _observe(self, indices=None):
        if indices is None:
            self._core.observe(self._observation)
            observation = self._observation.copy()
        else:
            self._core.observeByIdx(self._observation, indices)
            observation = self._observation[:len(indices), ...].copy()
        return observation

    def update_curriculum(self, epoch):
        self._core.updateCurriculum(epoch)

    def __getattr__(self, item):
        return getattr(self._core, item)
