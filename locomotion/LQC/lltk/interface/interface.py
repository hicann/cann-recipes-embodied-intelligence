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

import numpy as np
import torch
import yaml

import algorithms as alg



class LltkVec:
    Single: Any
    single_obj: Any

    def __init__(self, venv_cls, cfg: dict):
        from lltk.task_registry import RSC_PATH
        from lltk.utils import sprint

        self._core = venv_cls(RSC_PATH, yaml.safe_dump(cfg))
        self._enable_prv_ob = False
        self._get_environment_info()
        self._reward_summary = defaultdict(lambda: 0)

        self._ob_rms = alg.FiniteHistoryNormalizer(self._num_envs * self._ep_len * 50)
        # self._ob_rms.set_disable_mask(self.single_obj.ob_def.isBoolean())
        for start, end in self.single_obj.ob_def.getHomogeneousRanges(contains_ground_truth=False):
            self._ob_rms.add_group(start, end)

        self.is_symmetric = cfg.get('symmetry', False)
        if self.is_symmetric:
            from lltk.utils.velocity_tracking import SymmetryDef
            self.ob_sym = SymmetryDef(self.single_obj.symmetry(self.single_obj.ob_def), self.ob_dim)
            self.ac_sym = SymmetryDef(self.single_obj.symmetry(self.single_obj.ac_def), self.action_dim)
            self.ord_ob_sym = SymmetryDef(self.single_obj.symmetry(self.single_obj.ob_def, ordinary_only=True),
                                          self._ord_ob_dim)
            if self._core.verbose():
                sprint.bY(f'Statistics is calculated symmetrically.')

        self._extra_ob_buf = [
            np.zeros([self.num_envs, self.getExtraObDim(i)], dtype=np.float32)
            for i in range(self._core.getNumExtraOb())
        ]

    def _get_environment_info(self):
        self._num_envs = self._core.getNumOfEnvs()
        self._ob_dim = self._core.getObDim()
        self._ep_len = self.single_obj.getEpisodeLen()
        self._ord_mask = self.single_obj.ob_def.getOrdinaryMask()
        self._prv_mask = self.single_obj.ob_def.getPrivilegedMask()
        self._grt_mask = self.single_obj.ob_def.getGroundTruthMask()
        # masks for extended (ordinary + privileged) observation
        self._ext_mask = self._ord_mask | self._prv_mask
        self._ord_ext_mask = self._ord_mask[~self._grt_mask]
        self._grt_mapping = self.single_obj.ob_def.getGroundTruthMapping()
        self._ord_ob_dim = int(self._ord_mask.sum())  # np.int64 -> int
        self._prv_ob_dim = int(self._prv_mask.sum())
        self._grt_ob_dim = int(self._grt_mask.sum())
        self._ac_dim = self._core.getActionDim()
        self._ob_buf = np.zeros([self._num_envs, self._ob_dim], dtype=np.float32)
        self._rew_buf = np.zeros([self._num_envs, 1], dtype=np.float32)
        self._done_buf = np.zeros([self._num_envs, 1], dtype=bool)
        self._timeout_buf = np.zeros([self._num_envs, 1], dtype=bool)
        self._tr_type_buf = np.zeros([self._num_envs, 1], dtype=int)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def ob_dim(self):
        return self._ord_ob_dim + self._prv_ob_dim

    @property
    def action_dim(self):
        return self._ac_dim

    @property
    def verbose(self):
        return self._core.verbose()

    @property
    def ob_rms(self):
        return self._ob_rms

    @property
    def ordinary_ob_dim(self):
        return self._ord_ob_dim

    @property
    def ordinary_ob_rms(self):
        return alg.NormalizerRef(self._ob_rms, self._ord_ext_mask)

    def seed(self, seed=None):
        self._core.seed(seed)

    def set_interface_type(self, interface_type='classic'):
        if interface_type == 'classic':
            self._enable_prv_ob = False
        elif interface_type == 'privileged':
            self._enable_prv_ob = True
        else:
            raise ValueError(f'Unsupported interface type `{interface_type}`')

    def _take_core(self):
        """Transfer core ownership and clear it. For internal merge operations."""
        core = self._core
        self._core = None
        return core

    def take_core(self):
        """Public interface for core transfer during merge operations."""
        return self._take_core()

    def merge(self, other: 'LltkVec'):
        if self.single_obj.ob_def.options != other.single_obj.ob_def.options:
            raise ValueError('Cannot merge environments with different observation definitions')
        self._core.merge(other.take_core())
        self._get_environment_info()

    def reset(self):
        self._rew_buf[:] = 0.
        self._core.reset()

    def observe(self, *, normalize=True, update_stats=True, transform_fn=None, device=None):
        self._core.observe(self._ob_buf)
        observation = self._ob_buf.copy()
        if transform_fn is not None:
            observation = transform_fn(observation)
        return self._process_ob(observation, normalize, update_stats, device=device)

    def step(self, action):
        self._core.step(action, self._rew_buf, self._done_buf, self._timeout_buf)
        return self._rew_buf.copy(), self._done_buf.copy(), self._timeout_buf.copy()

    def reset_by_idx(self, indices):
        self._rew_buf[indices] = 0.
        self._core.resetByIdx(indices)

    def observe_by_idx(self, indices, *, normalize=True, update_stats=True, transform_fn=None, device=None):
        self._core.observeByIdx(self._ob_buf, indices)
        observation = self._ob_buf[:len(indices), ...].copy()
        if transform_fn is not None:
            observation = transform_fn(observation)
        return self._process_ob(observation, normalize, update_stats, device)

    def _process_ob(self, ob, normalize: bool, update_stats: bool, device: str | torch.device):
        if device is not None:
            ob = torch.as_tensor(ob, device=device)
        ext_ob = self._clone_array(ob[:, self._ext_mask])
        if update_stats:
            mean, var, count = alg.mean_var_count(ext_ob, ddp=alg.CONFIG.ddp)
            if self.is_symmetric is not None:
                count /= 2
                m_mean = self.ob_sym(mean)
                m_var = abs(self.ob_sym(var))
                self._ob_rms.update_from_stats(m_mean, m_var, count)
            self._ob_rms.update_from_stats(mean, var, count)
        if normalize:
            ext_ob = self._ob_rms.normalize(ext_ob)

        ord_ob = ext_ob[:, self._ord_ext_mask]
        if not self._enable_prv_ob:
            return ord_ob
        ob[:, self._grt_mapping[0]] = ob[:, self._grt_mapping[1]]
        ext_ob = self._clone_array(ob[:, self._ext_mask])
        if normalize:
            ext_ob = self._ob_rms.normalize(ext_ob)
        return ord_ob, ext_ob

    @staticmethod
    def _clone_array(array: np.ndarray | torch.Tensor):
        return array.copy() if isinstance(array, np.ndarray) else array.clone()

    def get_reward_dict(self, averaged=False) -> dict:
        summary = self._core.getRewardDict(averaged)
        self._core.clearRewardStats()
        return summary

    def load_scaling(self, scaling_dict):
        self._ob_rms.set_stats(**scaling_dict)

    def scaling_dict(self):
        return self._ob_rms.stats_dict()

    def close(self):
        self._core.close()

    def update_curriculum(self, epoch):
        self._core.updateCurriculum(epoch)  # update the next epoch

    def randomize_episode_index(self):
        self._core.randomizeEpisodeIndex()

    def print_options(self):
        self._core.printOptions()

    def get_terrain_type(self):
        self._core.getTerrainType(self._tr_type_buf)
        return self._tr_type_buf.copy()

    def add_extra_ob(self, ob_def):
        oid = self._core.addExtraOb(ob_def)
        self._extra_ob_buf.append(
            np.zeros([self.num_envs, self.getExtraObDim(oid)], dtype=np.float32)
        )
        return oid

    def get_extra_ob_dim(self, oid):
        return self._core.getExtraObDim(oid)

    def observe_extra(self, oid, indices=None):
        buffer = self._extra_ob_buf[oid]
        if indices is None:
            self._core.observeExtra(buffer, oid)
            return buffer.copy()
        self._core.observeExtraByIdx(buffer, oid, indices)
        return buffer[:len(indices), ...].copy()

    @staticmethod
    def observe_reference():
        return None

    def __getattr__(self, item):
        return getattr(self._core, item)


class LltkEnv:
    def __init__(self, impl, cfg, seed=None, normalize_ob=True):
        self._core = impl
        
        self.seed(seed)
        self._core.init()
        self._full_ob_dim = self._core.getObDim()
        self._full_ob = np.zeros(self._full_ob_dim, dtype=np.float32)
        self._normalize_ob = normalize_ob
        self._num_steps = 0
        self._episode_reward = 0
        self._reward_dict = defaultdict(lambda: 0)
        self._reward_count = 0

        # ordinary observation statistics
        self._ord_mask = self.ob_def.getOrdinaryMask()
        self._prv_mask = self.ob_def.getPrivilegedMask()
        self._grt_mask = self.ob_def.getGroundTruthMask()
        self._rms_mask = self._ord_mask[~self._grt_mask]
        self._grt_mapping = self.ob_def.getGroundTruthMapping()

        self.ob_dim = int(self._ord_mask.sum())  # np.int64 -> int
        self.extended_ob_dim = self.ob_dim + int(self._prv_mask.sum())  # np.int64 -> int
        self.action_dim = self._core.getActionDim()
        self._ob_mean = self._ob_std = None
        self._last_substep_info_skipped = False

        self._extra_ob_buffers = [
            np.zeros(self.getExtraObDim(i), dtype=np.float32)
            for i in range(self._core.getNumExtraOb())
        ]
        self._ob_stats = {}

        self.is_symmetric = cfg.get('symmetry', False)
        if self.is_symmetric:
            from lltk.utils.velocity_tracking import SymmetryDef
            self.ob_sym = SymmetryDef(self._core.symmetry(self.ob_def), self.extended_ob_dim)
            self.ac_sym = SymmetryDef(self._core.symmetry(self.ac_def), self.action_dim)
            self.ord_ob_sym = SymmetryDef(self._core.symmetry(self.ob_def, ordinary_only=True),
                                          self.ob_dim)

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def num_steps(self):
        return self._num_steps

    def load_scaling(self, scaling_dict):
        if not self._normalize_ob:
            raise RuntimeError("Observation normalization is off")
        self._ob_mean = scaling_dict['mean'].copy()
        self._ob_std = np.sqrt(scaling_dict['var'] + 1e-8)

    def step(self, action: np.ndarray):
        reward = self._core.step(action.squeeze())
        done = self._core.isTerminal()
        timeout = self._core.isTimeOut()
        ob = self._observe(self._normalize_ob)

        self._episode_reward += reward
        reward_info = self._core.getRewardInfo()
        for k, v in reward_info.items():
            self._reward_dict[k] += v
        self._reward_count += 1

        info = self._collect_info()
        info['Observation'] = ob.squeeze()
        info['Rewards'] = reward_info
        self._num_steps += 1
        return ob, reward, done, timeout, info

    def substeps(self, action: np.ndarray, skip_last_info=False):
        self._core.prestep(action.squeeze())
        num_substeps = self._core.getNumSubsteps()
        self._last_substep_info_skipped = skip_last_info
        for i in range(num_substeps):
            self._core.substep()
            if skip_last_info and i == num_substeps - 1:
                return
            yield self._collect_substep_info()

    def poststep(self, contains_substep_info=False):
        reward = self._core.poststep()
        done = self._core.isTerminal()
        timeout = self._core.isTimeOut()
        ob = self._observe(self._normalize_ob)

        self._episode_reward += reward
        reward_info = self._core.getRewardInfo()
        for k, v in reward_info.items():
            self._reward_dict[k] += v
        self._reward_count += 1

        info = self._collect_step_info()
        info['Observation'] = ob.squeeze()
        info['Rewards'] = reward_info
        if contains_substep_info or self._last_substep_info_skipped:
            info.update(self._collect_substep_info())
        self._num_steps += 1
        return ob, reward, done, timeout, info

    def _collect_substep_info(self):
        # @formatter:off
        return {
            'AngularVel'      : self._core.getAngularVelocity(),       # noqa: E203
            'BodyRelHeight'   : self._core.getBodyRelHeight(),         # noqa: E203
            'ContactBuf'      : self._core.getContactBuf(),            # noqa: E203
            'ContactSlope'    : self._core.getContactSlope(),          # noqa: E203
            'Disturbance'     : self._core.getExternalDisturbance(),   # noqa: E203
            'FootForce'       : self._core.getFootForce().T,           # noqa: E203
            'FootSlipVel'     : self._core.getFootSlipVel(),           # noqa: E203
            'FootVel'         : self._core.getFootVelocity().T,        # noqa: E203
            'FootVelAtContact': self._core.getFootVelAtContact().T,    # noqa: E203
            'GeneralizedCoord': self._core.getGeneralizedCoordinate(), # noqa: E203
            'GeneralizedForce': self._core.getGeneralizedForce(),      # noqa: E203
            'GeneralizedVel'  : self._core.getGeneralizedVelocity(),   # noqa: E203
            'JointPos'        : self._core.getJointPosition(),         # noqa: E203
            'JointPosTarget'  : self._core.getJointPosTarget(),        # noqa: E203
            'JointTor'        : self._core.getJointTorque(),           # noqa: E203
            'JointTorTarget'  : self._core.getDesiredTorque(),         # noqa: E203
            'JointVel'        : self._core.getJointVelocity(),         # noqa: E203
            'JointVelTarget'  : self._core.getJointVelTarget(),        # noqa: E203
            'LinearVel'       : self._core.getLinearVelocity(),        # noqa: E203
            'Rpy'             : self._core.getRpy(),                   # noqa: E203
            'RpyRate'         : self._core.getRpyRate(),               # noqa: E203
        }
        # @formatter:on

    def _collect_step_info(self):
        # @formatter:off
        return {
            'ActionFluc'      : abs(self._core.getActionFluc()),      # noqa: E203
            'AirTime'         : self._core.getAirTime(),              # noqa: E203
            'CmdHeight'       : self._core.getCmdHeight(False),       # noqa: E203
            'CmdPitch'        : self._core.getCmdPitch(),             # noqa: E203
            'CmdVel'          : self._core.getCmdVel(),               # noqa: E203
            'ContactEdge'     : self._core.getContactEdge(),          # noqa: E203
            'DistToEdge'      : self._core.getFootDistToStairEdge(),  # noqa: E203
            'EffVel'          : self._core.getEffVel(),               # noqa: E203
            'FootClr'         : self._core.getFootClr(),              # noqa: E203
            'FootContact'     : self._core.getFootContact(),          # noqa: E203
            'FootIntraDist'   : self._core.getFootIntraDist(),        # noqa: E203
            'FootLastClr'     : self._core.getFootLastClr(),          # noqa: E203
            'FootLocalPos'    : self._core.getFootLocalPosition().T,  # noqa: E203
            'FootPos'         : self._core.getFootPosition().T,       # noqa: E203
            'FootTrClr'       : self._core.getFootTerrainClr(),       # noqa: E203
            'GaitPeriod'      : self._core.getGaitPeriod(),           # noqa: E203
            'GeneralizedAccel': self._core.getGeneralizedAccel(),     # noqa: E203
            'JointAcc'        : self._core.getJointAccel(),           # noqa: E203
            'LinearAcc'       : self._core.getLinearAccel(),          # noqa: E203
            'Power'           : self._core.getPowerSum(),             # noqa: E203
            'TerrainSlope'    : self._core.getTerrainSlope(),         # noqa: E203
            # Optional info
            'Morphology'      : getattr(self._core, 'getMorphologyRatio', lambda: None)(),      # noqa: E203
        }
        # @formatter:on

    def _collect_info(self):
        return self._collect_substep_info() | self._collect_step_info()

    def seed(self, seed=None):
        if seed is None:
            seed = time.time_ns()
        self._core.seed(seed)

    def reset(self):
        self._core.reset()
        ob = self._observe(self._normalize_ob)
        self._num_steps = 0
        self._episode_reward = 0.
        return ob

    def print_options(self):
        self._core.printOptions()

    def wait_until_visualizer_connected(self):
        while not self._core.isVisualizerConnected():
            time.sleep(0.01)
        time.sleep(0.3)

    def _observe(self, normalize=True):
        self._core.observe(self._full_ob)
        ob = self._full_ob[self._ord_mask].copy()
        if normalize and self._ob_mean is not None:
            ob -= self._ob_mean[self._rms_mask]
            ob /= self._ob_std[self._rms_mask]
        return np.expand_dims(ob, axis=0)

    def observe(self, normalize=True):
        """Public method to observe the environment state."""
        return self._observe(normalize)

    def get_extended_observation(self, normalize=None):
        mask = self._ord_mask | self._prv_mask
        extended_ob = np.expand_dims(self._full_ob, axis=0)
        extended_ob[:, self._grt_mapping[0]] = extended_ob[:, self._grt_mapping[1]]
        extended_ob = extended_ob[:, mask]
        if normalize is None:
            normalize = self._normalize_ob
        if normalize and self._ob_mean is not None:
            extended_ob -= self._ob_mean
            extended_ob /= self._ob_std
        return extended_ob

    def get_reward_dict(self, averaged=False):
        if not averaged:
            return dict(self._reward_dict)
        return {key: val / max(self._reward_count, 1) for key, val in self._reward_dict.items()}

    def __getattr__(self, item):
        return getattr(self._core, item)

    def add_extra_ob(self, ob_def):
        oid = self._core.addExtraOb(ob_def)
        self._extra_ob_buffers.append(
            np.zeros(self.getExtraObDim(oid), dtype=np.float32)
        )
        return oid

    def get_extra_ob_dim(self, idx):
        return self._core.getExtraObDim(idx)

    def observe_extra(self, idx):
        self._core.observeExtra(self._extra_ob_buffers[idx], idx)
        ob = self._extra_ob_buffers[idx].copy()
        stats = self._ob_stats.get(idx, None)
        if stats is not None:
            ob -= stats['mean']
            ob /= stats['std']
        return np.expand_dims(ob, axis=0)

    def get_rgbd_image(self, width=640, height=480, camera_id=0):
        rgb, depth = self._core.getRgbdImage(width, height, camera_id)
        rgb = np.asarray(rgb).reshape(height, width, 3)
        depth = np.asarray(depth).reshape(height, width)
        return rgb, depth


