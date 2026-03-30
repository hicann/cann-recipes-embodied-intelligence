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

import algorithms as alg

__all__ = ['AgentFactory']


class AgentFactory:
    def __init__(self, alg_name, device=None):
        self.alg_name = alg_name
        try:
            self._factory = globals()[f'make_{alg_name.lower()}']
        except KeyError as e:
            raise ValueError(f'Unknown algorithm: {alg_name}') from e
        self._device = device

    def __call__(self, env, arch_cfg: dict, alg_cfg: dict, init_epoch=0, last_agent=None) -> alg.Algorithm:
        agent = self._factory(env, arch_cfg.copy(), alg_cfg.copy(), self._device, init_epoch=init_epoch)
        if last_agent is not None:
            agent.load_state_dict(last_agent.state_dict())
        return agent


def _has_symmetry_config(alg_cfg: dict) -> bool:
    """Check if symmetry-related configuration is present."""
    return (
        alg_cfg.get('symmetry_coef') is not None or
        alg_cfg.get('symmetry_loss_coef') is not None or
        alg_cfg.get('symmetry_reward_coef') is not None or
        alg_cfg.get('symmetric_experience')
    )


def make_ppo(env, arch_cfg: dict, alg_cfg: dict, device, init_epoch=0):
    agent_cls = alg.PPO
    if _has_symmetry_config(alg_cfg):
        if not getattr(env, 'is_symmetric', False):
            raise RuntimeError(f'Environment is not symmetric!')
        agent_cls = alg.SymmetryPPO
        alg_cfg['mirror_obs'] = env.ord_ob_sym
        alg_cfg['mirror_critic_obs'] = env.ob_sym
        alg_cfg['mirror_act'] = env.ac_sym

    return agent_cls(
        observation_dim=env.ordinary_ob_dim,
        critic_observation_dim=env.ob_dim,
        action_dim=env.action_dim,
        arch_cfg=arch_cfg,
        device=device,
        init_epoch=init_epoch,
        **alg_cfg
    )


def make_sac(env, arch_cfg: dict, alg_cfg: dict, device, init_epoch=0):
    actor = alg.GeneralActor.make(arch_cfg['actor'], env.ordinary_ob_dim, env.action_dim)
    critic = alg.MultiQNet.make(arch_cfg['critic'], env.ob_dim, env.action_dim)

    return alg.SAC(
        actor=actor, critic=critic, obs_rms=env.ordinary_ob_rms,
        critic_obs_rms=env.ob_rms, device=device, init_epoch=init_epoch, **alg_cfg
    )


def make_rsac(env, arch_cfg: dict, alg_cfg: dict, device, init_epoch=0):
    actor = alg.GeneralActor.make(arch_cfg['actor'], env.ordinary_ob_dim, env.action_dim)
    critic = alg.MultiQNetBase.make(arch_cfg['critic'], env.ob_dim, env.action_dim)

    return alg.RecurrentSAC(
        actor=actor, critic=critic, obs_rms=env.ordinary_ob_rms,
        critic_obs_rms=env.ob_rms, device=device, init_epoch=init_epoch, **alg_cfg
    )


def make_td3(env, arch_cfg: dict, alg_cfg: dict, device, init_epoch=0):
    actor = alg.GeneralActor.make(arch_cfg['actor'], env.ob_dim, env.action_dim)
    critic = alg.MultiQNet.make(arch_cfg['critic'], env.ob_dim, env.action_dim)

    return alg.TD3(actor=actor, critic=critic, obs_rms=env.ob_rms, device=device, init_epoch=init_epoch, **alg_cfg)
