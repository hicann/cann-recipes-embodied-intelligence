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

import collections
import gc
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu

import algorithms as alg

from lltk.bin import lltk_binary
from lltk.task_registry import Task, registry
from lltk.utils.logger import Logger, WandbLogger
from lltk.utils.profile import MultiProfile
from lltk.utils.sprint import sprint, float_fmt
from lltk.utils.timer import Timer
from lltk.utils.velocity_tracking.agent_factory import AgentFactory
from lltk.utils.velocity_tracking.arguments import make_train_argparser
from lltk.utils.velocity_tracking.cfg_helper import overwrite_cfg, CfgOverwriteOptions
from lltk.utils.velocity_tracking.run_loader import RunLoader

__all__ = ['Solver']


class TrainingStats:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.episode_rew = np.zeros([num_envs, 1])
        self.episode_len = np.zeros(num_envs)
        self.rew_buffer = collections.deque(maxlen=num_envs)
        self.len_buffer = collections.deque(maxlen=num_envs)
        self.num_steps = self.accum_num_steps = 0
        self.reward_sum = 0.

    def record_step(self, rewards):
        self.reward_sum += np.sum(rewards)
        self.episode_rew += rewards
        self.episode_len += 1
        self.num_steps += self.num_envs
        self.accum_num_steps += self.num_envs

    def record_episode(self, indices):
        self.rew_buffer.extend(self.episode_rew[indices].tolist())
        self.len_buffer.extend(self.episode_len[indices].tolist())
        self.episode_rew[indices] = 0.
        self.episode_len[indices] = 0.

    @property
    def average_episode_length(self):
        return np.average(self.len_buffer) if self.len_buffer else 0

    @property
    def average_episode_reward(self):
        return np.average(self.rew_buffer) if self.rew_buffer else 0

    @property
    def average_step_reward(self):
        return self.reward_sum / self.num_steps

    def clear_step_info(self):
        self.reward_sum = 0
        self.num_steps = 0


class Solver:
    @dataclass(slots=True)
    class Arguments:
        check: bool = True
        rebuild: bool = False

        name: str = ''
        project: str = ''
        debug: bool = False
        wandb_args: tuple[str] = ()

        tag: str = None
        extra_cfg_files: tuple[str] = ()
        num_threads: int = None
        overwrite: tuple[str] = ()
        overwrite_env: tuple[str] = ()

        weight: str = ''
        robot: str = None
        algorithm: str = 'ppo'
        algorithm_cfg_path: str = None
        device: str = None
        seed: int = None

        init_epoch: int = None
        warmup: int = 0
        log_interval: int = None
        save_interval: int = None
        random_episode_index: bool = True

        lock_rms: Optional[int] = None
        strictly_symmetric: bool = False

        task: Task = None
        logger: Logger = None

    def __init__(self, args: Arguments = None):
        # Configure logging to show info messages
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        
        self.args = args or self.parse_command_line_args()
        registry.check_library(self.args.check, self.args.rebuild)

        # task specification
        self.task = self.args.task or registry.get(self.args.robot, tag=self.args.tag)
        self.precedent_run = self.load_run(self.args.weight) if self.args.weight else None
        self.last_switch = self.epoch = (0 if self.precedent_run is None else self.precedent_run.iteration)

        # configurations
        algorithm = self.args.algorithm.lower()
        if self.args.algorithm_cfg_path is None:
            self.args.algorithm_cfg_path = algorithm
        self.cfg = self.load_task_cfg()
        self.cfg['algorithm'] = algorithm
        if self.args.log_interval is None:
            self.args.log_interval = self.cfg.get('log_interval', 5)
        if self.args.save_interval is None:
            self.args.save_interval = self.cfg.get('save_interval', 500)
        self.alg_prof = MultiProfile(self.cfg[algorithm])
        if self.args.init_epoch is not None:
            self.last_switch = self.epoch = self.args.init_epoch
        self.num_epochs = self.cfg['num_epochs']
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
        self.device = alg.get_device(self.args.device)
        sprint.b(f'Training on {self.device}')
        self.agent_factory = AgentFactory(algorithm, self.device)

        # create environment from the configuration file
        self.quiet = alg.CONFIG.ddp and alg.CONFIG.local_rank != 0

        self.logger = None if self.quiet else (self.args.logger or self.make_logger())
        self.env = self.make_env()
        self.num_envs = self.env.num_envs

        # create agent
        self.agent = self.make_agent()

        # training data
        self.stats = TrainingStats(self.num_envs)
        self.timer = Timer()

    @classmethod
    def parse_command_line_args(cls) -> Arguments:
        parser = make_train_argparser()
        parser.add_argument('--lock-rms', metavar="EPOCH", nargs='?', type=int, const=0, default=None,
                            help='lock observation normalization after <N> epochs')
        parser.add_argument('--strictly-symmetric', action='store_true', help='make policy strictly symmetric')
        return parser.parse_args(namespace=cls.Arguments())

    def load_run(self, weight):
        loaded_run = RunLoader(weight)
        # Avoid all environments finish at the same time
        self.args.random_episode_index = True
        return loaded_run

    def load_task_cfg(self):
        extra_cfg_files = list(self.args.extra_cfg_files)
        if self.args.algorithm_cfg_path:
            extra_cfg_files.insert(0, self.args.algorithm_cfg_path)
        cfg = self.task.load_cfg(*extra_cfg_files)
        overwrite_cfg(cfg, CfgOverwriteOptions(
            overwrite=self.args.overwrite,
            overwrite_env=self.args.overwrite_env,
            num_threads=self.args.num_threads
        ))
        cfg['strictly_symmetric'] = self.args.strictly_symmetric
        return cfg

    def make_env(self):
        if self.logger is not None:
            lltk_binary.setLogDir(f'{self.logger.log_dir}/logs')
        cfg = self.cfg['environment'].copy()
        if self.quiet:
            cfg['verbose'] = False
        if alg.CONFIG.ddp:
            cfg['num_envs'] = int(cfg['num_envs'] / alg.CONFIG.world_size)
            if 'num_threads' in cfg:
                cfg['num_threads'] = int(cfg['num_threads'] / alg.CONFIG.world_size)

        env = self.task.make_vec(cfg)
        env.set_interface_type('privileged')
        env.update_curriculum(self.epoch)
        if self.precedent_run is not None:
            env.load_scaling(self.precedent_run.scaling_dict)
        env.reset()
        if self.args.random_episode_index:
            env.randomize_episode_index()
        return env

    def make_agent(self):
        agent = self.agent_factory(
            self.env, self.cfg['architecture'],
            self.alg_prof.get_profile(self.epoch),
            init_epoch=self.epoch + 1
        )
        if self.precedent_run is not None:
            agent.load_state_dict(torch.load(self.precedent_run.weight_path))
            if self.env.verbose:
                logging.info(f'\nResuming {self.precedent_run}\n')
        return agent

    def switch_agent_profile(self):
        alg.ddp_print_once(f'Epoch {self.epoch}: Switching agent profile.')
        self.last_switch = self.epoch
        self.agent = self.agent_factory(
            self.env, self.cfg['architecture'],
            self.alg_prof.get_profile(self.epoch),
            last_agent=self.agent, init_epoch=self.epoch + 1
        )
        gc.collect()

    def make_logger(self):
        return WandbLogger(
            self.task.log_dir,
            run_name="debug" if self.args.debug else self.args.name,
            run_cfg=self.cfg,
            project=self.args.project,
            disabled=self.args.debug,
            arg_strs=self.args.wandb_args,
        )

    def start_training_loop(self, num_epochs=None):
        obs = priv_obs = None
        if num_epochs is not None:
            for _ in range(num_epochs):
                obs, priv_obs = self.step_epoch(obs, priv_obs)
            return
        while self.epoch < self.num_epochs:
            obs, priv_obs = self.step_epoch(obs, priv_obs)

    def step_epoch(self, obs=None, priv_obs=None):
        self.epoch += 1
        self.stats.clear_step_info()
        self.timer.clear_all()

        self.timer.start('epoch')
        self.env.update_curriculum(self.epoch)
        for step in range(self.agent.num_collects):
            obs, priv_obs = self.step_once(obs, priv_obs)
        warmup = (self.epoch - self.last_switch) < self.args.warmup
        with self.timer.record('update'):
            alg_info = self.agent.update(warmup=warmup)
        self.timer.stop('epoch')

        if self.args.log_interval > 0 and self.epoch % self.args.log_interval == 0:
            self.log_stats(alg_info)
        if not self.quiet:
            self.print_stats(warmup)
            if self.args.save_interval > 0 and self.epoch % self.args.save_interval == 0:
                self.save_checkpoint()

        if self.alg_prof.is_time_to_switch(self.epoch):
            self.switch_agent_profile()
        return obs, priv_obs

    def step_once(self, obs=None, priv_obs=None):
        rms = self.args.lock_rms is None or self.epoch <= self.args.lock_rms
        if obs is None:
            obs, priv_obs = self.env.observe(device=self.device)

        
        with self.timer.record('infer'):
            action = self.agent.act(obs, priv_obs)
        with self.timer.record('step'):
            reward, done, timeout = self.env.step(action)
        with self.timer.record('observe'):
            next_obs, next_priv_obs = self.env.observe(update_stats=rms, device=self.device)

        with self.timer.record('infer'):
            self.agent.step(
                next_obs, reward, done, timeout, next_priv_obs,
                task_id=self.env.get_terrain_type(),
                reference=self.env.observe_reference(),
            )
        self.stats.record_step(reward)

        if done.any():
            with self.timer.record('step1'):
                reset_indices = np.where(done)[0]
                self.env.reset_by_idx(reset_indices)
            with self.timer.record('observe1'):
                reset_obs, reset_priv_obs = self.env.observe_by_idx(
                    reset_indices,
                    update_stats=rms and not alg.CONFIG.ddp,
                    device=self.device,
                )
                next_obs[reset_indices] = reset_obs
                if priv_obs is not None:
                    next_priv_obs[reset_indices] = reset_priv_obs
            self.stats.record_episode(reset_indices)

        return next_obs, next_priv_obs

    def print_stats(self, is_warmup):
        header = f' {"Warmup" if is_warmup else "Train"} {self.epoch} / {self.num_epochs} '
        average_episode_length = float_fmt(self.stats.average_episode_length, 6)
        average_episode_reward = float_fmt(self.stats.average_episode_reward, 6)
        average_step_reward = float_fmt(self.stats.average_step_reward, 6)

        step_time = float_fmt(self.timer['step'], 4)
        observe_time = float_fmt(self.timer['observe'], 4)
        infer_time = float_fmt(self.timer['infer'], 4)
        update_time = float_fmt(self.timer['update'], 4)

        step1_time = float_fmt(self.timer['step1'], 4)
        observe1_time = float_fmt(self.timer['observe1'], 4)

        logging.info(
            f'┌{header.center(40, "─")}┐\n'
            f'│ {"average episode length":<28} {average_episode_length:<9} │\n'
            f'│ {"average episode reward":<28} {average_episode_reward:<9} │\n'
            f'│ {"average step reward   ":<28} {average_step_reward   :<9} │\n'
            f'│ {"env step / observe  ":<25} {step_time} / {observe_time}  │\n'
            f'│ {"agent infer / update":<25} {infer_time} / {update_time}  │\n'
            '└' + '─' * 40 + '┘'
        )

    def log_stats(self, info=None):
        if info is None:
            info = {}

        info['Train/episode_len'] = self.stats.average_episode_length
        info['Train/episode_rew'] = self.stats.average_episode_reward
        info.update({f'Reward/{name}': val for name, val in self.env.get_reward_dict(averaged=True).items()})
        info = alg.ddp_average_info_dict(info)

        num_steps = self.stats.num_steps * alg.CONFIG.world_size
        info['Perf/fps'] = num_steps / self.timer.get('step', 'observe')
        info['Perf/agent_fps'] = num_steps / self.timer.get('infer', 'update')
        info['Train/num_iter'] = self.stats.accum_num_steps * alg.CONFIG.world_size

        if self.logger is not None:
            self.logger.log(info, self.epoch)

    def save_checkpoint(self):
        logging.info(f"Epoch {self.epoch}: Saving current policy.")
        self.logger.save_th(self.agent.state_dict(), f'full_{self.epoch}.pt')
        self.logger.save_np(self.env.scaling_dict(), f'scaling_{self.epoch}.npz')
        logging.info(f"Epoch {self.epoch}: Policy saved.")