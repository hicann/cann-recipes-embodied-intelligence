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

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

import algorithms as alg

from lltk.interface import LltkEnv
from lltk.task_registry import registry
from lltk.utils.sprint import sprint
from lltk.utils.timer import Timer
from lltk.utils.velocity_tracking.arguments import make_eval_argparser
from lltk.utils.velocity_tracking.cfg_helper import (
    overwrite_cfg, sync_cfg, sync_env_cfg, process_curriculum, CfgOverwriteOptions
)
from lltk.utils.velocity_tracking.run_loader import RunLoader
from lltk.utils.velocity_tracking.symmetry import symmetry_wrapper

__all__ = ['Benchmarker', 'BenchmarkerPlugin', 'get_to_steady_state', 'matplotlib_use_qt5agg']


class BenchmarkerPlugin:
    def __init__(self):
        self.env = None

    def set_env(self, env):
        self.env = env

    def step(self, mask, reward, done, timeout):
        pass


class Benchmarker:
    @dataclass(slots=True)
    class Arguments:
        run: str = None
        robot: str = None
        tag: str = None
        extra_cfg_files: tuple[str] = ()
        extra_cfg_strs: tuple[str] = ()
        algorithm: str = 'ppo'
        algorithm_cfg_path: str = None

        num_envs: int = 1024
        num_threads: int = None
        seed: int = None
        quiet: bool = False

        overwrite: tuple[str] = ()
        overwrite_env: tuple[str] = ()
        sync_cfg: bool = True

        fp16: bool = False
        symmetry: str = None
        action_stats: bool = False

    def __init__(self, args: Arguments = None):
        # Configure logging to show info messages
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        
        self.args = args or self.argparser().parse_args(namespace=self.Arguments())

        # check if NPU is available
        self.device = alg.get_device()
        self.verbose = not self.args.quiet

        # parse run info from weight
        self.run = RunLoader(self.args.run)

        # task specification
        self.task = registry.get(
            self.args.robot or self.run.task.robot,
            tag=self.args.tag or self.run.tag,
        )
        self.cfg = self.load_cfg()

        self.env = None
        self.plugin: Optional[BenchmarkerPlugin] = None

    @classmethod
    def argparser(cls):
        parser = make_eval_argparser()
        parser.add_argument('--fp16', action='store_true', help='network datatype fp16')
        parser.add_argument('--symmetry', type=str, help='symmetric <arch/world>')
        parser.add_argument('--action-stats', action='store_true', help='calculate action statistics')
        return parser

    def load_cfg(self):
        # config read from both current and runtime cfg file
        try:
            cfg = self.task.load_cfg(self.args.algorithm)
        except FileNotFoundError:
            cfg = self.run.cfg
            self.args.sync_cfg = False
        process_curriculum(cfg['environment'], self.run.iteration)
        if self.args.sync_cfg:
            sync_cfg(cfg, self.run.cfg, self.verbose)
        cfg = self.task.update_cfg(cfg, files=self.args.extra_cfg_files, strings=self.args.extra_cfg_strs)
        overwrite_cfg(cfg, CfgOverwriteOptions(
            overwrite=self.args.overwrite,
            overwrite_env=self.args.overwrite_env,
            num_envs=self.args.num_envs,
            num_threads=self.args.num_threads,
            seed=self.args.seed,
            verbose=self.verbose
        ))
        return cfg

    def set_plugin(self, plugin: BenchmarkerPlugin):
        self.plugin = plugin

    def start(self, command_line=True):
        timer = Timer()

        # create environment from the configuration file
        timer.start('init_env')
        self.env = self.task.make_vec(self.cfg['environment'])
        self.env.load_scaling(self.run.scaling_dict)
        if self.plugin is not None:
            self.plugin.set_env(self.env)
        self.env.reset()
        timer.stop('init_env')

        timer.start('load_policy')
        if self.verbose:
            logging.info(f"Loading policy from {self.run}\n")
        actor = alg.GeneralActor.make(
            self.cfg['architecture']['actor'], self.env.ordinary_ob_dim, self.env.action_dim
        ).to(self.device).restore(self.run.weight_path)
        if self.args.symmetry is not None:
            actor = symmetry_wrapper(actor, self.args.symmetry, self.env, self.verbose)
        if self.args.fp16:
            actor.half()
            sprint('datatype: FP16', style='bC')
        actor = actor.inference()
        timer.stop('load_policy')

        num_envs = self.env.num_envs
        episode_len = self.env.single_obj.getEpisodeLen()
        mask = np.ones((num_envs, 1), dtype=bool)
        ep_reward_sum = np.zeros((num_envs, 1))
        ep_lengths = np.zeros((num_envs, 1))
        action_stats = alg.Normalizer() if self.args.action_stats else None

        for _ in tqdm(range(episode_len)) if command_line else range(episode_len):
            with timer.record('observe'):
                obs = self.env.observe(update_stats=False)
            with timer.record('infer'):
                action = actor(obs)
            with timer.record('step'):
                reward, done, timeout = self.env.step(action)
            if self.plugin is not None:
                self.plugin.step(mask, reward, done, timeout)

            ep_reward_sum[mask] += reward[mask]
            ep_lengths[mask] += 1
            if done.any():
                mask = np.logical_and(mask, np.logical_not(done))
                reset_indices = np.where(done)[0]
                self.env.reset_by_idx(reset_indices)
                actor.reset(reset_indices)

            if action_stats is not None:
                action_stats.update(action)

        if command_line:
            logging.info(f'Environment Init Time:    {timer["init_env"]   :.3f}s')
            logging.info(f'Policy Loading Time:      {timer["load_policy"]:.3f}s')
            logging.info(f'Policy Inference Time:    {timer["infer"]      :.3f}s')
            logging.info(f'Environment Step Time:    {timer["step"]       :.3f}s')
            logging.info(f'Environment Observe Time: {timer["observe"]    :.3f}s')
            logging.info(f'Average Episode Length: {np.average(ep_lengths)   :.3f}')
            logging.info(f'Average Episode Reward: {np.average(ep_reward_sum):.3f}')

            if self.args.action_stats:
                np.set_printoptions(3)
                logging.info(f'Action Mean: {action_stats.mean}')
                logging.info(f'Action Var:  {action_stats.var}')
            sprint.table(self.env.get_reward_dict(), num_cols=2, header='Reward Details')

        return {
            'episode_length': np.average(ep_lengths),
            'episode_reward': np.average(ep_reward_sum),
            'action_stats': {
                'mean': action_stats.mean,
                'var': action_stats.var
            } if action_stats else None,
            'reward_details': self.env.get_reward_dict(),
            'time': timer.get_all(),
        }


def get_to_steady_state(
    env: LltkEnv, actor, thr=0.05, max_time=10
):
    obs = env.observe()
    last_vel = None
    for j in range(int(max_time)):  # run until steady
        vels = []
        init_time = env.getSimulationTime()
        while env.getSimulationTime() - init_time < 1.0:
            action = actor(obs)
            obs, rew, done, _, info = env.step(action)
            lin_vel, ang_vel = info['LinearVel'], info['AngularVel']
            vels.append([lin_vel[0], lin_vel[1], ang_vel[2]])
        avg_vel = np.array(vels).mean(axis=0)
        if last_vel is not None and np.linalg.norm(last_vel - avg_vel) < thr:
            return True
        last_vel = avg_vel
    else:
        return False


def matplotlib_use_qt5agg():
    import matplotlib

    try:
        matplotlib.use('Qt5Agg')
    except ImportError:  # headless mode
        pass
