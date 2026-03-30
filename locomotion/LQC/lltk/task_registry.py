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

import copy
import os.path
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TaskConfig:
    """Configuration for task registration."""
    name: str
    cls: str
    cfgs: Iterable[str] = ()
    cfg_dir: str = None
    tag: str = None
    inherit: str = None

import yaml

import algorithms as alg

from lltk.bin import lltk_binary
from lltk.interface import LltkEnv, LltkVec

__all__ = ['Task', 'registry', 'relpath', 'HOME', 'RSC_PATH']


def relpath(*rel):
    return os.path.join(HOME, *rel)


HOME = os.path.realpath(f'{__file__}/../..')
RSC_PATH = relpath('resources')


@dataclass
class Task(object):
    name: str
    env: Any
    venv: Any
    cfgs: tuple[str] = ()
    cfg_dir: Any = None
    tag: str = None

    @property
    def robot(self):
        return self.name.split('.', 1)[0]

    @property
    def log_dir(self):
        if self.tag is None:
            return relpath(f'logs/{self.name}')
        return relpath(f'logs/{self.name}.{self.tag}')

    def make_env(self, cfg, render, verbose, *args, wrapper=LltkEnv, **kwargs):
        env = self.env(RSC_PATH, yaml.safe_dump(cfg, sort_keys=False), render, verbose)
        if wrapper is not None:
            env = wrapper(env, cfg, *args, **kwargs)
        return env

    def make_vec(self, cfg, wrapper=LltkVec, **kwargs):
        if wrapper is None:
            return self.venv(RSC_PATH, yaml.safe_dump(cfg, sort_keys=False), **kwargs)
        return wrapper(self.venv, cfg, **kwargs)

    def load_cfg(self, *extras):
        cfg_files = list(self.cfgs)
        cfg_files.extend(extras)

        if self.cfg_dir is not None:
            cfg_dir = Path(self.cfg_dir)
            if not cfg_dir.exists():
                raise FileNotFoundError(
                    f'Configuration not found for {self.name}'
                )
            if self.tag is not None:
                cfg_dir = cfg_dir / self.tag
                if not cfg_dir.exists():
                    raise FileNotFoundError(
                        f'Configuration not found for tag `{self.tag}` of {self.name}'
                    )
            for i, file in enumerate(cfg_files):
                if not file.endswith('.yml') and not file.endswith('.yaml'):
                    file = f'{file}.yml'
                if not Path(file).is_absolute():
                    file = cfg_dir / file
                cfg_files[i] = file

        cfg = self.update_cfg({}, files=cfg_files)
        return cfg

    def update_cfg(self, cfg, *, files=(), strings=()):
        for file in files:
            with open(file, 'r') as f:
                cfg = self.__load_yml(yaml.safe_load(f), cfg)
        for string in strings:
            cfg = self.__load_yml(yaml.safe_load(string), cfg)
        return cfg

    def __load_yml(self, new_cfg, base_cfg):
        if '$$inherit' in new_cfg:
            inherits = new_cfg.pop('$$inherit')
            if isinstance(inherits, str):
                inherits = [inherits]
            for inherit in inherits:
                name, path = inherit.split('/', 1)
                with open(os.path.join(registry.get(name).cfg_dir, path)) as f:
                    parent = yaml.safe_load(f)
                base_cfg.update(parent)
        if '$$overwrite' in new_cfg:
            overwrite = new_cfg.pop('$$overwrite')
            self.__recursively_overwrite(overwrite, base_cfg)

        base_cfg.update(new_cfg)
        return base_cfg

    @classmethod
    def __recursively_overwrite(cls, src: dict, dst: dict):
        if '$$delete' in src:
            delete_items = src.pop('$$delete')
            for item in delete_items:
                if item in dst:
                    dst.pop(item)

        for key, val in src.items():
            if key.startswith('$'):
                dst[key.removeprefix('$')] = val
                continue
            if dst.get(key) is None:
                if not isinstance(val, dict):
                    dst[key] = val
                    continue
                dst[key] = {}
            cls.__recursively_overwrite(val, dst[key])


class _TaskRegistry(object):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        self._tasks: dict[str, Task] = {}

    def _register(self, config: TaskConfig):
        robot, *_ = config.name.split('.', 1)
        try:
            robot_ns = getattr(lltk_binary, robot)
            env_cls = getattr(robot_ns, config.cls)

        except AttributeError:
            return None

        venv_cls = getattr(env_cls, 'Vectorized', None)

        if config.name in self._tasks:
            raise RuntimeError(f'Duplicated Environment {config.name}')

        cfgs = list(config.cfgs)
        cfg_dir = config.cfg_dir
        tag = config.tag
        if config.inherit:
            base = self._tasks[config.inherit]
            cfgs = [*base.cfgs, *cfgs]
            if cfg_dir is None:
                cfg_dir = base.cfg_dir
            if tag is None:
                tag = base.tag

        task = Task(
            name=config.name, env=env_cls, venv=venv_cls,
            cfgs=tuple(cfgs), cfg_dir=cfg_dir, tag=tag
        )
        self._tasks[config.name] = task
        return task

    def list_envs(self):
        if not self._tasks:
            self.__load_envs()
        return list(self._tasks.keys())

    def get(self, name, tag=None) -> Task:
        if not self._tasks:
            self.__load_envs()
        if tag is not None:
            name += f'.{tag}'
        if name not in self._tasks:
            raise ValueError(
                f'No environment named {name}, '
                f'all: {sorted(list(self._tasks.keys()))}'
            )
        task = self._tasks[name]
        task = copy.copy(task)
        return task

    def __load_envs(self):
        self._register_env('g1')
        self._register_env('g1_15dof')
        self._register_env('g1_15dof.rough')
        self._register_env('go2')

    def _register_env(self, name: str, cls='VelocityTracking', **kwargs):
        config = TaskConfig(
            name=name,
            cls=cls,
            cfgs=['env.yml'],
            cfg_dir=relpath('configs', name),
            **kwargs,
        )
        return self._register(config)
    

    @classmethod
    def check_library(cls, check, rebuild):
        cls.check_binary_exists()
        if check and not cls.check_binary_updated() and cls.rebuild(not rebuild):
            os.execl(sys.executable, sys.executable, *sys.argv)

    @classmethod
    def check_binary_exists(cls):
        version_mismatch = False
        for f in os.listdir(relpath("lltk/bin")):
            if f.startswith("lltk_py.cpython"):
                lib_py_ver = f.split('-')[1]
                cur_py_ver = f'{sys.version_info[0]}{sys.version_info[1]}'
                if lib_py_ver == cur_py_ver:
                    return
                version_mismatch = True

        if version_mismatch:
            raise RuntimeError('The environment is built with another version of Python!')
        else:
            raise RuntimeError('The environment is not built!')

    @classmethod
    def check_binary_updated(cls):
        build_dir = relpath('build')
        with open(f'{build_dir}/CMakeCache.txt') as f:
            for line in f:
                if line.startswith("CMAKE_GENERATOR"):
                    cmake_generator = line.rsplit('=', 1)[1].strip()
                    break
            else:
                raise RuntimeError("CMAKE_GENERATOR not detected!")
        if cmake_generator != 'Ninja':
            raise NotImplementedError('The environment must be generated with Ninja!')
        prompt = os.popen(f'cmake --build {build_dir} -- -n').read().strip()
        return prompt == 'ninja: no work to do.'

    @classmethod
    def rebuild(cls, get_permission_from_cl=True):
        if alg.CONFIG.ddp:
            raise RuntimeError('Rebuilding is not allowed in DDP mode!')
        if get_permission_from_cl:
            prompt = input(
                "The environment is modified after being built. "
                "Rebuild now? (Y/n/i) "
            ).strip().lower()
            if prompt == 'i':
                return False
            elif prompt and prompt != 'y':
                raise RuntimeError('Rebuild aborted by user')
        ret = os.system(f'{relpath("build.py")} --incremental')
        if ret != 0:
            from lltk.utils import sprint
            sprint.bR('Fail to rebuild!')
            raise RuntimeError('Rebuild failed')
        return True


registry = _TaskRegistry()
