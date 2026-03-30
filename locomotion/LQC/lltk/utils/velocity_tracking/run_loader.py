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

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from lltk.task_registry import registry, relpath


__all__ = ['RunLoader']


class RunLoader(object):
    def __init__(
        self,
        run_dir,
        pattern: str = 'full_*.pt',
        *patterns: str,
        iteration: int = None,
        datetime_fmt="%Y-%m-%d-%H-%M-%S",
        cache: bool = False,
    ):
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            if '@' not in run_dir.name:
                raise FileNotFoundError(f'{run_dir} not found')
            run_dir, iteration = str(run_dir).rsplit('@', 1)
            run_dir = Path(run_dir)
            iteration = int(iteration)
        weight_dir = run_dir / 'weights'
        if not weight_dir.exists():
            weight_dir = run_dir

        patterns = self._process_patterns(pattern, *patterns)
        self.weight_iters, self.weight_filenames = self._search_weight_files(weight_dir, patterns)
        if iteration is None:
            index, iteration = max(enumerate(self.weight_iters), key=lambda p: p[1])
        else:
            index = self.weight_iters.index(iteration)

        self._iter = iteration
        self._run_dir = run_dir.absolute()
        self._weight_dir = weight_dir.absolute()
        self._weight_filename = self.weight_filenames[index]
        self._weight_path = self._weight_dir / self._weight_filename
        self._scaling_path = self._weight_dir / f'scaling_{self._iter}.npz'
        if self._scaling_path.exists():
            self._scaling_dict = dict(np.load(self._scaling_path, allow_pickle=True))
        else:
            self._scaling_path = self._scaling_dict = None
        self._task_fullname = self._run_dir.parent.name
        has_dot = '.' in self._task_fullname
        has_at = '@' in self._task_fullname

        if has_dot:
            self._task_name, self._tag = self._task_fullname.split('.')
        elif has_at:
            self._task_name, self._tag = self._task_fullname.split('@')
        else:
            self._task_name, self._tag = self._task_fullname, None

        self._task = registry.get(self._task_name, tag=self._tag)
        self._run_dirname = self._run_dir.name
        try:
            self._datetime_str, self._run_name = self._run_dirname.split(':', 1)
            self._datetime = datetime.strptime(self._datetime_str, datetime_fmt)
        except ValueError:
            self._run_name = self._run_dirname
            self._datetime_str = 'Unknown'
            self._datetime = None

        with open(self.join('run.yml')) as f:
            self._cfg = yaml.safe_load(f)

        if cache:
            self._cache_log_files()

    def _cache_log_files(self):
        local_run_dir = Path(relpath('cached_logs'), self.task_fullname, self._run_dirname)
        if local_run_dir == self._run_dir:
            return
        # copy run.yml, weight and scaling of a mounted directory to local directory
        if not local_run_dir.exists():
            os.makedirs(local_run_dir)
            shutil.copy(self._run_dir / 'run.yml', local_run_dir)
        local_weight_dir = local_run_dir / 'weights'
        local_weight_path = local_weight_dir / self._weight_filename
        if not local_weight_path.exists():
            os.makedirs(local_weight_dir, exist_ok=True)
            shutil.copy(self._weight_path, local_weight_dir)
            if self.scaling_path is not None:
                shutil.copy(self.scaling_path, local_weight_dir)

        self._run_dir = local_run_dir
        self._weight_dir = local_weight_dir
        self._weight_path = local_weight_path

    # @formatter:off
    run_name      = property(lambda self: self._run_name)        # noqa: E221
    directory     = property(lambda self: self._run_dir)         # noqa: E221
    weight_dir    = property(lambda self: self._weight_dir)      # noqa: E221
    weight_path   = property(lambda self: self._weight_path)     # noqa: E221
    scaling_path  = property(lambda self: self._scaling_path)    # noqa: E221
    scaling_dict  = property(lambda self: self._scaling_dict)    # noqa: E221
    iteration     = property(lambda self: self._iter)            # noqa: E221
    task          = property(lambda self: self._task)            # noqa: E221
    task_fullname = property(lambda self: self._task_fullname)   # noqa: E221
    task_name     = property(lambda self: self._task_name)       # noqa: E221
    tag           = property(lambda self: self._tag)             # noqa: E221
    datetime      = property(lambda self: self._datetime)        # noqa: E221
    datetime_str  = property(lambda self: self._datetime_str)    # noqa: E221
    cfg           = property(lambda self: self._cfg)             # noqa: E221
    # @formatter:on

    @property
    def brief(self):
        return f'{self.task_fullname}/{self._run_name}/{self._iter}'

    def __str__(self):
        return (
            f'run\n'
            f'- task: {self.task_fullname}\n'
            f'- name: {self._run_name}\n'
            f'- start at: {self._datetime or self._datetime_str}\n'
            f'- iteration: {self._iter}'
        )

    def join(self, filename):
        return os.path.join(self._run_dir, filename)

    @classmethod
    def _search_weight_files(cls, weight_dir, patterns):
        weight_iters, weight_filenames = [], []
        for filename in os.listdir(weight_dir):
            for pattern in patterns:
                res = re.match(pattern, filename)
                if res is None:
                    continue
                ckpt_iter = int(res.group('iter'))
                weight_iters.append(ckpt_iter)
                weight_filenames.append(filename)
                break
        if not weight_iters:
            raise FileNotFoundError(f'Checkpoint not found in {weight_dir}.')
        return weight_iters, weight_filenames

    @classmethod
    def _process_patterns(cls, *patterns):
        results = []
        for pattern in patterns:
            if '*' in pattern:
                pattern = pattern.replace('*', '(?P<iter>[0-9]+)')
            results.append(pattern)
        return results
