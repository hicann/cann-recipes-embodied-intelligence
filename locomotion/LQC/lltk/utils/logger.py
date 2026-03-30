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
import shutil
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import yaml

from lltk.task_registry import HOME

__all__ = ['Logger', 'WandbLogger', 'TensorboardLogger']


class Logger(object):
    def __init__(
        self,
        log_dir: str,
        run_name: str = None,
        run_cfg: dict = None,
    ):
        self._run_name = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        if run_name:
            if '/' in run_name:
                raise ValueError(f"`run_name` could not contain `/`!")
            self._run_name += f':{run_name}'

        self._log_dir = os.path.join(log_dir, self._run_name)
        os.makedirs(self._log_dir)
        self._info_dir = os.path.join(self._log_dir, 'info')
        os.mkdir(self._info_dir)
        self._weight_dir = os.path.join(self._log_dir, 'weights')
        os.mkdir(self._weight_dir)

        if run_cfg:
            with open(os.path.join(self._log_dir, 'run.yml'), 'w') as f:
                yaml.safe_dump(run_cfg, f, sort_keys=False)

        os.system(
            f'cd {HOME} && '
            f'git log -3     > {self._info_dir}/git_log.txt && '
            f'git diff HEAD  > {self._info_dir}/git_diff.patch && '
            f'git status -vv > {self._info_dir}/git_status.txt'
        )

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def weight_dir(self):
        return self._weight_dir

    def log(self, data: dict, epoch: int = None):
        pass

    def backup(self, path, filename=None):
        if not os.path.exists(path):
            return
        if filename is None:
            filename = os.path.basename(path)
        shutil.copyfile(path, os.path.join(self._info_dir, filename))

    def save_th(self, state_dict, filename):
        import torch
        torch.save(state_dict, os.path.join(self._weight_dir, filename))

    def save_np(self, state_dict, filename):
        np.savez(os.path.join(self._weight_dir, filename), **state_dict)


class WandbLogger(Logger):
    def __init__(
        self,
        log_dir: str,
        run_name: str = None,
        run_cfg: dict = None,
        disabled: bool = False,
        arg_strs: Iterable[str] = None,
        clear_proxy: bool = True,
        **kwargs,
    ):
        super().__init__(log_dir, run_name, run_cfg)
        self.disabled = disabled
        if self.disabled:
            return
        if arg_strs is not None:
            kwargs.update(self.parse_arg_strs(arg_strs))
            if 'clear_proxy' in kwargs:
                clear_proxy = kwargs['clear_proxy']
        if clear_proxy:
            for key in os.environ:
                if key.endswith('_PROXY') or key.endswith('_proxy'):
                    os.environ.pop(key)

        import wandb
        self._run = wandb.init(name=run_name, save_code=True, **kwargs)
        self._logger = wandb

    @classmethod
    def parse_arg_strs(cls, arg_strs):
        wandb_args = {}
        for arg_str in arg_strs:
            k, v = arg_str.split('=')
            if v in ('True', 'False'):
                v = bool(v)
            wandb_args[k] = v
        return wandb_args

    def log(self, data: dict, epoch: int = None):
        if not self.disabled:
            self._logger.log(data, step=epoch)


class TensorboardLogger(Logger):
    def __init__(
        self,
        log_dir: str,
        run_name: str = None,
        run_cfg: dict = None,
    ):
        super().__init__(log_dir, run_name, run_cfg)

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=self._log_dir, flush_secs=60)

    def log(self, data: dict, epoch: int = None):
        for k, v in data.items():
            self._writer.add_scalar(k, v, epoch)
