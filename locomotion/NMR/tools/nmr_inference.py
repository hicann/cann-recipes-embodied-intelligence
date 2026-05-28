# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
# Portions are adapted from OpenMMLab tool entrypoints under Apache-2.0.
# Use and redistribution of those portions remain subject to the upstream
# Apache-2.0 license. NMR modifications are licensed under Apache-2.0.
# See THIRD_PARTY_LICENSES.md.
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
import argparse
import os
import os.path as osp
import sys

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
import src

# PyTorch 2.6 默认 weights_only=True，这里改回 False 以兼容 mmengine checkpoint
_torch_load = torch.load


def _checkpoint_load_compat(*args, **kwargs):
    return _torch_load(
        *args,
        weights_only=kwargs.pop('weights_only', False),
        **kwargs,
    )


torch.load = _checkpoint_load_compat


def add_runtime_args(parser):
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)


def parse_and_set_local_rank(parser):
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def load_test_config(args):
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=(
            'override config options with key=value pairs; list values may be '
            'passed as key="[a,b]" or key=a,b, and nested values such as '
            'key="[(a,b),(c,d)]" should be quoted without spaces.'
        ))
    add_runtime_args(parser)
    return parse_and_set_local_rank(parser)


def main():
    args = parse_args()

    cfg = load_test_config(args)
    cfg.load_from = args.checkpoint

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
