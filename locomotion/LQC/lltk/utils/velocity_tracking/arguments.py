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

import argparse

__all__ = ['make_train_argparser', 'make_eval_argparser',
           'make_play_argparser', 'make_export_argparser']


def make_train_argparser(parser=None, minimal=False) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # environment integrity checking
    arg('--check', action=argparse.BooleanOptionalAction, default=True,
        help='check if the environment binary is updated')
    arg('--rebuild', action='store_true',
        help='rebuild the environment if not updated')

    # wandb logger args
    arg('-n', '--name', type=str, default='', help='run name')
    arg('-p', '--project', type=str, help='wandb: project')
    arg('-D', '--debug', action='store_true', help='debug mode')
    arg('--wandb-args', nargs='+', metavar='KEY=VAL', default=[],
        help='wandb arguments')

    # environment config overwrite
    arg('--tag', type=str, help='configuration tag')
    arg('-T', '--num-threads', type=int,
        help='overwrite num_threads in configuration file')
    arg('-cfg', '--extra-cfg-files', nargs='+', default=(),
        help='extra configuration files to load')
    arg('-o', '--overwrite', nargs='+', metavar='KEY=VAL', default=[],
        help='overwrite configuration')
    arg('-oe', '--overwrite-env', nargs='+', metavar='KEY=VAL', default=[],
        help='abbreviation of `-o environment.KEY=VAL`')

    # load pretrained runs
    arg('-w', '--weight', type=str, metavar="PATH/DIR",
        help='pretrained run directory or model weight path')

    if minimal:
        return parser

    # environment initialization
    arg('-r', '--robot', type=str, required=True, help='robot name')

    # algorithm config
    arg('-alg', '--algorithm', type=str, default='ppo', help='algorithm name')
    arg('--algorithm-cfg-path', type=str, default=None,
        help='configuration path of the used algorithm')
    arg('--device', type=str, help='device for training')
    arg('--seed', type=int, help='torch seed')

    arg('--init-epoch', metavar='EPOCH', type=int,
        help='initial epoch (valid only when -w is specified)')
    arg('--warmup', metavar='EPOCH', type=int, default=0,
        help='warmup epochs')
    arg('--random-episode-index', action=argparse.BooleanOptionalAction,
        default=True, help='randomize episode index at initialization')
    arg('--log-interval', type=int, help='log interval in epochs')
    arg('--save-interval', type=int, help='save interval in epochs')
    return parser


def make_eval_argparser(parser=None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    arg = parser.add_argument

    arg('run', type=str, metavar='RUN', help='run directory or model weight path')
    arg('-r', '--robot', type=str, help='robot name (default deduced from the loaded run)')
    arg('--tag', type=str, help='configuration tag')

    arg('-V', '--num-envs', type=int, default=1024, help='overwrite number of environments')
    arg('-T', '--num-threads', type=int, help='overwrite num_threads in configuration file')
    arg('--seed', type=int, help='vectorized environment seed')
    arg('-q', '--quiet', action='store_true', help='print less information')

    arg('-alg', '--algorithm', type=str, default='ppo', help='algorithm used for training')
    arg('--algorithm-cfg-path', type=str, default=None, help='configuration path of the used algorithm')
    arg('-cfg', '--extra-cfg-files', nargs='+', default=(),
        help='extra configuration files to load')
    arg('-o', '--overwrite', nargs='+', metavar='KEY=VALUE', default=[], help='overwrite configuration')
    arg('-oe', '--overwrite-env', nargs='+', metavar='KEY=VALUE',
        default=[], help='overwrite environment configuration')
    arg('--sync-cfg', action=argparse.BooleanOptionalAction, default=True,
        help='if synchronize configuration or not')
    return parser


def make_play_argparser(parser=None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # run loading and environment initialization
    arg('run', type=str, metavar='RUN', help='run directory or model weight path')
    arg('-r', '--robot', type=str, help='robot name (default deduced from the loaded run)')
    arg('--tag', type=str, help='configuration tag')
    arg('-q', '--quiet', action='store_true', help='print less information')
    arg('--seed', type=int, help='environment seed')
    arg('--headless', action='store_true', default=False, help='disable raisim server')

    arg('-alg', '--algorithm', type=str, default='ppo', help='algorithm used for training')
    arg('--algorithm-cfg-path', type=str, default=None, help='configuration path of the used algorithm')
    arg('-cfg', '--extra-cfg-files', nargs='+', default=(),
        help='extra configuration files to load')
    arg('-o', '--overwrite', nargs='+', metavar='KEY=VALUE', default=[], help='overwrite configuration')
    arg('-oe', '--overwrite-env', nargs='+', metavar='KEY=VALUE',
        default=[], help='overwrite environment configuration')
    arg('--sync-cfg', action=argparse.BooleanOptionalAction, default=True,
        help='if synchronize configuration or not')

    # policy config
    arg('--device', type=str, help='device for inference')
    arg('--fp16', action='store_true', help='convert network datatype to fp16')
    arg('--symmetry', type=str, help='symmetric <arch/world>')

    # play config
    arg('-t', '--seconds', type=int, default=60, help='play time in seconds')
    arg('-c', '--command', type=str, default='js',
        help='command type (js, random, step, fixed or guided)')
    arg('-S', '--speed', type=float, default=1., help='simulation time ratio')
    arg('-E', '--endless', action='store_true', help='play permanently')
    arg('--dump', type=str, help='dump data to a file')
    arg('--timeout', nargs='?', type=float, const=0, help='enable timeout reset')
    arg('--foxglove', action='store_true', help='use Foxglove for data visualization')

    return parser


def make_export_argparser(parser=None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('run', type=str, metavar='RUN', help='run directory or model weight path')
    arg('-o', '--outdir', type=str, default='export/', help='output directory')
    arg('-n', '--name', type=str, default='', help='output name')
    arg('--split-rnn', action='store_true', help='split lstm to matrix operations')
    arg('--blind-only', action='store_true', help='simplify the network as blind only')
    arg('--opset-version', type=int, help='onnx opset version')
    arg('--simplify', action=argparse.BooleanOptionalAction, default=True,
        help='use onnxsim to simplify the onnx model')
    return parser
