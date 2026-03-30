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
from dataclasses import dataclass
from typing import Optional

from lltk.utils.profile import recursively_merge

_logger = logging.getLogger(__name__)


@dataclass
class CfgOverwriteOptions:
    """Options for configuration overwrite."""
    overwrite: Optional[list] = None
    overwrite_env: Optional[list] = None
    num_envs: Optional[int] = None
    num_threads: Optional[int] = None
    seed: Optional[int] = None
    verbose: Optional[bool] = None

__all__ = ['overwrite_cfg', 'sync_cfg', 'sync_env_cfg', 'process_curriculum']


def _parse_list(list_str):
    if not (list_str[0] == '[' and list_str[-1] == ']'):
        raise ValueError(f'Invalid list format: {list_str}')
    list_str = list_str[1:-1].strip()
    result = []
    while list_str:
        val, list_str = _parse_first_element(list_str)
        result.append(val)
    return result


def _parse_dict(dict_str):
    if not (dict_str[0] == '{' and dict_str[-1] == '}'):
        raise ValueError(f'Invalid dict format: {dict_str}')
    dict_str = dict_str[1:-1].strip()
    result = {}
    while dict_str:
        segment = dict_str.index('=')
        key = _parse_value(dict_str[:segment])
        val, dict_str = _parse_first_element(dict_str[segment + 1:])
        result[key] = val
    return result


def _get_paired(s, char2):
    char1 = s[0]
    count = 1
    for idx, char in enumerate(s[1:], 1):
        if char == char1:
            count += 1
        elif char == char2:
            count -= 1
        if count == 0:
            return idx + 1
    raise ValueError(f"Unpaired '{char1}' and '{char2}'.")


def _parse_first_element(val_str: str):
    val_str = val_str.strip()
    if val_str[0] == '[':
        segment = _get_paired(val_str, ']')
        first_element = _parse_list(val_str[:segment])
        remains = val_str[segment:].strip()
        if remains and remains[0] == ',':
            remains = remains[1:].strip()
    elif val_str[0] == '{':
        segment = _get_paired(val_str, '}')
        first_element = _parse_dict(val_str[:segment])
        remains = val_str[segment:].strip()
        if remains and remains[0] == ',':
            remains = remains[1:].strip()
    elif ',' in val_str:
        segment = val_str.index(',')
        first_element = _parse_value(val_str[:segment])
        remains = val_str[segment + 1:].strip()
    else:
        first_element = _parse_value(val_str)
        remains = ''
    return first_element, remains


def _parse_value(val_str: str):
    val_str = val_str.strip().lower()
    if val_str in ('true', 'yes', 'on'):
        return True
    if val_str in ('false', 'no', 'off'):
        return False
    if val_str in ('null',):
        return None
    if val_str.isdigit():
        return int(val_str)
    try:
        return float(val_str)
    except ValueError:
        return val_str


def parse_value(val_str: str):
    value, remains = _parse_first_element(val_str)
    if remains:
        raise ValueError(f'Unexpected trailing characters after parsing: {remains}')
    return value


def update_value(dict_, key: str, val: str):
    if '.' in key:
        key, key_seq = key.split('.', 1)
        if key.isdigit():
            key = int(key)
        update_value(dict_[key], key_seq, val)
        return
    if key.isdigit():
        key = int(key)
    dict_[key] = parse_value(val)


def overwrite_cfg(cfg, options: CfgOverwriteOptions):
    if options.overwrite is not None:
        for kv in options.overwrite:
            update_value(cfg, *kv.split('=', 1))
    if options.overwrite_env is not None:
        for kv in options.overwrite_env:
            update_value(cfg['environment'], *kv.split('=', 1))
    if options.num_envs is not None:
        cfg['environment']['num_envs'] = options.num_envs
    if options.num_threads is not None:
        cfg['environment']['num_threads'] = options.num_threads
    if options.seed is not None:
        cfg['environment']['seed'] = options.seed
    if options.verbose is not None:
        cfg['environment']['verbose'] = options.verbose
    return cfg


def _compare_dicts(d1: dict, d2: dict, includes=None, excludes=()):
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        raise ValueError('Both arguments must be dictionaries.')
    difference = {}
    missing_keys = set()
    if includes is None:
        includes = set(d1.keys()).union(d2.keys())
    for key in includes:
        if key not in d2:
            if key in d1:
                missing_keys.add(key)
            continue
        if d1.get(key, None) != d2[key]:
            difference[key] = d2[key]
    for key in excludes:
        difference.pop(key, None)
        missing_keys.discard(key)
    return difference, missing_keys


def _del_key(d: dict, key, verbose=True, prefix=''):
    if key not in d:
        return
    d.pop(key)
    if verbose:
        _logger.info(f'Del {prefix}{key}')


def _set_val(d: dict, key, val, verbose=True, prefix=''):
    d[key] = val
    if verbose:
        _logger.info(f'Set {prefix}{key} = {val}')


@dataclass
class SyncOptions:
    """Options for dictionary synchronization."""
    key: Optional[str] = None
    includes: Optional[set] = None
    excludes: tuple = ()
    verbose: bool = True
    prefix: str = ''


def _sync_dict(d1: dict, d2: dict, options: SyncOptions):
    if options.key is not None:
        if options.key not in d2:
            _del_key(d1, options.key, verbose=options.verbose, prefix=options.prefix)
            return
        val = d2[options.key]
        if not isinstance(d1.get(options.key), dict) or not isinstance(val, dict):
            if d1.get(options.key) != val:
                _set_val(d1, options.key, val, verbose=options.verbose, prefix=options.prefix)
            return
        d1, d2 = d1[options.key], val
        options.prefix += f'{options.key}.'
    difference, missing_keys = _compare_dicts(d1, d2, options.includes, options.excludes)

    for key, val in difference.items():
        if isinstance(val, dict) and isinstance(d1.get(key), dict):
            _sync_dict(d1[key], val, SyncOptions(verbose=options.verbose, prefix=options.prefix + f'{key}.'))
        else:
            _set_val(d1, key, val, verbose=options.verbose, prefix=options.prefix)
    for key in missing_keys:
        _del_key(d1, key, verbose=options.verbose, prefix=options.prefix)


def sync_env_cfg(cfg: dict, rt: dict, verbose=True):
    _sync_dict(cfg, rt, SyncOptions(includes={'actuator', 'action', 'action_std', 'control_dt'}, verbose=verbose))
    _sync_dict(cfg, rt, SyncOptions(key='observation', verbose=verbose))
    _sync_dict(cfg, rt, SyncOptions(
        key='command_velocity',
        includes={'enabled', 'velocity_range', 'close_loop', 'decompose', 'angular'},
        verbose=verbose
    ))
    _sync_dict(cfg, rt, SyncOptions(key='command_pose', verbose=verbose))
    _sync_dict(cfg, rt, SyncOptions(key='command_height', verbose=verbose))
    _sync_dict(cfg, rt, SyncOptions(key='command_pitch', verbose=verbose))
    _sync_dict(cfg, rt, SyncOptions(key='command_gait', verbose=verbose))
    _sync_dict(cfg, rt, SyncOptions(key='velocity_range', verbose=verbose))
    _sync_dict(cfg, rt, SyncOptions(
        key='height_sensor',
        includes={'dimension', 'grid_size', 'tr_max_mask_ratio', 'xy_dots', 'xy_grid'},
        verbose=verbose
    ))
    return cfg


def sync_cfg(cfg: dict, rt: dict, verbose=True):
    sync_env_cfg(cfg['environment'], rt['environment'], verbose=verbose)
    _sync_dict(cfg['architecture'], rt['architecture'], SyncOptions(verbose=verbose))
    return cfg


def process_curriculum(cfg: dict, it: int):
    if 'curriculum' not in cfg:
        return
    epochs = sorted(cfg['curriculum'].keys())
    for epoch in epochs:
        if epoch >= it:
            break
        recursively_merge(cfg['curriculum'][epoch], cfg)
    cfg.pop('curriculum')
