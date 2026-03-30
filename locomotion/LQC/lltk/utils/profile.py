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
import re
from dataclasses import dataclass
from typing import Any, Union

from lltk.utils.sprint import sprint

__all__ = ['recursively_merge', 'MultiProfile']


def recursively_merge(src: dict, dst: dict):
    for key, val in src.items():
        if key.startswith('$'):
            dst[key[1:]] = copy.deepcopy(val)
            continue
        if not isinstance(val, dict) or not isinstance(dst.get(key), dict):
            dst[key] = copy.deepcopy(val)
            continue
        recursively_merge(val, dst[key])


@dataclass
class Profiles:
    schedule: Union[str, dict[int, str], tuple[tuple[int, str]]]
    profiles: dict[str, dict[str, Any]]


class MultiProfile:
    def __init__(self, profiles: Union[dict, Profiles]):
        if isinstance(profiles, dict):
            try:
                profiles = Profiles(**profiles)
            except TypeError:
                sprint('MultiProfile is ignored due to illegal configuration.', style='bY')
                profiles = Profiles("base", {"base": profiles})

        self._profiles = {}
        for key, val in profiles.profiles.items():
            res = re.match(r'(?P<name>\w+)\[(?P<base>\w+)\]', key)
            if res is not None:
                key = res.group('name')
                base_cfg = copy.deepcopy(self._profiles[res.group('base')])
                recursively_merge(val, base_cfg)
                val = base_cfg
            self._profiles[key] = val

        if isinstance(profiles.schedule, str):
            self._schedule = {0: profiles.schedule}
        else:
            if isinstance(profiles.schedule, dict):
                profiles.schedule = tuple(profiles.schedule.items())
            self._schedule = dict(sorted(profiles.schedule))
            if 0 not in self._schedule:
                raise ValueError('Initial profile not defined!')

        for profile_name in self._schedule.values():
            if profile_name not in self._profiles:
                raise ValueError(f'Profile {profile_name} not defined!')

    def is_time_to_switch(self, epoch):
        return epoch in self._schedule

    def get_profile(self, curr_epoch):
        prof_name = None
        for epoch, name in self._schedule.items():
            if epoch > curr_epoch:
                break
            prof_name = name
        if prof_name is None:
            raise RuntimeError(f'Profile at epoch {curr_epoch} not defined!')
        return self._profiles[prof_name]
