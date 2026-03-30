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
from typing import Optional

import numpy as np

from lltk.utils.joystick import Joystick

__all__ = ['make_commander']


class Commander:
    @staticmethod
    def get_cmd_vel() -> Optional[tuple[np.ndarray, bool, bool]]:
        return None


    @staticmethod
    def get_cmd_height(last_cmd) -> Optional[float]:
        return None

    @staticmethod
    def get_cmd_pitch(last_cmd) -> Optional[float]:
        return None


class FixedCommander(Commander):
    def __init__(self, cmd_str: str, guided=False):
        cmd_str = cmd_str.strip().split(',')
        if len(cmd_str) < 1 or len(cmd_str) > 5:
            raise ValueError('Invalid command format.')

        self.vx = self._to_float(cmd_str.pop(0))
        self.vy = self._to_float(cmd_str.pop(0)) if cmd_str else None
        self.wz = cmd_str.pop(0) if cmd_str else None
        self.guided = guided
        self.wz = self._to_float(self.wz)
        self.ch = self._to_float(cmd_str.pop(0)) if cmd_str else None
        self.cp = self._to_float(cmd_str.pop(0)) if cmd_str else None

    @staticmethod
    def _to_float(str_):
        str_ = str_.strip()
        if str_ == 'n':
            return None
        return float(str_)

    def get_cmd_vel(self):
        return np.array([self.vx, self.vy, self.wz]), self.guided, self.guided

    def get_cmd_height(self, last_cmd):
        return self.ch

    def get_cmd_pitch(self, last_cmd):
        return self.cp


class StepCommander(Commander):
    def __init__(self, step_len):
        if step_len <= 0:
            raise ValueError(f'step_len must be positive, got {step_len}')
        self.step_len = step_len
        self.idx = 0

    def get_cmd_vel(self):
        self.idx += 1
        step = self.idx // self.step_len * 0.1
        cmd_vel = np.array((np.clip(step, 0., 2.0 - step), 0., 0.))
        return cmd_vel, False, True


class JoystickCommander(Commander):
    def __init__(self):
        self.js = Joystick()

    @classmethod
    def is_available(cls):
        return Joystick.available()

    def get_cmd_vel(self):
        lin_x = -self.js.get_axis('LY')
        lin_y = -self.js.get_axis('LX')
        rot_z = -self.js.get_axis('RX')
        return np.array([lin_x, lin_y, rot_z]), False, True

    def get_cmd_height(self, last_cmd) -> float:
        return last_cmd + (int(self.js.on_press('Y')) - int(self.js.on_press('A'))) * 0.05

    def get_cmd_pitch(self, last_cmd) -> float:
        return self.js.get_axis('RY') * np.pi / 6


def make_commander(cmder_type: str, env_cfg: dict, verbose=True) -> Optional[Commander]:
    commander = None
    cmder_cfg = None
    if ':' in cmder_type:
        cmder_type, cmder_cfg = cmder_type.strip().split(':', 1)
    elif cmder_type[0].isdigit() or cmder_type[0] == '-' or cmder_type[0] == '.':
        cmder_cfg = cmder_type
        cmder_type = 'fixed'

    if cmder_type == 'step':
        dt = env_cfg['control_dt']
        commander = StepCommander(int(2.0 / dt))
    elif 'fixed'.startswith(cmder_type):
        commander = FixedCommander(cmder_cfg, guided=False)
    elif 'guided'.startswith(cmder_type):
        commander = FixedCommander(cmder_cfg, guided=True)
    elif cmder_type == 'js' or 'joystick'.startswith(cmder_type):
        if JoystickCommander.is_available():
            commander = JoystickCommander()
        elif verbose:
            logging.warning('Joystick not detected. Use random command instead.')
    elif not 'random'.startswith(cmder_type):
        raise ValueError(f'Unknown command type {cmder_type}')

    if commander is None:
        return None

    if isinstance(env_cfg.get('command_velocity'), dict):
        env_cfg['command_velocity']['randomized'] = False
    cmd_pitch = env_cfg.get('command_pitch')
    if cmd_pitch is not None:
        if isinstance(cmd_pitch, dict):
            env_cfg['command_pitch']['enabled'] = False
        else:
            env_cfg['command_pitch'] = False
    cmd_height = env_cfg.get('command_height')
    if cmd_height is not None:
        if isinstance(cmd_height, dict):
            env_cfg['command_height']['enabled'] = False
        else:
            env_cfg['command_height'] = False
    return commander