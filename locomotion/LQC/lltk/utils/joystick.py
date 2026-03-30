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
import threading
import time
from dataclasses import dataclass

import evdev
from evdev import ecodes

from lltk.utils.sprint import ss, sprint

__all__ = ['Joystick']


@dataclass
class Button:
    on_press: bool = False
    pressed: bool = False
    on_release: bool = False


class Joystick:
    def __init__(self):
        # Ensure logging is configured (only applies if not already configured)
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        
        all_devices = evdev.list_devices()
        if not all_devices:
            raise RuntimeError('Joystick not detected.')
        if len(all_devices) == 1:
            idx = 0
        else:
            sprint.bG('All devices:')
            for i, path in enumerate(all_devices):
                device = evdev.InputDevice(path)
                print(f'{ss.bG_}{i}{ss.CLR}: {device.path} {device.name}')
            idx = int(input(f'{ss.b_}Enter the index of your device{ss.CLR}:  '))
            if idx < 0 or idx >= len(all_devices):
                raise RuntimeError('Invalid index.')
        self.device = evdev.InputDevice(all_devices[idx])
        logging.info(f'{self.device.name} connected.')

        self.device_cap = self.device.capabilities()
        self.axis_cap = dict(self.device_cap[ecodes.EV_ABS])
        self.axes = {code: 0. for code in self.axis_cap}
        self.buttons = {code: Button() for code in self.device_cap[ecodes.EV_KEY]}
        self.axis_map, self.button_map = self._detect_mapping()

        self.mutex = threading.Lock()
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.status = True
        self.thread.start()

    def __del__(self):
        self.status = False
        if hasattr(self, 'device'):
            self.device.close()
        if hasattr(self, 'thread'):
            self.thread.join()

    @classmethod
    def available(cls):
        return len(evdev.list_devices()) > 0

    def connected(self):
        return self.status

    def get_axis(self, axis, suppress=0.05):
        with self.mutex:
            self._assert_connected()
            axis = self._get_axis_code(axis)
            val = self.axes[axis]
            return 0. if abs(val) < suppress else val

    def on_release(self, button):
        with self.mutex:
            self._assert_connected()
            button = self._get_button_code(button)
            val = self.buttons[button].on_release
            self.buttons[button].on_release = False
            return val

    def on_press(self, button):
        with self.mutex:
            self._assert_connected()
            button = self._get_button_code(button)
            val = self.buttons[button].on_press
            self.buttons[button].on_press = False
            return val

    def pressed(self, button):
        with self.mutex:
            self._assert_connected()
            button = self._get_button_code(button)
            return self.buttons[button].pressed

    def _detect_mapping(self):
        available_mapping_fns = [self._xbox_mapping, self._beitong_mapping]
        for mapping_fn in available_mapping_fns:
            axis_map, button_map = mapping_fn()
            if (all(axis in self.axis_cap for axis in axis_map.values()) and
                all(button in self.device_cap[ecodes.EV_KEY] for button in button_map.values())):
                return axis_map, button_map
        raise RuntimeError('Unsupported joystick mapping.')

    @classmethod
    def _xbox_mapping(cls):
        return {
            'LX': ecodes.ABS_X,
            'LY': ecodes.ABS_Y,
            'RX': ecodes.ABS_RX,
            'RY': ecodes.ABS_RY,
            'LT': ecodes.ABS_Z,
            'RT': ecodes.ABS_RZ,
            'DX': ecodes.ABS_HAT0X,
            'DY': ecodes.ABS_HAT0Y,
        }, {
            'A': ecodes.BTN_A,
            'B': ecodes.BTN_B,
            'X': ecodes.BTN_X,
            'Y': ecodes.BTN_Y,
            'LB': ecodes.BTN_TL,
            'RB': ecodes.BTN_TR,
            'LAS': ecodes.BTN_THUMBL,
            'RAS': ecodes.BTN_THUMBR,
            'START': ecodes.BTN_START,
            'SELECT': ecodes.BTN_SELECT,
        }

    @classmethod
    def _beitong_mapping(cls):
        return {
            'LX': ecodes.ABS_X,
            'LY': ecodes.ABS_Y,
            'RX': ecodes.ABS_Z,
            'RY': ecodes.ABS_RZ,
            'LT': ecodes.ABS_BRAKE,
            'RT': ecodes.ABS_GAS,
            'DX': ecodes.ABS_HAT0X,
            'DY': ecodes.ABS_HAT0Y,
        }, {
            'A': ecodes.BTN_A,
            'B': ecodes.BTN_B,
            'X': ecodes.BTN_X,
            'Y': ecodes.BTN_Y,
            'LB': ecodes.BTN_TL,
            'RB': ecodes.BTN_TR,
            'LAS': ecodes.BTN_THUMBL,
            'RAS': ecodes.BTN_THUMBR,
            'START': ecodes.BTN_START,
            'SELECT': ecodes.BTN_SELECT,
        }

    def _main_loop(self):
        while self.status:
            try:
                with self.mutex:
                    self._update()
            except OSError:
                self.status = False
                break
            time.sleep(0.005)

    def _update(self):
        event = self.device.read_one()
        while event is not None:
            if event.type == ecodes.EV_KEY:
                cat = evdev.categorize(event)
                if cat.keystate == evdev.KeyEvent.key_down:
                    self.buttons[cat.scancode].on_press = True
                    self.buttons[cat.scancode].pressed = True
                elif cat.keystate == evdev.KeyEvent.key_up:
                    self.buttons[cat.scancode].on_release = True
                    self.buttons[cat.scancode].pressed = False
                elif cat.keystate == evdev.KeyEvent.key_hold:
                    self.buttons[cat.scancode].pressed = True
            if event.type == ecodes.EV_ABS:
                info: evdev.AbsInfo = self.axis_cap[event.code]
                value = (event.value - info.min) / (info.max - info.min) * 2 - 1
                self.axes[event.code] = value
            event = self.device.read_one()

    def _assert_connected(self):
        if not self.connected():
            raise EOFError('Joystick disconnected')

    def _get_axis_code(self, axis: str):
        code = self.axis_map.get(axis.upper())
        if code is not None:
            return code
        raise ValueError(f'Invalid axis name: {axis}')

    def _get_button_code(self, button: str):
        code = self.button_map.get(button.upper())
        if code is not None:
            return code
        raise ValueError(f'Invalid button name: {button}')
