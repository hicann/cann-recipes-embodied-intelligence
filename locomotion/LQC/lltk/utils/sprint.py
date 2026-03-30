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

import math

__all__ = ['ss', 'sprint', 'float_fmt', 'float_fmt_e']


class _StringStyler(object):
    """
        Generate string with color and style prefix / suffix.
        See https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
        example:
            ss = _StringStyler()
            ss.r_           # '\033[0m'
            ss.uM_          # '\033[4;35m'
            ss.i('abc')     # '\033[3mabc\033[0m'
            ss.cyan('abc')  # '\033[0;36mabc\033[0m'
            ss.bB('abc')    # '\033[1;34mabc\033[0m'
            ss.bold_yellow('abc')  # '\033[1;33mabc\033[0m'
            ss.yellow_bold('abc')  # '\033[1;33mabc\033[0m'
    """

    CLR = '\033[0m'
    styles = {
        'r': '0', 'regular': '0',
        'b': '1', 'bold': '1',
        'i': '3', 'italic': '3',
        'u': '4', 'underlined': '4',
    }
    colors_fore = {
        'K': '30', 'black': '30',
        'R': '31', 'red': '31',
        'G': '32', 'green': '32',
        'Y': '33', 'yellow': '33',
        'B': '34', 'blue': '34',
        'M': '35', 'magenta': '35',
        'C': '36', 'cyan': '36',
        'W': '37', 'white': '37',
        'D': '49', 'default': '49',
    }

    def __getattr__(self, pattern: str):
        if pattern.endswith('_'):
            ret_escape_seq = True
            pattern = pattern.removesuffix('_')
        else:
            ret_escape_seq = False
        if pattern in self.styles:
            escape_seq = f'\033[{self.styles[pattern]}m'
        elif pattern in self.colors_fore:
            escape_seq = f'\033[0;{self.colors_fore[pattern]}m'
        else:
            if '_' in pattern and pattern.count('_') == 1:
                style_str, color_str = pattern.split('_')
            elif len(pattern) == 2:
                style_str, color_str = pattern[0], pattern[1]
            else:
                raise ValueError(f'Unknown style `{pattern}`')
            if style_str in self.styles and color_str in self.colors_fore:
                style, color = self.styles[style_str], self.colors_fore[color_str]
            elif style_str in self.colors_fore and color_str in self.styles:
                style, color = self.styles[color_str], self.colors_fore[style_str]
            else:
                raise ValueError(f'Unknown style `{pattern}`')
            escape_seq = f'\033[{style};{color}m'
        if ret_escape_seq:
            return escape_seq
        return lambda *args, **kwargs: self._proc(*args, **kwargs, escape_seq=escape_seq)

    @classmethod
    def _proc(cls, *args, escape_seq, sep=' '):
        return f'{escape_seq}{sep.join(str(arg) for arg in args)}{cls.CLR}'


ss = _StringStyler()


class _StylePrinter(object):
    def __call__(self, *args, sep=' ', style=None, **kwargs):
        if style is not None:
            print(getattr(ss, style)(*args, sep=sep), **kwargs)
        else:
            print(*args, sep=sep, **kwargs)

    def __getattr__(self, item):
        return lambda *args, **kwargs: self(*args, **kwargs, style=item)

    @classmethod
    def table(cls, data: dict[str, float], num_cols: int = 1, sep='   ', header='', precision=3):
        key_col_lens, val_col_lens = [[0] * num_cols for _ in range(2)]
        keys, num_keys = sorted(data.keys()), len(data)
        num_rows = math.ceil(num_keys / num_cols)
        for i in range(num_keys):
            row, col = int(i / num_cols), i % num_cols
            key = keys[col * num_rows + row]
            val = f'{data[key]:.{precision}f}'
            key_len = len(key)
            if key_len > key_col_lens[col]:
                key_col_lens[col] = key_len
            val_len = len(val)
            if val_len > val_col_lens[col]:
                val_col_lens[col] = val_len

        if header:
            line_len = (
                sum(key_col_lens) + sum(val_col_lens) +
                len(sep) * (num_cols - 1) + 3 * num_cols
            )
            prefix_len = int((line_len - len(header)) / 2) - 1
            suffix_len = line_len - len(header) - prefix_len - 1
            print(f"{'─' * prefix_len} {ss.i(header)} {'─' * suffix_len}")
        for i in range(num_keys):
            row, col = int(i / num_cols), i % num_cols
            key = keys[col * num_rows + row]
            val = f'{data[key]:.{precision}f}'
            key_col_len, val_col_len = key_col_lens[col], val_col_lens[col]
            entry = ss.bB(f"{key:<{key_col_len}}") + f" : {val:>{val_col_len}}"
            print(entry, end='\n' if (i + 1) % num_cols == 0 else sep)
        print()


def float_fmt(number, digit):
    string = f'{number:.{digit}f}'[:digit]
    if string[-1] != '.':
        return string
    return ' ' + string[:-1]


def float_fmt_e(number, digit):
    exponent = 0
    while abs(number) > 10:
        exponent += 1
        number /= 10
    while abs(number) < 1:
        exponent -= 1
        number *= 10
    return f'{float_fmt(number, digit)}e{exponent}'


sprint = _StylePrinter()
