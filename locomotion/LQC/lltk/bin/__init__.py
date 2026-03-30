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

import importlib

__all__ = ['lltk_binary']


class LazyModule:
    def __init__(self, name, package=None):
        self._name = name
        self._package = package
        self._module = None

    def set_module(self, module):
        self._module = module

    def __getattr__(self, item):
        if self._module is None:
            self._module = importlib.import_module(self._name, self._package)
        return getattr(self._module, item)


lltk_binary = LazyModule('.lltk_py', 'lltk.bin')
