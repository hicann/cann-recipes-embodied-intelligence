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

import time
from collections import defaultdict
from contextlib import contextmanager

__all__ = ['Timer']


class Timer:
    def __init__(self):
        self._starts = {}
        self._total = defaultdict(lambda: 0)

    def start(self, name):
        self._starts[name] = time.time()

    def stop(self, name):
        try:
            start = self._starts.pop(name)
        except KeyError as e:
            raise RuntimeError(f'Timer {name} not started!') from e
        self._total[name] += time.time() - start

    @contextmanager
    def record(self, name):
        self.start(name)
        yield
        self.stop(name)

    def __getitem__(self, item):
        return self._total[item]

    def get(self, *names):
        return sum(self._total[name] for name in names)

    def get_all(self):
        return dict(self._total)

    def clear_all(self):
        self._starts.clear()
        self._total.clear()
