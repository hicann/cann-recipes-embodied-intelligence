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

from functools import singledispatchmethod

import numpy as np
import torch

import algorithms as alg

from lltk.bin import lltk_binary
from lltk.utils.sprint import sprint

__all__ = ['SymmetryDef', 'symmetry_wrapper']


class SymmetryDef:
    def __init__(self, sym, dim: int = None):
        self._core = sym
        self._ex = sym.ex_pair + [(j, i) for i, j in sym.ex_pair] + [(i, i) for i in sym.cm_idx]
        self._nx = sym.nx_pair + [(j, i) for i, j in sym.nx_pair] + [(i, i) for i in sym.ng_idx]

        self._all = sorted(self._ex + self._nx)
        self.dst = [j for _, j in self._all]
        self.neg = [i for i, _ in sorted(self._nx)]
        self.dim = len(self._ex) + len(self._nx)

        if dim is not None and self.dim != dim:
            raise ValueError(f'Dimension mismatch: expected {dim}, got {self.dim}')
        if [i for i, _ in self._all] != list(range(self.dim)):
            raise ValueError('Invalid symmetry indices: source indices do not match expected range')
        if sorted(self.dst) != list(range(self.dim)):
            raise ValueError('Invalid symmetry indices: destination indices do not match expected range')

    def indices_to_tensor(self):
        return (
            torch.tensor(self.dst),
            torch.tensor(self.neg),
        )

    @singledispatchmethod
    def __call__(self, x: np.ndarray):
        mirrored = np.zeros_like(x)
        if x.ndim == 1:
            self._core.mirror(x, mirrored)
        else:
            # 10x faster than numpy implementation
            lltk_binary.extensions.vecMirror(self._core, x, mirrored)
        return mirrored

    @__call__.register(torch.Tensor)
    def _(self, x: torch.Tensor):
        mirrored = x[..., self.dst].clone()
        mirrored[..., self.neg] *= -1
        return mirrored


def symmetry_wrapper(policy, sym_type: str, env, verbose=True):
    if 'architecture'.startswith(sym_type):
        if verbose:
            sprint.bC('Symmetric architecture')
        return alg.SymmetricActor(policy, env.ord_ob_sym, env.ac_sym)
    if 'world'.startswith(sym_type):
        if verbose:
            sprint.bC('Symmetric world')
        return alg.MirroredActor(policy, env.ord_ob_sym, env.ac_sym)
    raise ValueError(f'Unknown symmetry type: {sym_type}')
