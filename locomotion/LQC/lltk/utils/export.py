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

import os
from dataclasses import dataclass

import numpy as np
import torch

from algorithms import NetFactory

from lltk.utils.sprint import sprint

__all__ = ['export_nn']


@dataclass
class ExportProtocol:
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    demo_inputs: tuple
    expected_outputs: tuple
    demo_input_dict: dict[str, np.ndarray]
    expected_output_dict: dict[str, np.ndarray]

    @classmethod
    def make(cls, policy: NetFactory):
        dummy_x = torch.rand(1, policy.input_dim)
        if policy.io_type == 'mlp':
            demo_inputs = (dummy_x,)
            input_names = ('x',)
            output_names = ('y',)
            with torch.inference_mode():
                y = policy.nn(*demo_inputs)
            expected_outputs = (y,)
        elif policy.io_type == 'lstm':
            h0 = c0 = torch.rand(policy.rnn_num_layers, policy.rnn_hidden_dim)
            demo_inputs = (dummy_x, (h0, c0))
            input_names = ('x', 'h0', 'c0')
            output_names = ('y', 'hn', 'cn')
            with torch.inference_mode():
                y, (hn, cn) = policy.nn(*demo_inputs)
            expected_outputs = (y, hn, cn)
        elif policy.io_type == 'gru':
            h0 = torch.rand(policy.rnn_num_layers, policy.rnn_hidden_dim)
            demo_inputs = (dummy_x, h0)
            input_names = ('x', 'h0')
            output_names = ('y', 'hn')
            with torch.inference_mode():
                y, hn = policy.nn(*demo_inputs)
            expected_outputs = (y, hn)
        elif policy.io_type == 'symmetric_lstm':
            h0 = c0 = hs0 = cs0 = torch.rand(policy.rnn_num_layers, policy.rnn_hidden_dim)
            demo_inputs = (dummy_x, (h0, c0, hs0, cs0))
            input_names = ('x', 'h0', 'c0', 'hs0', 'cs0')
            output_names = ('y', 'hn', 'cn', 'hsn', 'csn')
            with torch.inference_mode():
                y, (hn, cn, hsn, csn) = policy.nn(*demo_inputs)
            expected_outputs = (y, hn, cn, hsn, csn)
        else:
            raise RuntimeError(f'Unsupported policy type `{policy.io_type}`!')

        if isinstance(y, tuple):
            output_names = output_names[:1] + ('aux',) + output_names[1:]

        def flatten(tensors):
            for item in tensors:
                if isinstance(item, torch.Tensor):
                    yield item
                else:  # tuple[Tensor]
                    yield from flatten(item)

        return cls(
            input_names, output_names,
            demo_inputs, expected_outputs,
            {k: v.numpy() for k, v in zip(input_names, flatten(demo_inputs))},
            {k: v.numpy() for k, v in zip(output_names, flatten(expected_outputs))},
        )


def check_ort_results(ort_sess, protocol: ExportProtocol, eps=1e-4):
    ort_outputs = ort_sess.run(None, protocol.demo_input_dict)
    if len(ort_outputs) != len(protocol.expected_output_dict):
        raise RuntimeError("Number of outputs mismatch.")
    for (o1, o2) in zip(ort_outputs, protocol.expected_output_dict.values()):
        if not np.allclose(o1, o2, atol=eps):
            return False
    return True


def export_nn(
    factory: NetFactory,
    outdir: str | os.PathLike,
    name: str,
    opset_version=None,
    simplify=True,
):
    protocol = ExportProtocol.make(factory)

    import onnx
    import onnxruntime as ort
    import onnxsim
    from onnxsim import model_info

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    onnx_path = os.path.join(outdir, f'{name}.onnx')
    torch.onnx.export(
        factory.nn, protocol.demo_inputs, onnx_path,
        input_names=protocol.input_names,
        output_names=protocol.output_names,
        opset_version=opset_version,
        verbose=True,
    )

    if simplify:
        model_ori = onnx.load(onnx_path)
        model_opt, status = onnxsim.simplify(model_ori)
        if status:
            sprint.green('Simplifying...')
            model_info.print_simplifying_info(model_ori, model_opt)
            onnx.save(model_opt, onnx_path)

    ort_sess = ort.InferenceSession(onnx_path)
    if check_ort_results(ort_sess, protocol):
        sprint.green('Check Passed')
    else:
        sprint.red('Check Failed')
