# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch_npu


class LinearW8A8(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        # Offline per-channel int8 quantization of weights
        w_int8, w_perchannel_scale = torch_npu.npu_dynamic_quant(linear.weight.data)
        # Use weightNZ: w_int8_nz = torch_npu.npu_format_cast(w_int8.t(), 29)
        self.register_buffer("w_int8", w_int8.t())  # [in, out]
        self.register_buffer("w_scale", w_perchannel_scale)  # [out]
        self.bias = linear.bias

    def forward(self, x):
        # Online per-token int8 quantization of activations
        B, N, C = x.shape
        x = x.view(-1, C)
        x_int8, x_pertoken_scale = torch_npu.npu_dynamic_quant(x)
        out_bf16 = torch_npu.npu_quant_matmul(x_int8, self.w_int8,
                                    self.w_scale.view(-1),
                                    pertoken_scale=x_pertoken_scale,
                                    bias=self.bias,
                                    output_dtype=torch.bfloat16)
        out_bf16 = out_bf16.view(B, N, -1)
        return out_bf16
