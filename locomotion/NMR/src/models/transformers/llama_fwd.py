# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
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

import torch
from mmengine.registry import MODELS
from .llama_ar import LlamaHfAr

 
@MODELS.register_module()
class LlamaHfFwd(LlamaHfAr):

    def __init__(self, **kwargs) -> None:
        '''
        end_token_idx: vocab size - 2
        pad_token_idx: vocab size - 1
        '''
        super().__init__(**kwargs)
        del self.transformer.wte

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        raise NotImplementedError("LlamaHfFwd 不支持 sample 方法（wte 已被删除）")


    def forward(self, input_embd, masks): # type: ignore
         
        _batch_size, time_steps, _channels = input_embd.size()
        x = input_embd
        if time_steps > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {time_steps}, block size is only {self.config.block_size}"
            )
        for block in self.transformer.h: # type: ignore
            x, _ = block(x, masks)
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)  # (b, t, vocab_size)
        return logits
