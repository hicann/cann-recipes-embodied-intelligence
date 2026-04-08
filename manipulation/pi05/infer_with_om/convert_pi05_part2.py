# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 Shanghai Jiao Tong University
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import onnxruntime as ort
import torch
import torch.onnx

from lerobot.policies.pi05.modeling_pi05_part2 import PI05Policy

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Export PI05 Part2 model to ONNX")
parser.add_argument(
    "--pretrained-path", 
    type=str, 
    default="models/pi05-libero", 
    help="Path to the pretrained policy model"
)
parser.add_argument(
    "--output-dir", 
    type=str, 
    default="output/onnx_models/pi05", 
    help="Directory to save the exported ONNX model"
)
parser.add_argument(
    "--onnx-filename", 
    type=str, 
    default="pi05-part2.onnx", 
    help="Filename of the exported ONNX model"
)
cli_args = parser.parse_args()

# Device / env
DEVICE = "cpu"
BATCH_SIZE = 1
LANG_TOKENS_LEN = 200

# Load policy
pretrained_policy_path = cli_args.pretrained_path
policy = PI05Policy.from_pretrained(pretrained_policy_path, local_files_only=True, strict=False)
policy.model = policy.model.half()  # convert model to fp16

# runtime-saved tensors used by the model
past_kv_tensor = torch.load("runtime_save/past_kv_tensor.pth", map_location=DEVICE)
prefix_pad_masks = torch.load("runtime_save/prefix_pad_masks.pth", map_location=DEVICE)

# other inputs
time = torch.tensor(1.0, dtype=torch.float16, device=DEVICE)
action_shape = (BATCH_SIZE, 50, 32)
noise = torch.zeros(action_shape, dtype=torch.float16, device=DEVICE)

# Build observation dict (preserve key order and values as original)
observation = {
    "past_kv_tensor": past_kv_tensor,
    "prefix_pad_masks": prefix_pad_masks,
    "time": time,
    "noise": noise,
}


class ONNXWrapper(torch.nn.Module):
    def __init__(self, policy_model, observation_inputs):
        super().__init__()
        self.policy = policy_model
        self.policy.to(DEVICE)
        self.observation = observation_inputs
        self.action_dim = 7
        # default sequence length used by some policies (preserve original behavior)
        self.n_action_steps = 10
        self._keys = list(observation_inputs.keys())

    def forward(self, *args):
        """Map positional args (in the same order as `self._keys`) to the policy input dict.

        Coerce dtypes that the policy expects and return the policy action tensor.
        """
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} inputs, got {len(args)}")

        input_dict = {}
        for key, tensor in zip(self._keys, args):
            if key == "lang_tokens":
                input_dict[key] = tensor.to(torch.long)
            elif key == "lang_masks":
                input_dict[key] = tensor.to(torch.bool)
            elif key == "input_value":
                input_dict[key] = tensor.to(torch.float16)
            elif key == "prefix_pad_masks":
                input_dict[key] = tensor.to(torch.bool)
            else:
                input_dict[key] = tensor.to(torch.float16)

        with torch.no_grad():
            self.policy.eval()
            actions = self.policy.select_action(input_dict)
            return actions

# Instantiate wrapper and set eval mode (preserve original)
onnx_wrapper = ONNXWrapper(policy, observation)
onnx_wrapper.policy.eval()

# Output path
output_directory = Path(cli_args.output_dir)
output_directory.mkdir(parents=True, exist_ok=True)
onnx_output_path = output_directory / cli_args.onnx_filename

# Build ordered observation values matching wrapper key order
dummy_keys = list(observation.keys())
observation_values = []
for k in dummy_keys:
    v = observation[k]
    if k == "lang_tokens":
        observation_values.append(v.to(torch.long))
    elif k == "lang_masks":
        observation_values.append(v.to(torch.bool))
    elif k == "prefix_pad_masks":
        observation_values.append(v.to(torch.bool))
    elif k == "prefix_att_masks":
        observation_values.append(v.to(torch.bool))
    else:
        logging.warning(f"Warning: Unsupported key {k}. Defaulting to float16.")
        observation_values.append(v.to(torch.float16))

logging.info(dummy_keys)

# Export to ONNX (preserve original export args)
torch.onnx.export(
    onnx_wrapper,
    tuple(observation_values),
    str(onnx_output_path),
    opset_version=14,
    verbose=True,
    input_names=dummy_keys,
    output_names=["action"],
    do_constant_folding=True,
    dynamo=True,
)

logging.info(" ")
logging.info("Model has been converted to ONNX")

# ---------------------------
# 验证ONNX输出与PyTorch输出是否一致
# ---------------------------
logging.info("正在验证ONNX模型输出...")

onnx_wrapper.eval()
with torch.no_grad():
    pytorch_output = onnx_wrapper(*observation_values)

# torch.set_printoptions(threshold=torch.inf)
# logging.info(observation_values)
logging.info(f"pytorch output: {pytorch_output}")
# 保存在./runtime_save/pytorch_output.pth以便后续对比
torch.save(pytorch_output, "runtime_save/pytorch_output_part2.pth")

# Load and run ONNX model
ort_session = ort.InferenceSession(str(onnx_output_path))

# Prepare ONNX inputs (only include names that ONNX runtime expects)
valid_input_names = {inp.name for inp in ort_session.get_inputs()}

onnx_inputs = {}
for name, value in zip(dummy_keys, observation_values):
    if name not in valid_input_names:
        continue
    if isinstance(value, torch.Tensor):
        np_val = value.cpu().numpy()
        if value.dtype == torch.bool:
            np_val = np_val.astype(bool)
        onnx_inputs[name] = np_val
    else:
        onnx_inputs[name] = value

logging.info(f"onnx_inputs {onnx_inputs}")

onnx_outputs = ort_session.run(None, onnx_inputs)
onnx_output = onnx_outputs[0]
logging.info(f"onnx output: {onnx_output}")

# Compare outputs
pytorch_np = pytorch_output.cpu().numpy()
max_diff = np.abs(pytorch_np - onnx_output).max()
mean_diff = np.abs(pytorch_np - onnx_output).mean()

logging.info(f"PyTorch输出形状: {pytorch_output.shape}")
logging.info(f"ONNX输出形状: {onnx_output.shape}")
logging.info(f"最大差异: {max_diff}")
logging.info(f"平均差异: {mean_diff}")

# ---------------------------
# 按最后一个维度逐向量比较余弦相似度
# ---------------------------

logging.info("\n正在计算按最后一维的余弦相似度...")

# pytorch_np 和 onnx_output 都是 numpy 数组
pt = pytorch_np
onnx = onnx_output

# reshape 成二维：N × D ，其中 D 是最后一个维度
pt_vecs = pt.reshape(-1, pt.shape[-1])
onnx_vecs = onnx.reshape(-1, onnx.shape[-1])


# 计算余弦相似度
def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    sim = (a * b).sum(axis=1, keepdims=True) / (a_norm * b_norm)
    return sim.squeeze()

cos_sims = cosine_similarity(pt_vecs, onnx_vecs)

# 输出统计信息
logging.info(f"余弦相似度形状: {cos_sims.shape}")
logging.info(f"最大相似度: {cos_sims.max()}")
logging.info(f"最小相似度: {cos_sims.min()}")
logging.info(f"平均相似度: {cos_sims.mean()}")
logging.info(f"中位数相似度: {np.median(cos_sims)}")