# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Export PI0 (action_expert) to ONNX and verify ONNXRuntime vs PyTorch outputs.

Refactored to be CLI-friendly and deterministic:
- Configurable pretrained path / output path / device / batch / lang length.
- Zero dummy inputs (匹配原脚本清零语义)并加入语言 tokens/masks。
- 可选跳过 ONNX 验证。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.onnx

from lerobot.policies.pi0.modeling_pi0_action_expert import PI0Policy

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PI0 action_expert to ONNX and verify outputs.")
    parser.add_argument(
        "--pretrained-policy-path",
        type=str,
        default="./models/pi0/pytorch",
        help="Path to the pretrained PI0 policy (contains config/model).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/onnx/pi0-action_expert.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--past-kv-path",
        type=str,
        default="runtime_save/past_kv_tensor.pth",
        help="Path to past_kv_tensor checkpoint (default: runtime_save/past_kv_tensor.pth).",
    )
    parser.add_argument(
        "--prefix-pad-masks-path",
        type=str,
        default="runtime_save/prefix_pad_masks.pth",
        help="Path to prefix_pad_masks checkpoint (default: runtime_save/prefix_pad_masks.pth).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy inputs.")
    parser.add_argument("--lang-len", type=int, default=48, help="Language token length.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip ONNX vs PyTorch output comparison.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def build_inputs(
    device: str,
    batch_size: int,
    lang_len: int,
    *,
    past_kv_path: str,
    prefix_pad_masks_path: str,
):
    state = torch.zeros((14,), device=device, dtype=torch.float16)
    state = state.unsqueeze(0).repeat(batch_size, 1)

    # Language tokens/masks: zeros to keep determinism (与原脚本清零行为一致)
    lang_tokens = torch.zeros((batch_size, lang_len), device=device, dtype=torch.int64)
    lang_masks = torch.zeros((batch_size, lang_len), device=device, dtype=torch.bool)

    # Runtime tensors (allow overriding default locations)
    past_kv_tensor = torch.load(past_kv_path, map_location=device)
    prefix_pad_masks = torch.load(prefix_pad_masks_path, map_location=device)

    time = torch.tensor(1.0, dtype=torch.float16, device=device)
    time = time.view(1).repeat(batch_size)
    noise = torch.zeros((batch_size, 50, 32), dtype=torch.float16, device=device)

    observation = {
        "observation.state": state,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "past_kv_tensor": past_kv_tensor,
        "prefix_pad_masks": prefix_pad_masks,
        "time": time,
        "noise": noise,
    }
    return observation


def cosine_stats_last_dim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float]:
    """Compute cosine similarity per vector (flatten leading dims, last dim as vector).

    Returns (min, max, mean).
    """
    if a.shape != b.shape:
        raise ValueError(f"Cosine similarity requires same shape, got {a.shape} vs {b.shape}")
    if a.ndim == 0:
        raise ValueError("Cosine similarity requires at least 1D tensors")
    vec_dim = int(a.shape[-1])
    if vec_dim <= 0:
        raise ValueError(f"Invalid last-dim for cosine similarity: {a.shape}")

    a2 = a.reshape(-1, vec_dim).astype(np.float32, copy=False)
    b2 = b.reshape(-1, vec_dim).astype(np.float32, copy=False)
    denom = np.maximum(np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1), eps)
    cos = (a2 * b2).sum(axis=1) / denom
    return float(cos.min()), float(cos.max()), float(cos.mean())



class ONNXWrapper(torch.nn.Module):
    def __init__(self, policy, observation):
        super().__init__()
        self.policy = policy
        self.observation = observation
        self.action_dim = 7
        self.n_action_steps = 100
        if hasattr(policy, "cfg"):
            cfg = policy.cfg
            if getattr(cfg, "use_vae", False):
                self.n_action_steps = getattr(cfg, "n_action_steps", 100)
        self._keys = list(observation.keys())

    def forward(self, *args):
        """Map positional args (same order as `self._keys`) to policy input dict with expected dtypes."""
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} inputs, got {len(args)}")

        input_dict = {}
        for key, tensor in zip(self._keys, args, strict=False):
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


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    device = args.device

    past_kv_path = Path(args.past_kv_path).expanduser()
    prefix_pad_masks_path = Path(args.prefix_pad_masks_path).expanduser()

    policy = PI0Policy.from_pretrained(args.pretrained_policy_path, local_files_only=True, strict=False)
    policy.model = policy.model.half()
    policy.to(device)

    observation = build_inputs(
        device,
        args.batch_size,
        args.lang_len,
        past_kv_path=str(past_kv_path),
        prefix_pad_masks_path=str(prefix_pad_masks_path),
    )

    onnx_wrapper = ONNXWrapper(policy, observation)
    onnx_wrapper.policy.eval()

    onnx_output_path = Path(args.output).expanduser().resolve()
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

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
        else:
            observation_values.append(v.to(torch.float16))

    LOGGER.info("Loading past_kv_tensor from %s", past_kv_path)
    LOGGER.info("Loading prefix_pad_masks from %s", prefix_pad_masks_path)
    LOGGER.info("Exporting ONNX to %s", onnx_output_path)
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

    if args.skip_verify:
        LOGGER.info("Skip verification as requested.")
        return 0

    LOGGER.info("正在验证ONNX模型输出...")
    onnx_wrapper.eval()
    with torch.no_grad():
        pytorch_output = onnx_wrapper(*observation_values)

    ort_session = ort.InferenceSession(str(onnx_output_path))
    valid_input_names = {inp.name for inp in ort_session.get_inputs()}

    onnx_inputs = {}
    for name, value in zip(dummy_keys, observation_values, strict=False):
        if name not in valid_input_names:
            continue
        if isinstance(value, torch.Tensor):
            np_val = value.cpu().numpy()
            if value.dtype == torch.bool:
                np_val = np_val.astype(bool)
            onnx_inputs[name] = np_val
        else:
            onnx_inputs[name] = value

    LOGGER.info("onnx_inputs keys: %s", list(onnx_inputs.keys()))
    onnx_outputs = ort_session.run(None, onnx_inputs)
    onnx_output = onnx_outputs[0]
    LOGGER.info("onnx output: %s", onnx_output)

    pytorch_np = pytorch_output.cpu().numpy()
    max_diff = np.abs(pytorch_np - onnx_output).max()
    mean_diff = np.abs(pytorch_np - onnx_output).mean()
    cos_min, cos_max, cos_mean = cosine_stats_last_dim(pytorch_np, onnx_output)

    LOGGER.info("PyTorch输出形状: %s", pytorch_output.shape)
    LOGGER.info("ONNX输出形状: %s", onnx_output.shape)
    LOGGER.info("最大差异: %s", max_diff)
    LOGGER.info("平均差异: %s", mean_diff)
    LOGGER.info(
        "余弦相似度（按最后一维） min/max/mean: %.4f / %.4f / %.4f",
        cos_min,
        cos_max,
        cos_mean,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
