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
"""Export PI0 vlm to ONNX and validate, mirroring dp/convert_and_verify_onnx structure."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.onnx

from lerobot.policies.pi0.modeling_pi0_vlm import PI0Policy

LOGGER = logging.getLogger(__name__)


# -----------------
# Helpers
# -----------------


def _normalize_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _parse_device(device: str) -> torch.device:
    try:
        parsed = torch.device(device)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid --device '{device}'") from exc

    if parsed.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("No CUDA available; falling back to CPU")
        return torch.device("cpu")

    if parsed.type == "cuda" and parsed.index is not None:
        count = torch.cuda.device_count()
        if parsed.index < 0 or parsed.index >= count:
            LOGGER.warning(
                "Requested cuda:%s not available (device_count=%s); falling back to CPU", parsed.index, count
            )
            return torch.device("cpu")

    return parsed


def _abs_diff_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(a - b)
    if diff.size == 0:
        return 0.0, 0.0
    return float(np.max(diff)), float(np.mean(diff))


def _cosine_similarity_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Cosine stats per batch (flattened non-batch dims)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for cosine similarity: {a.shape} vs {b.shape}")

    if a.ndim == 0:
        a = a.reshape(1)
        b = b.reshape(1)

    if a.ndim == 1:
        a2 = a.reshape(1, -1)
        b2 = b.reshape(1, -1)
    else:
        a2 = a.reshape(a.shape[0], -1)
        b2 = b.reshape(b.shape[0], -1)

    dot = np.sum(a2 * b2, axis=1)
    na = np.linalg.norm(a2, axis=1)
    nb = np.linalg.norm(b2, axis=1)
    denom = na * nb
    eps = 1e-12
    cos = dot / np.maximum(denom, eps)

    return float(np.min(cos)), float(np.max(cos)), float(np.mean(cos))


def _run_onnxruntime_cpu(onnx_path: Path, inputs: dict[str, np.ndarray]) -> tuple[list[np.ndarray], float]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    os.environ.setdefault("ORT_SEED", "42")

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    valid_inputs = {i.name for i in session.get_inputs()}
    feed = {k: v for k, v in inputs.items() if k in valid_inputs}

    t0 = time.perf_counter()
    outputs = session.run(None, feed)
    dt = time.perf_counter() - t0
    return [np.asarray(output) for output in outputs], float(dt)


def _move_policy_to_device(policy: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    policy = policy.to(device)
    for module in policy.modules():
        for buffer in module.buffers():
            if buffer is not None and buffer.device != device:
                buffer.data = buffer.data.to(device)
    return policy


def _prepare_base_tensors(
    device: torch.device, batch_size: int, lang_tokens_len: int, seed: int
) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    state = torch.zeros((batch_size, 14), dtype=torch.float32, device=device)
    image = torch.zeros((batch_size, 3, 480, 640), dtype=torch.float32, device=device) / 255.0
    lang_tokens = torch.randint(0, 1000, (batch_size, lang_tokens_len), dtype=torch.long, device=device)
    lang_masks = torch.ones((batch_size, lang_tokens_len), dtype=torch.bool, device=device)

    return {
        "observation.state": state,
        "observation.images.top": image,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
    }



class ONNXWrapper(torch.nn.Module):
    """Expose policy.select_action as a single forward op for ONNX export."""

    def __init__(self, policy: PI0Policy, example_observation: dict, device: torch.device):
        super().__init__()
        self.policy = policy.to(device)
        self.device = device
        self._keys = list(example_observation.keys())

    def forward(self, *args):
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} inputs, got {len(args)}")
        input_dict = {}
        for key, tensor in zip(self._keys, args, strict=False):
            if key == "lang_tokens":
                input_dict[key] = tensor.to(torch.long)
            elif key == "lang_masks":
                input_dict[key] = tensor.to(torch.bool)
            else:
                input_dict[key] = tensor.to(torch.float32)

        self.policy.eval()
        with torch.no_grad():
            past_kv_tensor, prefix_pad_masks = self.policy.select_action(input_dict)
        return past_kv_tensor, prefix_pad_masks


def export_onnx(
    *,
    wrapper: ONNXWrapper,
    observation: dict,
    onnx_output_path: Path,
    opset: int,
    do_constant_folding: bool,
) -> None:
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_keys = list(observation.keys())
    observation_values = [observation[k] for k in dummy_keys]

    LOGGER.info("Exporting ONNX to %s", onnx_output_path)
    torch.onnx.export(
        wrapper,
        tuple(observation_values),
        str(onnx_output_path),
        opset_version=int(opset),
        input_names=dummy_keys,
        output_names=["past_kv_tensor", "prefix_pad_masks"],
        do_constant_folding=bool(do_constant_folding),
        verbose=False,
        dynamo=True,
    )


def validate_onnx(
    *,
    wrapper: ONNXWrapper,
    observation: dict,
    onnx_output_path: Path,
    runtime_save_dir: Path,
    seed: int,
) -> None:
    dummy_keys = list(observation.keys())
    observation_values = [observation[k] for k in dummy_keys]

    wrapper.eval()
    with torch.no_grad():
        torch.manual_seed(int(seed))
        pytorch_past_kv, pytorch_prefix_mask = wrapper(*observation_values)

    runtime_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_past_kv, runtime_save_dir / "past_kv_tensor.pth")
    torch.save(pytorch_prefix_mask, runtime_save_dir / "prefix_pad_masks.pth")

    # ONNXRuntime
    onnx_inputs = {}
    for name, val in zip(dummy_keys, observation_values, strict=False):
        if isinstance(val, torch.Tensor):
            arr = val.cpu().numpy()
            if val.dtype == torch.bool:
                arr = arr.astype(bool)
            onnx_inputs[name] = arr
        else:
            onnx_inputs[name] = val

    os.environ["ORT_SEED"] = str(int(seed))
    onnx_outputs, elapsed = _run_onnxruntime_cpu(onnx_output_path, onnx_inputs)
    onnx_past_kv = np.asarray(onnx_outputs[0])
    onnx_prefix_mask = np.asarray(onnx_outputs[1])

    pyt_np = pytorch_past_kv.detach().cpu().numpy()
    pref_np = pytorch_prefix_mask.detach().cpu().numpy()

    max_diff, mean_diff = _abs_diff_metrics(pyt_np.astype(np.float32), onnx_past_kv.astype(np.float32))
    cos_min, cos_max, cos_mean = _cosine_similarity_stats(
        pyt_np.astype(np.float32), onnx_past_kv.astype(np.float32)
    )

    LOGGER.info("ONNX inference time (CPU): %.6f sec", elapsed)
    LOGGER.info("past_kv_tensor shape: torch=%s onnx=%s", tuple(pyt_np.shape), tuple(onnx_past_kv.shape))
    LOGGER.info("Max abs diff: %.6g", max_diff)
    LOGGER.info("Mean abs diff: %.6g", mean_diff)
    LOGGER.info("Cosine similarity (min/max/mean): %.6f / %.6f / %.6f", cos_min, cos_max, cos_mean)

    # prefix_pad_masks diff (bool safe)
    if np.issubdtype(pref_np.dtype, np.bool_) or np.issubdtype(onnx_prefix_mask.dtype, np.bool_):
        pref_num = pref_np.astype(np.int8)
        onnx_pref_num = onnx_prefix_mask.astype(np.int8)
        LOGGER.info("prefix_pad_masks max diff: %.6g", float(np.abs(pref_num - onnx_pref_num).max()))
        LOGGER.info("prefix_pad_masks mismatches: %d", int((pref_num != onnx_pref_num).sum()))
    else:
        LOGGER.info("prefix_pad_masks max diff: %.6g", float(np.abs(pref_np - onnx_prefix_mask).max()))
        LOGGER.info("prefix_pad_masks mean diff: %.6g", float(np.abs(pref_np - onnx_prefix_mask).mean()))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export PI0 vlm to ONNX and optionally validate.")
    p.add_argument(
        "--pretrained-policy-path", type=str, required=True, help="Path or repo with config+weights"
    )
    p.add_argument("--output", type=str, default="outputs/onnx/pi0-vlm.onnx", help="Output ONNX path")
    p.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0")
    p.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    p.add_argument("--seed", type=int, default=42, help="Seed for dummy inputs and ORT")
    p.add_argument("--batch-size", type=int, default=1, help="Dummy batch size")
    p.add_argument("--lang-tokens-len", type=int, default=48, help="Dummy language token length")
    p.add_argument(
        "--runtime-save-dir", type=str, default="runtime_save", help="Where to dump runtime tensors"
    )
    p.add_argument("--no-validate", action="store_true", help="Skip ONNXRuntime validation")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    p.add_argument(
        "--local-files-only", action="store_true", default=True, help="Load policy without network"
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    policy_path = args.pretrained_policy_path
    onnx_output_path = _normalize_path(args.output)
    runtime_save_dir = _normalize_path(args.runtime_save_dir)
    device = _parse_device(args.device)

    LOGGER.info("Loading PI0Policy from %s", policy_path)
    policy = PI0Policy.from_pretrained(
        policy_path, local_files_only=bool(args.local_files_only), strict=False
    )
    policy = _move_policy_to_device(policy, device)
    # keep half precision as in original script
    try:
        policy.model = policy.model.half()
    except Exception:
        LOGGER.warning("Failed to convert policy.model to half; continuing in float32")
    policy.eval()

    # Optional prefix embeddings
    prefix_embs_path = Path("prefix_embs.pt")
    if prefix_embs_path.exists():
        try:
            _ = torch.load(prefix_embs_path, map_location=device)
        except Exception:
            LOGGER.warning("Failed to load prefix_embs.pt; continuing without it.")

    observation = _prepare_base_tensors(
        device, int(args.batch_size), int(args.lang_tokens_len), int(args.seed)
    )
    wrapper = ONNXWrapper(policy, observation, device)

    export_onnx(
        wrapper=wrapper,
        observation=observation,
        onnx_output_path=onnx_output_path,
        opset=int(args.opset),
        do_constant_folding=True,
    )
    LOGGER.info("ONNX export finished")

    if not args.no_validate:
        LOGGER.info("Validating ONNX output vs PyTorch (CPU ORT)...")
        validate_onnx(
            wrapper=wrapper,
            observation=observation,
            onnx_output_path=onnx_output_path,
            runtime_save_dir=runtime_save_dir,
            seed=int(args.seed),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
