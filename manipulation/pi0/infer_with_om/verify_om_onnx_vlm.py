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
"""Compare PI0 vlm ONNX vs Ascend OM outputs.

Inputs (matching convert_verify_vlm export):
- observation.state: float32, shape (B, 14)
- observation.images.top: float32, shape (B, 3, H, W)
- lang_tokens: int64, shape (B, L)
- lang_masks: bool, shape (B, L)

Outputs:
- past_kv_tensor (float)
- prefix_pad_masks (bool)

This script runs ONNXRuntime (CPU) and Ascend OM, then reports abs/rel errors
and cosine similarity for past_kv_tensor plus mismatch stats for prefix_pad_masks.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception as err:  # pragma: no cover
    raise RuntimeError("Missing dependency: onnxruntime. Please install it first.") from err

LOGGER = logging.getLogger(__name__)


def _import_acl() -> tuple[Any, Any, Any]:
    try:
        import acl  # type: ignore
        import acllite_utils as utils  # type: ignore
        from acllite_model import AclLiteModel  # type: ignore
    except Exception as acl_import_err:  # pragma: no cover
        raise RuntimeError(
            "Missing Ascend ACL dependencies (acl/acllite_utils/acllite_model). "
            "Run this script in the Ascend environment where those are available."
        ) from acl_import_err
    return acl, utils, AclLiteModel



class AclLiteResource:
    def __init__(self, device_id: int) -> None:
        self.device_id = int(device_id)
        self._acl = None
        self._utils = None
        self.context = None
        self.stream = None

    def init(self) -> None:
        acl, utils, _ = _import_acl()
        self._acl = acl
        self._utils = utils

        LOGGER.info("Initializing ACL (device_id=%s)", self.device_id)
        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)
        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)
        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)
        LOGGER.info("ACL initialized")

    def close(self) -> None:
        if self._acl is None:
            return
        acl = self._acl
        LOGGER.info("Releasing ACL")
        try:
            if self.stream:
                acl.rt.destroy_stream(self.stream)
            if self.context:
                acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
        finally:
            self.stream = None
            self.context = None
        LOGGER.info("ACL released")

    def __enter__(self) -> AclLiteResource:
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, tb) -> None:
        self.close()


def _cosine_similarity_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Cosine similarity requires same shape, got {a.shape} vs {b.shape}")
    if a.ndim == 0:
        raise ValueError("Cosine similarity requires at least 1D tensors")

    vec_dim = int(a.shape[-1])
    if vec_dim <= 0:
        raise ValueError(f"Invalid last-dim for cosine similarity: {a.shape}")

    a2 = a.reshape(-1, vec_dim)
    b2 = b.reshape(-1, vec_dim)
    eps = 1e-12
    denom = np.maximum(np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1), eps)
    cos = (a2 * b2).sum(axis=1) / denom
    return float(cos.min()), float(cos.max()), float(cos.mean())


def output_error_metrics(
    onnx_outputs: list[np.ndarray],
    other_outputs: list[np.ndarray],
    *,
    include_cosine: bool = True,
) -> list[dict[str, float]]:
    if len(onnx_outputs) != len(other_outputs):
        raise ValueError(f"Output count mismatch: {len(onnx_outputs)} vs {len(other_outputs)}")

    results: list[dict[str, float]] = []
    for i, (a, b) in enumerate(zip(onnx_outputs, other_outputs, strict=False)):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.shape != b.shape:
            raise ValueError(f"Output {i} shape mismatch: {a.shape} vs {b.shape}")
        abs_error = np.abs(a - b)
        rel_error = abs_error / (np.abs(a) + 1e-8)
        item: dict[str, float] = {
            "output_index": float(i),
            "max_abs_error": float(abs_error.max()),
            "mean_abs_error": float(abs_error.mean()),
            "max_rel_error": float(rel_error.max()),
            "mean_rel_error": float(rel_error.mean()),
        }
        if include_cosine:
            cmin, cmax, cmean = _cosine_similarity_stats(a, b)
            item["cosine_min"] = cmin
            item["cosine_max"] = cmax
            item["cosine_mean"] = cmean
        results.append(item)
    return results


def _to_numpy(x: torch.Tensor, *, dtype: np.dtype | None = None) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def make_pi0_vlm_inputs(
    *,
    batch_size: int,
    lang_len: int,
    height: int,
    width: int,
    seed: int,
) -> dict[str, np.ndarray]:
    torch.manual_seed(int(seed))
    batch = int(batch_size)
    lang_len_int = int(lang_len)
    height_int = int(height)
    width_int = int(width)

    state = torch.zeros((batch, 14), dtype=torch.float32)
    image = torch.zeros((batch, 3, height_int, width_int), dtype=torch.float32)
    lang_tokens = torch.randint(0, 1000, (batch, lang_len_int), dtype=torch.long)
    lang_masks = torch.ones((batch, lang_len_int), dtype=torch.bool)

    return {
        "observation.state": _to_numpy(state, dtype=np.float32),
        "observation.images.top": _to_numpy(image, dtype=np.float32),
        "lang_tokens": _to_numpy(lang_tokens, dtype=np.int64),
        "lang_masks": _to_numpy(lang_masks, dtype=bool),
    }


def run_onnxruntime_cpu(
    onnx_model_path: Path, inputs: dict[str, np.ndarray]
) -> tuple[list[np.ndarray], list[str], float]:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(
        str(onnx_model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    input_names = [i.name for i in session.get_inputs()]
    missing = [name for name in input_names if name not in inputs]
    if missing:
        raise ValueError(f"Missing required ONNX inputs: {missing}. Available: {sorted(inputs.keys())}")

    onnx_inputs = {name: inputs[name] for name in input_names}
    start = time.perf_counter()
    outputs = session.run(None, onnx_inputs)
    elapsed = time.perf_counter() - start
    return [np.asarray(output) for output in outputs], input_names, elapsed


def infer_om_model(
    om_model_path: Path,
    *,
    input_order: list[str],
    inputs: dict[str, np.ndarray],
    device_id: int,
) -> list[np.ndarray]:
    acl, utils, acllite_model_cls = _import_acl()
    _ = (acl, utils)

    with AclLiteResource(device_id=int(device_id)):
        om_model = acllite_model_cls(str(om_model_path))
        LOGGER.info("Loaded OM model: %s", om_model_path)

        LOGGER.info("OM input order: %s", list(input_order))
        input_list = []
        for name in input_order:
            arr = inputs[name]
            if arr.dtype == np.bool_:
                arr = arr.astype(np.bool_, copy=False)
            elif np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            elif np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int64, copy=False)
            input_list.append(arr)

        start = time.perf_counter()
        outputs = om_model.execute(input_list)
        elapsed = time.perf_counter() - start
        LOGGER.info("OM inference time: %.6f sec", elapsed)
        return [np.asarray(output) for output in outputs]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare PI0 vlm ONNX vs Ascend OM outputs")
    p.add_argument("--onnx-model-path", type=str, required=True, help="Path to ONNX model")
    p.add_argument("--om-model-path", type=str, required=True, help="Path to OM model")
    p.add_argument("--device-id", type=int, default=0, help="Ascend device id")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size B")
    p.add_argument("--lang-len", type=int, default=48, help="Language token length L")
    p.add_argument("--height", type=int, default=480, help="Image height")
    p.add_argument("--width", type=int, default=640, help="Image width")
    p.add_argument("--seed", type=int, default=42, help="Seed for dummy inputs and ORT")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    onnx_model_path = Path(args.onnx_model_path).expanduser().resolve()
    om_model_path = Path(args.om_model_path).expanduser().resolve()

    inputs = make_pi0_vlm_inputs(
        batch_size=int(args.batch_size),
        lang_len=int(args.lang_len),
        height=int(args.height),
        width=int(args.width),
        seed=int(args.seed),
    )
    for k, v in inputs.items():
        LOGGER.info("Input %s shape=%s dtype=%s", k, tuple(v.shape), v.dtype)

    onnx_outputs, input_order, onnx_elapsed = run_onnxruntime_cpu(onnx_model_path, inputs)
    LOGGER.info("ONNX input order: %s", list(input_order))
    LOGGER.info("ONNX inference time: %.6f sec", onnx_elapsed)

    om_outputs = infer_om_model(
        om_model_path, input_order=input_order, inputs=inputs, device_id=int(args.device_id)
    )

    LOGGER.info("===== ONNX vs OM error metrics =====")
    kv_metrics = output_error_metrics([onnx_outputs[0]], [om_outputs[0]], include_cosine=True)[0]
    LOGGER.info(
        "past_kv_tensor: max_abs=%.6g mean_abs=%.6g max_rel=%.6g mean_rel=%.6g cos(min/max/mean)=%.3f/%.3f/%.3f",
        kv_metrics["max_abs_error"],
        kv_metrics["mean_abs_error"],
        kv_metrics["max_rel_error"],
        kv_metrics["mean_rel_error"],
        kv_metrics.get("cosine_min", float("nan")),
        kv_metrics.get("cosine_max", float("nan")),
        kv_metrics.get("cosine_mean", float("nan")),
    )

    pref_np = np.asarray(onnx_outputs[1]).astype(np.bool_)
    om_pref = np.asarray(om_outputs[1]).astype(np.bool_)
    mismatches = int((pref_np != om_pref).sum())
    LOGGER.info("prefix_pad_masks mismatches: %d", mismatches)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
