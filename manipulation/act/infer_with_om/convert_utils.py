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
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency: onnxruntime. Please install it first.") from exc

from policy_input_schema import (
    PolicyInputSchema,
    extract_policy_input_schema,
    load_config_json,
    normalize_path,
    resolve_config_path,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedPolicySchema:
    config_path: Path
    config: dict
    schema: PolicyInputSchema


def load_schema(*, pretrained_policy_path: str | None, config: str | None) -> LoadedPolicySchema:
    pretrained = normalize_path(pretrained_policy_path) if pretrained_policy_path else None
    explicit = normalize_path(config) if config else None
    config_path = resolve_config_path(pretrained_policy_path=pretrained, explicit_config_path=explicit)
    cfg = load_config_json(config_path)
    schema = extract_policy_input_schema(cfg)
    return LoadedPolicySchema(config_path=config_path, config=cfg, schema=schema)


def log_schema(schema: PolicyInputSchema) -> None:
    LOGGER.info("Input count: %d", len(schema.input_keys))
    LOGGER.info("VISUAL inputs: %s", list(schema.visual_keys))
    LOGGER.info("STATE inputs: %s", list(schema.state_keys))


def make_dummy_torch_observation(
    schema: PolicyInputSchema, *, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    obs: dict[str, torch.Tensor] = {}
    for key in schema.input_keys:
        shape = schema.input_shapes[key]
        if len(shape) == 1:
            obs[key] = torch.randn((1, shape[0]), generator=gen, device=device, dtype=torch.float32)
        elif len(shape) == 3:
            c, h, w = shape
            obs[key] = torch.rand((1, c, h, w), generator=gen, device=device, dtype=torch.float32)
        else:
            obs[key] = torch.randn((1, *shape), generator=gen, device=device, dtype=torch.float32)
    return obs


def make_dummy_numpy_inputs(schema: PolicyInputSchema, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    inputs: dict[str, np.ndarray] = {}
    for key in schema.input_keys:
        shape = schema.input_shapes[key]
        if len(shape) == 1:
            arr = rng.standard_normal((1, shape[0]), dtype=np.float32)
        elif len(shape) == 3:
            c, h, w = shape
            arr = rng.random((1, c, h, w), dtype=np.float32)
        else:
            arr = rng.standard_normal((1, *shape), dtype=np.float32)
        inputs[key] = arr.astype(np.float32, copy=False)
    return inputs


def torch_obs_to_numpy_inputs(observation: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().float().cpu().numpy() for k, v in observation.items()}


def run_onnxruntime_cpu(onnx_model_path: Path, inputs: dict[str, np.ndarray]) -> tuple[list[np.ndarray], float]:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(onnx_model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])

    expected = [i.name for i in session.get_inputs()]
    missing = [name for name in expected if name not in inputs]
    if missing:
        raise ValueError(f"Missing required ONNX inputs: {missing}. Available: {sorted(inputs.keys())}")

    onnx_inputs = {name: inputs[name] for name in expected}
    start = time.perf_counter()
    outputs = session.run(None, onnx_inputs)
    elapsed = time.perf_counter() - start
    return [np.asarray(output) for output in outputs], elapsed


def abs_diff_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    return float(diff.max()), float(diff.mean())


def _cosine_similarity_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Cosine similarity stats treating last dim as a vector.

    Flattens all leading dims into a batch of vectors.
    Returns (min_cos, max_cos, mean_cos).
    """
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
    include_cosine: bool = False,
) -> list[dict[str, float]]:
    if len(onnx_outputs) != len(other_outputs):
        raise ValueError(f"Output count mismatch: {len(onnx_outputs)} vs {len(other_outputs)}")

    results: list[dict[str, float]] = []
    for i, (a, b) in enumerate(zip(onnx_outputs, other_outputs)):
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
