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

"""Export DiffusionPolicy (DP) UNet to ONNX and optionally validate.

This follows the conversion pattern in dp/pytorch_to_onnx/dp_unet_convert.py:
- Load a DiffusionPolicy.
- Build deterministic observation inputs.
- Compute `global_cond = policy.diffusion._prepare_global_conditioning(observation)`.
- Export ONLY the UNet forward to ONNX with inputs (sample, t, global_cond).
- Optionally validate ONNXRuntime output vs PyTorch (abs diff + cosine similarity).

Example:
    python3 dp/convert_and_verify.py \
        --pretrained-policy-path dp_model \
        --output dp/eval/dp_onnx/unet.onnx
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import onnx  # noqa: F401
except Exception as import_exc:  # pragma: no cover
    raise RuntimeError("Missing dependency: onnx. Please install it first (pip install onnx).") from import_exc

try:
    import onnxruntime as ort
except Exception as import_exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: onnxruntime. Please install it first (pip install onnxruntime)."
    ) from import_exc

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

LOGGER = logging.getLogger(__name__)


def _normalize_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _parse_device(device: str) -> torch.device:
    try:
        parsed = torch.device(device)
    except Exception as device_err:
        raise ValueError(f"Invalid --device '{device}'") from device_err

    if parsed.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("No available CUDA device; falling back to CPU")
        return torch.device("cpu")

    if parsed.type == "cuda" and parsed.index is not None:
        count = torch.cuda.device_count()
        if parsed.index < 0 or parsed.index >= count:
            LOGGER.warning(
                "Requested CUDA device cuda:%s is not available (device_count=%s); falling back to CPU",
                parsed.index,
                count,
            )
            return torch.device("cpu")

    return parsed


def _abs_diff_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = np.abs(a - b)
    if diff.size == 0:
        return 0.0, 0.0
    return float(np.max(diff)), float(np.mean(diff))


def _cosine_similarity_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Compute cosine similarity stats between two tensors.

    Flattens all non-batch dims and computes cosine similarity per batch element.
    Returns (min, max, mean).
    """
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

    # cosine = dot(a,b) / (||a||*||b||)
    dot = np.sum(a2 * b2, axis=1)
    na = np.linalg.norm(a2, axis=1)
    nb = np.linalg.norm(b2, axis=1)
    denom = na * nb
    eps = 1e-12
    cos = dot / np.maximum(denom, eps)

    return float(np.min(cos)), float(np.max(cos)), float(np.mean(cos))


def _run_onnxruntime_cpu(onnx_path: Path, inputs: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], float]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    # ORT uses a global seed env var for some ops; keep deterministic when possible.
    os.environ.setdefault("ORT_SEED", "42")

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    valid_inputs = {i.name for i in session.get_inputs()}
    feed = {k: v for k, v in inputs.items() if k in valid_inputs}

    import time

    t0 = time.perf_counter()
    outputs = session.run(None, feed)
    dt = time.perf_counter() - t0
    return [np.asarray(output) for output in outputs], float(dt)


def _move_policy_to_device(policy: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return policy.to(device)


def _infer_action_dim(policy: DiffusionPolicy) -> int:
    cfg = policy.config
    output_features = getattr(cfg, "output_features", None)
    if output_features is None:
        raise ValueError("DiffusionPolicy config is missing output_features")

    action_ft = output_features.get("action")
    if action_ft is None:
        raise ValueError("DiffusionPolicy config is missing output_features['action']")

    if not hasattr(action_ft, "shape") or len(action_ft.shape) == 0:
        raise ValueError("output_features['action'] must have a shape attribute with at least one dimension")

    return int(action_ft.shape[0])


def _infer_n_action_steps(policy: DiffusionPolicy) -> int:
    cfg = policy.config
    if hasattr(cfg, "n_action_steps") and cfg.n_action_steps is not None:
        return int(cfg.n_action_steps)
    # Default used in older diffusion configs
    return 100


def _infer_horizon(policy: DiffusionPolicy) -> int:
    cfg = policy.config
    if hasattr(cfg, "horizon") and cfg.horizon is not None:
        return int(cfg.horizon)
    # Fallback: use n_action_steps if horizon is missing
    return _infer_n_action_steps(policy)


def _prepare_global_conditioning(policy: DiffusionPolicy, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    public_prepare = getattr(policy.diffusion, "prepare_global_conditioning", None)
    if callable(public_prepare):
        return public_prepare(obs)

    private_prepare = getattr(policy.diffusion, "_prepare_global_conditioning", None)
    if callable(private_prepare):
        return private_prepare(obs)

    raise AttributeError("DiffusionPolicy.diffusion does not expose a conditioning prepare method")


def _build_deterministic_observation(
    *,
    policy: DiffusionPolicy,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build deterministic observation inputs that match the policy config.

    This is intentionally close to dp/pytorch_to_onnx/dp_unet_convert.py:
    - Use fixed constants (not random noise) for stability across platforms.
    - Provide both per-camera keys and the aggregated OBS_IMAGES key.
    """
    torch.manual_seed(int(seed))

    cfg = policy.config
    n_obs_steps = int(cfg.n_obs_steps)

    state_ft = cfg.robot_state_feature
    if state_ft is None:
        raise ValueError("DiffusionPolicy config is missing required input feature: observation.state")

    inputs: Dict[str, torch.Tensor] = {}
    state_dim = int(state_ft.shape[0])
    inputs[OBS_STATE] = torch.full((1, n_obs_steps, state_dim), 0.5, dtype=torch.float32, device=device)

    camera_keys = list(getattr(cfg, "image_features", {}).keys())
    if camera_keys:
        # All image shapes are validated to match by DiffusionConfig.validate_features().
        first_img_shape = cfg.image_features[camera_keys[0]].shape
        c, h, w = (int(first_img_shape[0]), int(first_img_shape[1]), int(first_img_shape[2]))

        for k in camera_keys:
            inputs[k] = (torch.full((1, n_obs_steps, c, h, w), 0.5, dtype=torch.float32, device=device) / 255.0)

        if len(camera_keys) == 1:
            inputs[OBS_IMAGE] = inputs[camera_keys[0]]
            # Important: OBS_IMAGES must include a camera dimension (n=1)
            inputs[OBS_IMAGES] = inputs[camera_keys[0]].unsqueeze(2)
        else:
            # Stack per-camera tensors into camera dimension at index 2: (B, s, n, C, H, W)
            inputs[OBS_IMAGES] = torch.stack([inputs[k] for k in camera_keys], dim=2)

    env_ft = cfg.env_state_feature
    if env_ft is not None:
        env_dim = int(env_ft.shape[0])
        inputs[OBS_ENV_STATE] = torch.full((1, n_obs_steps, env_dim), 0.5, dtype=torch.float32, device=device)

    if not camera_keys and env_ft is None:
        raise ValueError("DiffusionPolicy requires at least one image input or observation.environment_state")

    return inputs


class _UNetOnnxWrapper(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, sample: torch.Tensor, t: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        out = self.unet(sample, t, global_cond)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def export_onnx(
    *,
    wrapper: torch.nn.Module,
    sample: torch.Tensor,
    t: torch.Tensor,
    global_cond: torch.Tensor,
    onnx_output_path: Path,
    opset: int,
    do_constant_folding: bool,
) -> None:
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    args = (sample, t, global_cond)

    LOGGER.info("Exporting ONNX to %s", onnx_output_path)
    torch.onnx.export(
        wrapper,
        args,
        str(onnx_output_path),
        opset_version=int(opset),
        verbose=False,
        input_names=["sample", "t", "global_cond"],
        output_names=["model_output"],
        do_constant_folding=bool(do_constant_folding),
        export_params=True,
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        dynamo=False,
    )


def validate_onnx(
    *,
    wrapper: torch.nn.Module,
    sample: torch.Tensor,
    t: torch.Tensor,
    global_cond: torch.Tensor,
    onnx_output_path: Path,
    seed: int,
) -> None:
    # PyTorch output
    wrapper.eval()
    with torch.inference_mode():
        torch.manual_seed(int(seed))
        pytorch_out = wrapper(sample, t, global_cond)
    pytorch_np = pytorch_out.detach().float().cpu().numpy()

    # ONNXRuntime output
    os.environ["ORT_SEED"] = str(int(seed))
    onnx_inputs = {
        "sample": sample.detach().cpu().numpy(),
        "t": t.detach().cpu().numpy(),
        "global_cond": global_cond.detach().cpu().numpy(),
    }
    onnx_outputs, elapsed = _run_onnxruntime_cpu(onnx_output_path, onnx_inputs)
    onnx_out = np.asarray(onnx_outputs[0], dtype=np.float32)

    max_diff, mean_diff = _abs_diff_metrics(pytorch_np.astype(np.float32), onnx_out)
    cos_min, cos_max, cos_mean = _cosine_similarity_stats(pytorch_np.astype(np.float32), onnx_out)

    LOGGER.info("ONNX inference time (CPU): %.6f sec", elapsed)
    LOGGER.info("PyTorch output shape: %s", tuple(pytorch_out.shape))
    LOGGER.info("ONNX output shape: %s", tuple(onnx_out.shape))
    LOGGER.info("Max abs diff: %.6g", max_diff)
    LOGGER.info("Mean abs diff: %.6g", mean_diff)
    LOGGER.info("Cosine similarity (min/max/mean): %.6f / %.6f / %.6f", cos_min, cos_max, cos_mean)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export DiffusionPolicy UNet to ONNX and optionally validate.")
    p.add_argument(
        "--pretrained-policy-path",
        type=str,
        required=True,
        help="Local directory or HF Hub repo id containing config.json + model.safetensors",
    )
    p.add_argument(
        "--output",
        type=str,
        default="dp/eval/dp_onnx/unet.onnx",
        help="Output ONNX path",
    )
    p.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0")
    p.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    p.add_argument("--seed", type=int, default=42, help="Seed for dummy inputs")
    p.add_argument(
        "--constant-folding",
        dest="do_constant_folding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable torch.onnx constant folding (default: enabled)",
    )
    p.add_argument("--no-validate", action="store_true", help="Skip ONNXRuntime validation")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    policy_path = str(args.pretrained_policy_path)
    onnx_output_path = _normalize_path(args.output)
    device = _parse_device(args.device)

    LOGGER.info("Loading DiffusionPolicy from %s", policy_path)
    policy = DiffusionPolicy.from_pretrained(policy_path)
    policy.eval()
    policy = _move_policy_to_device(policy, device)

    obs = _build_deterministic_observation(policy=policy, seed=int(args.seed), device=device)
    with torch.inference_mode():
        global_cond = _prepare_global_conditioning(policy, obs)

    action_dim = _infer_action_dim(policy)
    horizon = _infer_horizon(policy)
    sample = torch.zeros((1, horizon, action_dim), dtype=torch.float32, device=device)
    t = torch.tensor([99], dtype=torch.long, device=device)

    wrapper = _UNetOnnxWrapper(policy.diffusion.unet)
    wrapper.eval()

    export_onnx(
        wrapper=wrapper,
        sample=sample,
        t=t,
        global_cond=global_cond,
        onnx_output_path=onnx_output_path,
        opset=int(args.opset),
        do_constant_folding=bool(args.do_constant_folding),
    )
    LOGGER.info("ONNX export finished")

    if not args.no_validate:
        LOGGER.info("Validating ONNX output vs PyTorch (CPU ORT)...")
        validate_onnx(
            wrapper=wrapper,
            sample=sample,
            t=t,
            global_cond=global_cond,
            onnx_output_path=onnx_output_path,
            seed=int(args.seed),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
