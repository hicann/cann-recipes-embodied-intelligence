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
"""Convert an ACT policy to ONNX and optionally validate outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch

try:
    from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: pydantic (v2). Install with: pip install 'pydantic>=2,<3' "
        "(or conda install -c conda-forge pydantic)."
    ) from exc

try:
    import onnx
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency: onnx. Please install it first.") from exc

from lerobot.policies.act.modeling_act import ACTPolicy
from convert_utils import (
    abs_diff_metrics,
    load_schema,
    make_dummy_torch_observation,
    run_onnxruntime_cpu,
    torch_obs_to_numpy_inputs,
)



LOGGER = logging.getLogger(__name__)


def _normalize_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _parse_device(device: str) -> torch.device:
    try:
        parsed = torch.device(device)
    except Exception as err:
        raise ValueError(f"Invalid --device '{device}'") from err
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            LOGGER.warning("No available CUDA device; falling back to CPU")
            return torch.device("cpu")

        # If a specific CUDA index was requested, ensure it exists.
        if parsed.index is not None:
            count = torch.cuda.device_count()
            if parsed.index < 0 or parsed.index >= count:
                LOGGER.warning(
                    "Requested CUDA device cuda:%s is not available (device_count=%s); falling back to CPU",
                    parsed.index,
                    count,
                )
                return torch.device("cpu")
    return parsed


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    pretrained_policy_path: Path = Field(
        ...,
        validation_alias=AliasChoices("pretrained_policy_path", "pretrained-policy-path", "pretrained_path"),
    )
    onnx_output_path: Path = Field(
        Path("outputs/onnx/act.onnx"),
        validation_alias=AliasChoices("output", "onnx_output_path", "onnx-output-path"),
    )
    device: torch.device = Field(torch.device("cpu"), validation_alias=AliasChoices("device",))
    opset: int = Field(14, validation_alias=AliasChoices("opset",))
    seed: int = Field(42, validation_alias=AliasChoices("seed",))
    input_keys: Tuple[str, ...]
    input_shapes: Dict[str, Tuple[int, ...]]
    save_external_data: bool = Field(False, validation_alias=AliasChoices("save_external_data", "save-external-data"))
    do_constant_folding: bool | None = Field(
        None,
        validation_alias=AliasChoices(
            "do_constant_folding",
            "do-constant-folding",
            "constant_folding",
            "constant-folding",
        ),
    )
    no_validate: bool = Field(False, validation_alias=AliasChoices("no_validate", "no-validate"))
    log_level: str = Field("INFO", validation_alias=AliasChoices("log_level", "log-level"))

    @field_validator("pretrained_policy_path", "onnx_output_path", mode="before")
    @classmethod
    def _normalize_paths(cls, v: Any) -> Any:
        if v is None:
            return v
        return _normalize_path(str(v))

    @field_validator("device", mode="before")
    @classmethod
    def _parse_torch_device(cls, v: Any) -> torch.device:
        if isinstance(v, torch.device):
            return v
        return _parse_device(str(v))

    @field_validator("opset", "seed")
    @classmethod
    def _positive_ints(cls, v: int, info) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v


def load_settings(args: argparse.Namespace) -> Settings:
    """Load config.json (required) and apply CLI overrides.

    Model input schema (keys + shapes) is always derived from config.json.
    """
    if not args.pretrained_policy_path:
        raise ValueError("Provide --pretrained-policy-path (policy weights directory)")

    loaded = load_schema(pretrained_policy_path=args.pretrained_policy_path, config=args.config)
    config: Dict[str, Any] = dict(loaded.config)
    config["input_keys"] = loaded.schema.input_keys
    config["input_shapes"] = loaded.schema.input_shapes
    LOGGER.info(
        "Using %d input(s) from config.json; VISUAL inputs: %s",
        len(loaded.schema.input_keys),
        list(loaded.schema.visual_keys),
    )

    # CLI overrides: only include args explicitly provided (non-None).
    overrides: Dict[str, Any] = {
        "pretrained_policy_path": args.pretrained_policy_path,
        "output": args.output,
        "device": args.device,
        "opset": args.opset,
        "seed": args.seed,
        "save_external_data": args.save_external_data,
        "do_constant_folding": args.do_constant_folding,
        "no_validate": args.no_validate,
        "log_level": args.log_level,
    }
    data = dict(config)
    data.update({k: v for k, v in overrides.items() if v is not None})

    # Respect requested log level during validation/default-logging.
    requested_level = str(data.get("log_level") or "INFO")
    logging.getLogger().setLevel(getattr(logging, requested_level.upper(), logging.INFO))

    settings = Settings.model_validate(data)

    # Log defaults (fields not provided via CLI/config).
    for field_name, field_info in Settings.model_fields.items():
        if field_name not in settings.model_fields_set and field_info.default is not None:
            LOGGER.info("%s not provided via CLI/config; using default: %r", field_name, getattr(settings, field_name))

    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return settings


def move_policy_to_device(policy: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Best-effort device move for policy and a few known ACT internals."""
    policy = policy.to(device)
    model = getattr(policy, "model", None)
    if model is not None:
        if hasattr(model, "vae_encoder_pos_enc"):
            model.vae_encoder_pos_enc = model.vae_encoder_pos_enc.to(device)
        if hasattr(model, "encoder_1d_feature_pos_embed") and hasattr(model.encoder_1d_feature_pos_embed, "weight"):
            model.encoder_1d_feature_pos_embed.weight = model.encoder_1d_feature_pos_embed.weight.to(device)

    # Ensure buffers follow.
    for module in policy.modules():
        for buf in module.buffers(recurse=False):
            if buf is not None and buf.device != device:
                buf.data = buf.data.to(device)
    return policy


def make_dummy_observation(
    *,
    input_keys: Tuple[str, ...],
    input_shapes: Dict[str, Tuple[int, ...]],
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create a deterministic dummy observation matching the policy input schema."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    obs: Dict[str, torch.Tensor] = {}
    for key in input_keys:
        shape = input_shapes[key]
        if len(shape) == 1:
            obs[key] = torch.randn((1, shape[0]), generator=gen, device=device, dtype=torch.float32)
        elif len(shape) == 3:
            c, h, w = shape
            obs[key] = torch.rand((1, c, h, w), generator=gen, device=device, dtype=torch.float32)
        else:
            obs[key] = torch.randn((1, *shape), generator=gen, device=device, dtype=torch.float32)
    return obs


class ONNXPolicyWrapper(torch.nn.Module):
    """Wraps a policy so torch.onnx.export sees a pure-tensor signature."""

    def __init__(self, policy: ACTPolicy, *, input_keys: Tuple[str, ...]) -> None:
        super().__init__()
        self.policy = policy
        self.input_keys = input_keys

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != len(self.input_keys):
            raise ValueError(f"Expected {len(self.input_keys)} inputs, got {len(inputs)}")
        observation = {k: v for k, v in zip(self.input_keys, inputs)}
        # Avoid `select_action`: it is stateful (action queue / temporal ensembling) and is not
        # reliably traceable/exportable. Instead, export a stateless forward.
        actions = self.policy.predict_action_chunk(observation)
        # `predict_action_chunk` returns (batch, chunk, action_dim). Export a single step.
        return actions


def export_onnx(
    *,
    policy: ACTPolicy,
    observation: Dict[str, torch.Tensor],
    input_keys: Tuple[str, ...],
    onnx_output_path: Path,
    opset: int,
    save_external_data: bool,
    do_constant_folding: bool | None,
) -> None:
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = tuple(observation[k] for k in input_keys)

    wrapper = ONNXPolicyWrapper(policy, input_keys=input_keys)
    wrapper.eval()

    LOGGER.info("Exporting ONNX to %s", onnx_output_path)
    torch.backends.cudnn.benchmark = False

    def _do_export(*, do_constant_folding: bool) -> None:
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_output_path),
            opset_version=opset,
            verbose=False,
            input_names=list(input_keys),
            output_names=["action"],
            do_constant_folding=do_constant_folding,
            export_params=True,
            keep_initializers_as_inputs=False,
            dynamo=False,
        )

    if do_constant_folding is None:
        # Auto mode: try constant folding, and fall back if a known mixed-device folding issue occurs.
        try:
            _do_export(do_constant_folding=True)
        except RuntimeError as err:
            msg = str(err)
            if "Expected all tensors to be on the same device" in msg:
                LOGGER.warning(
                    "ONNX export failed during constant folding due to mixed CPU/CUDA tensors; "
                    "retrying with do_constant_folding=False"
                )
                _do_export(do_constant_folding=False)
            else:
                raise
    else:
        LOGGER.info("Exporting with do_constant_folding=%s (explicit)", do_constant_folding)
        _do_export(do_constant_folding=bool(do_constant_folding))

    if save_external_data:
        # Only enable external data when explicitly requested.
        model = onnx.load(str(onnx_output_path), load_external_data=True)
        onnx.save_model(
            model,
            str(onnx_output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
        )


def validate_onnx(
    *,
    policy: ACTPolicy,
    observation: Dict[str, torch.Tensor],
    input_keys: Tuple[str, ...],
    onnx_output_path: Path,
    device: torch.device,
) -> None:
    """Run PyTorch + ONNXRuntime (CPU) and log basic abs-diff metrics."""
    if device.type == "cuda":
        LOGGER.info(f"ONNXRuntime validation uses CUDAExecutionProvider; diff computed on {device}")
    else:
        LOGGER.info(f"ONNXRuntime validation uses CPUExecutionProvider; diff computed on {device}")

    with torch.inference_mode():
        # Must match ONNX export behavior.
        pytorch_output = policy.predict_action_chunk(observation)
    # Always compare on CPU in float32 for stable metrics.
    pytorch_np = pytorch_output.detach().float().cpu().numpy()

    numpy_inputs = torch_obs_to_numpy_inputs({k: observation[k] for k in input_keys})
    onnx_outputs, elapsed = run_onnxruntime_cpu(onnx_output_path, numpy_inputs)
    onnx_output = np.asarray(onnx_outputs[0], dtype=np.float32)
    max_diff, mean_diff = abs_diff_metrics(pytorch_np, onnx_output)

    LOGGER.info("ONNX inference time: %.6f sec", elapsed)
    LOGGER.info("PyTorch output shape: %s", tuple(pytorch_output.shape))
    LOGGER.info("ONNX output shape: %s", tuple(onnx_output.shape))
    LOGGER.info("Max abs diff: %.6g", max_diff)
    LOGGER.info("Mean abs diff: %.6g", mean_diff)
    return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ACT policy to ONNX (and optionally validate).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to config.json. If omitted, will try '<pretrained_policy_path>/config.json' when available. "
            "CLI args take precedence over config.json."
        ),
    )
    parser.add_argument(
        "--pretrained-policy-path",
        type=str,
        default=None,
        help="Local path to ACT pretrained policy (no remote download).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version")
    parser.add_argument("--seed", type=int, default=None, help="Seed for dummy inputs")
    parser.add_argument(
        "--save-external-data",
        action="store_true",
        default=None,
        help="Save ONNX weights as external data (creates additional .data file).",
    )
    parser.add_argument(
        "--constant-folding",
        dest="do_constant_folding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable torch.onnx constant folding. "
            "If omitted, uses auto mode (tries folding and may retry without folding on known mixed-device errors)."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=None,
        help="Skip ONNXRuntime validation.",
    )
    parser.add_argument("--log-level", type=str, default=None, help="Logging level: DEBUG/INFO/WARNING/ERROR")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    # Ensure we can emit logs during config resolution.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = load_settings(args)
    except ValueError as err:
        LOGGER.error("%s", err)
        return 1

    LOGGER.info("Loading policy from %s", settings.pretrained_policy_path)
    policy = ACTPolicy.from_pretrained(str(settings.pretrained_policy_path))
    policy.reset()
    policy.eval()
    policy = move_policy_to_device(policy, settings.device)

    schema = SimpleNamespace(input_keys=settings.input_keys, input_shapes=settings.input_shapes)
    observation = make_dummy_torch_observation(schema, seed=settings.seed, device=settings.device)

    export_onnx(
        policy=policy,
        observation=observation,
        input_keys=settings.input_keys,
        onnx_output_path=settings.onnx_output_path,
        opset=settings.opset,
        save_external_data=settings.save_external_data,
        do_constant_folding=settings.do_constant_folding,
    )
    LOGGER.info("ONNX export finished")

    if not settings.no_validate:
        LOGGER.info("Validating ONNX output vs PyTorch...")
        validate_onnx(
            policy=policy,
            observation=observation,
            input_keys=settings.input_keys,
            onnx_output_path=settings.onnx_output_path,
            device=settings.device,
        )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
