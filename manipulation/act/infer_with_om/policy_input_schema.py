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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PolicyInputSchema:
    """Input schema derived from a LeRobot policy config.json.

    - `input_keys` preserves the order in `config["input_features"]`.
    - `input_shapes` maps key -> shape tuple (without batch dim).
    - `visual_keys` is the subset of keys with type == "VISUAL".
    - `state_keys` is the subset of keys with type == "STATE".
    """

    input_keys: tuple[str, ...]
    input_shapes: dict[str, tuple[int, ...]]
    visual_keys: tuple[str, ...]
    state_keys: tuple[str, ...]


def normalize_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def load_config_json(config_path: Path) -> dict[str, Any]:
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read config.json at {config_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a JSON object")
    return data


def resolve_config_path(*, pretrained_policy_path: Path | None, explicit_config_path: Path | None) -> Path:
    if explicit_config_path is not None:
        return explicit_config_path
    if pretrained_policy_path is None:
        raise ValueError("Provide either explicit_config_path or pretrained_policy_path")
    return pretrained_policy_path / "config.json"


def extract_policy_input_schema(config: Mapping[str, Any]) -> PolicyInputSchema:
    input_features = config.get("input_features")
    if not isinstance(input_features, dict) or not input_features:
        raise ValueError("config.json must contain a non-empty 'input_features' object")

    input_keys: list[str] = []
    input_shapes: dict[str, tuple[int, ...]] = {}
    visual_keys: list[str] = []
    state_keys: list[str] = []

    for key, spec in input_features.items():
        if not isinstance(key, str) or not key:
            continue
        if not isinstance(spec, dict):
            continue

        shape = spec.get("shape")
        if not isinstance(shape, list) or not shape or not all(isinstance(x, int) and x > 0 for x in shape):
            raise ValueError(f"Invalid shape for input feature '{key}': expected list of positive ints")

        feature_type = spec.get("type")
        if feature_type == "VISUAL" and len(shape) != 3:
            raise ValueError(f"VISUAL feature '{key}' must have shape [C,H,W], got {shape}")
        if feature_type == "STATE" and len(shape) != 1:
            raise ValueError(f"STATE feature '{key}' must have shape [D], got {shape}")

        input_keys.append(key)
        input_shapes[key] = tuple(int(x) for x in shape)

        if feature_type == "VISUAL":
            visual_keys.append(key)
        if feature_type == "STATE":
            state_keys.append(key)

    if not visual_keys:
        raise ValueError("config.json input_features must include at least one VISUAL feature")
    if not state_keys:
        raise ValueError("config.json input_features must include at least one STATE feature")

    return PolicyInputSchema(
        input_keys=tuple(input_keys),
        input_shapes=input_shapes,
        visual_keys=tuple(visual_keys),
        state_keys=tuple(state_keys),
    )
