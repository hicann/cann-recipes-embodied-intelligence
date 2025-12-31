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
"""Compare ACT ONNX vs Ascend OM outputs.

This script:
- Loads the LeRobot policy config.json to determine *all* input feature names and shapes.
- Generates deterministic dummy inputs (supports multiple image inputs like observation.images.top/front/...)
- Runs ONNXRuntime (CPU) inference and Ascend OM inference.
- Reports numerical error metrics.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from convert_utils import load_schema, make_dummy_numpy_inputs, output_error_metrics, run_onnxruntime_cpu


LOGGER = logging.getLogger(__name__)


def _import_acl() -> tuple[Any, Any, Any]:
    """Import Ascend ACL runtime modules lazily."""
    try:
        import acl  # type: ignore
        import acllite_utils as utils  # type: ignore
        from acllite_model import AclLiteModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing Ascend ACL dependencies (acl/acllite_utils/acllite_model). "
            "Run this script in the Ascend environment where those are available."
        ) from exc
    return acl, utils, AclLiteModel


class AclLiteResource:
    def __init__(self, device_id: int) -> None:
        self.device_id = device_id
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

    def __enter__(self) -> "AclLiteResource":
        self.init()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def infer_om_model(om_model_path: Path, *, schema, inputs: dict[str, np.ndarray], device_id: int) -> list[np.ndarray]:
    acl, utils, acl_lite_model_cls = _import_acl()
    _ = (acl, utils)

    with AclLiteResource(device_id=device_id):
        om_model = acl_lite_model_cls(str(om_model_path))
        LOGGER.info("Loaded OM model: %s", om_model_path)

        input_list = [inputs[k] for k in schema.input_keys]
        LOGGER.info("OM input order: %s", list(schema.input_keys))
        start = time.perf_counter()
        outputs = om_model.execute(input_list)
        elapsed = time.perf_counter() - start
        LOGGER.info("OM inference time: %.6f sec", elapsed)
        return [np.asarray(output) for output in outputs]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare ACT ONNX vs Ascend OM outputs")
    p.add_argument("--pretrained-policy-path", type=str, default=None, help="Path containing config.json")
    p.add_argument("--config", type=str, default=None, help="Explicit path to config.json")
    p.add_argument("--onnx-model-path", type=str, required=True, help="Path to ONNX model")
    p.add_argument("--om-model-path", type=str, required=True, help="Path to OM model")
    p.add_argument("--device-id", type=int, default=0, help="Ascend device id")
    p.add_argument("--seed", type=int, default=42, help="Seed for dummy inputs")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    loaded = load_schema(pretrained_policy_path=args.pretrained_policy_path, config=args.config)
    schema = loaded.schema
    LOGGER.info("Config path: %s", loaded.config_path)
    LOGGER.info("VISUAL inputs: %s", list(schema.visual_keys))

    onnx_model_path = Path(args.onnx_model_path).expanduser().resolve()
    om_model_path = Path(args.om_model_path).expanduser().resolve()

    inputs = make_dummy_numpy_inputs(schema, seed=int(args.seed))
    for k in schema.input_keys:
        LOGGER.info("Input %s shape=%s", k, tuple(inputs[k].shape))

    onnx_outputs, onnx_elapsed = run_onnxruntime_cpu(onnx_model_path, inputs)
    LOGGER.info("ONNX inference time: %.6f sec", onnx_elapsed)
    om_outputs = infer_om_model(om_model_path, schema=schema, inputs=inputs, device_id=int(args.device_id))

    metrics = output_error_metrics(onnx_outputs, om_outputs, include_cosine=True)
    LOGGER.info("===== ONNX vs OM error metrics =====")
    for m in metrics:
        idx = int(m["output_index"])
        LOGGER.info(
            "output_%d: max_abs=%.6g mean_abs=%.6g max_rel=%.6g mean_rel=%.6g cos(min/max/mean)=%.3f/%.3f/%.3f",
            idx,
            m["max_abs_error"],
            m["mean_abs_error"],
            m["max_rel_error"],
            m["mean_rel_error"],
            m["cosine_min"],
            m["cosine_max"],
            m["cosine_mean"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
