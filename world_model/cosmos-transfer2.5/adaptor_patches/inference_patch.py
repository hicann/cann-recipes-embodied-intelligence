# Adapted from
# https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2._src.transfer2.inference.utils import compile_tokenizer_if_enabled
from cosmos_transfer2.config import (
    MODEL_CHECKPOINTS,
    InferenceArguments,
    ModelKey,
    SetupArguments,
    is_rank0,
    path_to_str,
)
from cosmos_transfer2.inference import Control2WorldInference


def init(
    self,
    args: SetupArguments,
    batch_hint_keys: list[str],
) -> None:
    log.debug(f"{args.__class__.__name__}({args})({batch_hint_keys})")
    self.setup_args = args
    self.batch_hint_keys = batch_hint_keys
    if len(self.batch_hint_keys) == 1:
        # pyrefly: ignore  # bad-argument-type
        checkpoint = MODEL_CHECKPOINTS[ModelKey(variant=self.batch_hint_keys[0])]
        self.checkpoint_list = [checkpoint.path]
        self.experiment = checkpoint.experiment
    else:
        # pyrefly: ignore  # bad-argument-type
        self.checkpoint_list = [MODEL_CHECKPOINTS[ModelKey(variant=key)].path for key in self.batch_hint_keys]
        self.experiment = "multibranch_720p_t24_spaced_layer4_cr1pt1_rectified_flow_inference"

    log.debug(f"Loading keys for batch hints {self.batch_hint_keys=}")
    torch.enable_grad(False)  # Disable gradient calculations for inference

    self.device_rank = 0
    cfg_parallel = args.enable_cfg_parallel
    process_group = None
    # pyrefly: ignore  # unsupported-operation
    if args.context_parallel_size > 1:
        from megatron.core import parallel_state

        distributed.init()

        # pyrefly: ignore  # bad-argument-type
        if cfg_parallel:
            parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size // 2)
        else:
            parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
        
        process_group = parallel_state.get_context_parallel_group()

    if args.enable_guardrails and self.device_rank == 0:
        self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
            offload_model_to_cpu=args.offload_guardrail_models
        )
        self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
            offload_model_to_cpu=args.offload_guardrail_models
        )
    else:
        # pyrefly: ignore  # bad-assignment
        self.text_guardrail_runner = None
        # pyrefly: ignore  # bad-assignment
        self.video_guardrail_runner = None

    self.benchmark_timer = misc.TrainingTimer()
    # Initialize the inference class
    self.inference_pipeline = ControlVideo2WorldInference(
        registered_exp_name=EXPERIMENTS[self.experiment].registered_exp_name,
        checkpoint_paths=self.checkpoint_list,
        s3_credential_path="",
        exp_override_opts=EXPERIMENTS[self.experiment].command_args,
        process_group=process_group,
        use_cp_wan=args.enable_parallel_tokenizer,
        wan_cp_grid=args.parallel_tokenizer_grid,
        benchmark_timer=self.benchmark_timer if args.benchmark else None,
        cfg_parallel=cfg_parallel,
    )

    compile_tokenizer_if_enabled(self.inference_pipeline, args.compile_tokenizer.value)

    if self.device_rank == 0:
        log.info(f"Found {len(self.batch_hint_keys)} hint keys across all samples")
        if len(self.batch_hint_keys) > 1:
            log.warning(
                "Loading the multicontrol model. Multicontrol inference is not strictly equal to single control"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = args.output_dir / "config.yaml"
        # pyrefly: ignore  # bad-argument-type
        LazyConfig.save_yaml(self.inference_pipeline.config, config_path)
        log.info(f"Saved config to {config_path}")


Control2WorldInference.__init__ = init
