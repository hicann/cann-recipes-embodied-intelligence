#!/bin/bash
# Copyright (c) 2026, Institute of Computing Technology, Chinese Academy of Sciences. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/step-18000.safetensors}"
MODEL_DIR="${MODEL_DIR:-/path/to/Wan2.1-Fun-V1.1-1.3B-Control}"
TEST_ROOT="${TEST_ROOT:-/path/to/test/info_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/inference}"
CFG_SCALE="${CFG_SCALE:-3.0}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CHUNK_SIZE="${CHUNK_SIZE:-49}"
DEVICE="${DEVICE:-npu}"
META_EMAIL="${META_EMAIL:-your_email@example.com}"

python "${SCRIPT_DIR}/infer_fun_control_1_3b_text.py" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --model_dir "${MODEL_DIR}" \
    --test_root "${TEST_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --cfg_scale "${CFG_SCALE}" \
    --only_mp4 \
    --seed "${SEED}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --chunk_size "${CHUNK_SIZE}" \
    --device "${DEVICE}" \
    --meta_email "${META_EMAIL}" \
    --save_mp4
