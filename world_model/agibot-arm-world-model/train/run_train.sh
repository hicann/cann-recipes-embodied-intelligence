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

export TORCHINDUCTOR_DISABLE=1
export NPU_DISABLE_TORCHINDUCTOR=1
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-21000}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29502}"

MODEL_DIR="${MODEL_DIR:-/path/to/Wan2.1-Fun-V1.1-1.3B-Control}"
DATA_ROOT="${DATA_ROOT:-/path/to/agibot_world_dataset}"
SPLIT_FILE="${SPLIT_FILE:-${DATA_ROOT}/split.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/agibot_arm_world_model}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

accelerate launch --main_process_port "${MASTER_PORT}" --num_processes "${NUM_PROCESSES}" \
    --config_file "${SCRIPT_DIR}/configs/accelerate_config_zero2.yaml" \
    "${SCRIPT_DIR}/train_fun_control_1_3b_text_49frames.py" \
    --model_paths "[\"${MODEL_DIR}/diffusion_pytorch_model.safetensors\",\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_DIR}/Wan2.1_VAE.pth\",\"${MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\"]" \
    --data_root "${DATA_ROOT}" \
    --height 480 \
    --width 640 \
    --num_frames 49 \
    --learning_rate 1e-5 \
    --num_epochs 100 \
    --save_steps 2000 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${OUTPUT_DIR}" \
    --trainable_models "dit" \
    --extra_inputs "control_video,reference_image,input_image" \
    --text_dropout_prob 0.1 \
    --gradient_accumulation_steps 2 \
    --split_file "${SPLIT_FILE}"
