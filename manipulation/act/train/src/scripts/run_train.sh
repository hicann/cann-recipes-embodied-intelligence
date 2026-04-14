#!/bin/bash
#
# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 Shanghai Jiao Tong University
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
set -euo pipefail

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000
export HOST_CACHE_CAPACITY=20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$WORKSPACE_ROOT/lerobot}"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORKSPACE_ROOT/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORKSPACE_ROOT/.cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$WORKSPACE_ROOT/.cache/huggingface/transformers}"
export TOKENIZERS_PARALLELISM=false

NPROC=8
MASTER_PORT=29500
MODEL_TYPE=""
CUSTOM_CONFIG=""
USE_RESUME=false
USE_MIXED_PRECISION=false
MIXED_PRECISION_TYPE="bf16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        --resume)
            USE_RESUME=true
            shift
            ;;
        --mix|--mixed|--mixed_precision)
            USE_MIXED_PRECISION=true
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                MIXED_PRECISION_TYPE="$2"
                shift 2
            else
                MIXED_PRECISION_TYPE="fp16"
                shift
            fi
            ;;
        -h|--help)
            echo "Usage: $0 [--config <path>] [<model_type>] [--nproc <num>] [--port <port>] [--resume] [--mix [fp16|bf16]]"
            echo "Examples:"
            echo "  $0 act_aloha_smoke"
            echo "  $0 act_aloha --resume --port 29510"
            echo "  $0 --config manipulation/act/train/src/configs/act_aloha.yaml"
            exit 0
            ;;
        *)
            if [[ -z "$MODEL_TYPE" ]]; then
                MODEL_TYPE="$1"
            else
                echo "Unknown option or too many positional args: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -n "$CUSTOM_CONFIG" ]]; then
    if [[ "$CUSTOM_CONFIG" = /* ]]; then
        CONFIG_PATH="$CUSTOM_CONFIG"
    else
        CONFIG_PATH="$RECIPE_REPO_ROOT/$CUSTOM_CONFIG"
    fi
elif [[ -n "$MODEL_TYPE" ]]; then
    CONFIG_PATH="$CONFIG_DIR/${MODEL_TYPE}.yaml"
else
    echo "Error: Either --config <path> or <model_type> must be provided."
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

if [[ ! -d "$LEROBOT_ROOT" ]]; then
    echo "LeRobot repo not found: $LEROBOT_ROOT"
    echo "Run: ./manipulation/act/train/src/scripts/setup.sh"
    exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun not found, please activate the proper environment."
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$WORKSPACE_ROOT/ckpt/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${MODEL_TYPE:-custom}_${TIMESTAMP}.log"

RAW_OUTPUT_DIR=$(awk '/^[[:space:]]*output_dir:/{gsub(/^[[:space:]]*output_dir:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")
RAW_JOB_NAME=$(awk '/^[[:space:]]*job_name:/{gsub(/^[[:space:]]*job_name:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")
OUTPUT_DIR_FINAL="${RAW_OUTPUT_DIR}_${TIMESTAMP}"
JOB_NAME_FINAL="${RAW_JOB_NAME}_${TIMESTAMP}"

TRAIN_ARGS=(--config_path="$CONFIG_PATH")
if [[ "$USE_RESUME" == true ]]; then
    TRAIN_ARGS+=(--resume=true)
fi

ACCELERATE_ARGS=(--main_process_port "$MASTER_PORT" --num_processes "$NPROC")
if (( NPROC > 1 )); then
    ACCELERATE_ARGS+=(--multi_gpu)
fi
if [[ "$USE_MIXED_PRECISION" == true ]]; then
    ACCELERATE_ARGS+=(--mixed_precision "$MIXED_PRECISION_TYPE")
fi

cd "$LEROBOT_ROOT"
nohup accelerate launch "${ACCELERATE_ARGS[@]}" \
    "$(command -v lerobot-train)" \
    "${TRAIN_ARGS[@]}" \
    --output_dir="$OUTPUT_DIR_FINAL" \
    --job_name="$JOB_NAME_FINAL" \
    > "$LOG_FILE" 2>&1 &
PID=$!

sleep 2
if ! ps -p "$PID" >/dev/null 2>&1; then
    echo "ACT training failed to stay alive after launch."
    echo "Check log file: $LOG_FILE"
    tail -n 40 "$LOG_FILE" || true
    exit 1
fi

echo "============================================="
echo "ACT training started"
echo "LeRobot root: $LEROBOT_ROOT"
echo "Config: $CONFIG_PATH"
echo "Log file: $LOG_FILE"
echo "Output dir: $OUTPUT_DIR_FINAL"
echo "Job name: $JOB_NAME_FINAL"
echo "Num processes: $NPROC"
echo "Master port: $MASTER_PORT"
echo "Resume: $USE_RESUME"
echo "Mixed precision: ${USE_MIXED_PRECISION:+$MIXED_PRECISION_TYPE}"
echo "PID: $PID"
echo "============================================="
