#!/usr/bin/env bash
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
# Orchestrate PI0 vlm and action_expert ONNX export with matching runtime artifacts.
# Usage:
#   ./script/run_pi0_export.sh \
#     --pretrained-policy-path ./models/pi0/pytorch \
#     --vlm-output outputs/onnx/pi0-vlm.onnx \
#     --action_expert-output outputs/onnx/pi0-action_expert.onnx \
#     [--runtime-save-dir runtime_save] [--device cpu] [--extra-vlm "--opset 14"] [--extra-action_expert "--lang-len 48"]

set -euo pipefail

PRETRAINED="./models/pi0/pytorch"
PART1_OUT="outputs/onnx/pi0-vlm.onnx"
PART2_OUT="outputs/onnx/pi0-action_expert.onnx"
RUNTIME_SAVE="runtime_save"
DEVICE="cpu"
EXTRA_PART1=""
EXTRA_PART2=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pretrained-policy-path)
      PRETRAINED="$2"; shift 2 ;;
    --vlm-output)
      PART1_OUT="$2"; shift 2 ;;
    --action_expert-output)
      PART2_OUT="$2"; shift 2 ;;
    --runtime-save-dir)
      RUNTIME_SAVE="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --extra-vlm)
      EXTRA_PART1="$2"; shift 2 ;;
    --extra-action_expert)
      EXTRA_PART2="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,20p' "$0"; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "[vlm] exporting to $PART1_OUT (runtime_save_dir=$RUNTIME_SAVE)"
python -u convert_verify_onnx_vlm.py \
  --pretrained-policy-path "$PRETRAINED" \
  --output "$PART1_OUT" \
  --runtime-save-dir "$RUNTIME_SAVE" \
  --device "$DEVICE" \
  $EXTRA_PART1

echo "[action_expert] exporting to $PART2_OUT (loading runtime tensors from $RUNTIME_SAVE)"
python -u convert_verify_onnx_action_expert.py \
  --pretrained-policy-path "$PRETRAINED" \
  --output "$PART2_OUT" \
  --past-kv-path "$RUNTIME_SAVE/past_kv_tensor.pth" \
  --prefix-pad-masks-path "$RUNTIME_SAVE/prefix_pad_masks.pth" \
  --device "$DEVICE" \
  $EXTRA_PART2

echo "Done. Part1 -> $PART1_OUT, Part2 -> $PART2_OUT"
