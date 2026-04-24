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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$WORKSPACE_ROOT/lerobot}"

export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

infer_policy_device() {
    local previous=""
    local arg=""

    for arg in "$@"; do
        if [[ "$previous" == "--policy.device" ]]; then
            printf '%s' "$arg"
            return 0
        fi
        case "$arg" in
            --policy.device=*)
                printf '%s' "${arg#--policy.device=}"
                return 0
                ;;
        esac
        previous="$arg"
    done

    return 1
}

usage() {
    echo "Usage: $0 <lerobot-eval args>"
    echo "Example:"
    echo "  $0 --policy.path=/path/to/pretrained_model --policy.device=npu --env.type=libero --env.task=libero_spatial --eval.n_episodes=100 --eval.batch_size=1 --output_dir=/path/to/eval_out"
}

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    if command -v lerobot-eval >/dev/null 2>&1; then
        "$(command -v lerobot-eval)" --help
    else
        usage
    fi
    exit 0
fi

if [[ ! -d "$LEROBOT_ROOT" ]]; then
    echo "LeRobot repo not found: $LEROBOT_ROOT"
    echo "Run: ./manipulation/smolvla/train/src/scripts/setup.sh"
    exit 1
fi

if ! command -v lerobot-eval >/dev/null 2>&1; then
    echo "lerobot-eval not found, please activate the proper environment."
    exit 1
fi

POLICY_DEVICE="$(infer_policy_device "$@" || true)"
if [[ -z "${LEROBOT_EVAL_NPU_DEVICE:-}" && "$POLICY_DEVICE" == npu* ]]; then
    export LEROBOT_EVAL_NPU_DEVICE="$POLICY_DEVICE"
fi

cd "$LEROBOT_ROOT"
"$(command -v lerobot-eval)" "$@"
