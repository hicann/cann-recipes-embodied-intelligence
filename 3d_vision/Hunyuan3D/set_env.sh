#!/bin/bash

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJ_DIR="$(cd "${SCRIPT_PATH}/../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${SCRIPT_PATH}:${PROJ_DIR}:${PYTHONPATH}"
