# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
CUR_DIR=$(dirname $(readlink -f $0))
SCRIPTS_DIR=${CUR_DIR}/scripts
BUILD_PACKAGES_DIR=${CUR_DIR}/meta_gauss_render/packages
SUPPORTED_PY_VERSION=(3.7 3.8 3.9 3.10 3.11)
PY_VERSION='3.7'
SINGLE_OP=''
BUILD_TYPE='Release'

function check_python_version() {
    matched_py_version='false'
    for ver in ${SUPPORTED_PY_VERSION[*]}; do
        if [ "${PY_VERSION}" = "${ver}" ]; then
            matched_py_version='true'
            return 0
        fi
    done
    if [ "${matched_py_version}" = 'false' ]; then echo "${PY_VERSION} is an unsupported python version, we suggest ${SUPPORTED_PY_VERSION[*]}"
        exit 1
    fi
}
function usage() {
    echo "Usage: $0 --python=3.7 [--single_op=xxx] [--debug]" 1>&2
}
function parse_script_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --python=*)
                PY_VERSION="${1#*=}"
                shift 1
                ;;
            --single_op=*)
                SINGLE_OP="${1#*=}"
                shift 1
                ;;
            --debug)
                BUILD_TYPE='Debug'
                shift 1
                ;;
            *)
            usage
            return 1
            ;;
        esac
    done
    return 0
}

function main()
{
    if [ -z "$ASCEND_OPP_PATH" ]; then
        echo "ImportError: libhccl.so: cannot open shared object file: No such file or directory. Please check that the cann package is installed. Please run 'source set_env.sh' in the CANN installation path."
        exit 1
    else
        echo "ASCEND_OPP_PATH = $ASCEND_OPP_PATH"
    fi

    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi

    check_python_version

    chmod -R 777 ${SCRIPTS_DIR}
    export BUILD_PYTHON_VERSION=${PY_VERSION}
    rm -rf ${BUILD_PACKAGES_DIR}

    python"${PY_VERSION}" setup.py bdist_wheel
    if [ $? != 0 ]; then
        echo "Failed to compile the wheel file. Please check the source code by yourself."
        exit 1
    fi

    exit 0
}

main "$@"
