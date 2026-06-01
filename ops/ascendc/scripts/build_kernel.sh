#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# The code snippet comes from DrivingSDK (https://gitcode.com/Ascend/DrivingSDK).
#
# Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
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
#
# --------------------------------------------------------------------------------

script_path=$(realpath $(dirname $0))
root_path=$(realpath $script_path/..)
rm -rf build_out
mkdir build_out
cd build_out
SINGLE_OP=""
BUILD_TYPE="Release"

function parse_script_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --single_op=*)
                SINGLE_OP="${1#*=}"
                shift 1
                ;;
            --build_type=*)
                BUILD_TYPE="${1#*=}"
                shift 1
                ;;
            *)
              echo "Usage: $0 --single_op=xxx --build_type=xxx"
              return 1
            ;;
        esac
    done
    return 0
}

parse_script_args $@

cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')
if [ "$cmake_version" \< "3.19.0" ]; then
  opts=$(python3 $root_path/cmake/util/preset_parse.py $root_path/CMakePresets.json)
  echo $opts
  cmake .. $opts -DSINGLE_OP=$SINGLE_OP -DCMAKE_BUILD_TYPE=$BUILD_TYPE
else
  cmake .. --preset=default -DSINGLE_OP=$SINGLE_OP -DCMAKE_BUILD_TYPE=$BUILD_TYPE
fi

cmake --build . -j16
if [ $? -ne 0 ]; then exit 1; fi
