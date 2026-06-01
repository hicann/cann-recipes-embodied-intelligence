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

cmake_minimum_required(VERSION 3.19.0)
project(meta_gauss_render)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/kernels)
add_subdirectory(${PROJECT_SOURCE_DIR}/meta_gauss_render/csrc)