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

add_library(intf_pub INTERFACE)
target_compile_options(
  intf_pub
  INTERFACE
    -fPIC
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-O0
    -g>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-ftrapv
    -fstack-check>
    $<$<COMPILE_LANGUAGE:C>:-pthread
    -Wfloat-equal
    -Wshadow
    -Wformat=2
    -Wno-deprecated
    -Wextra>
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
)
target_compile_definitions(
  intf_pub INTERFACE _GLIBCXX_USE_CXX11_ABI=0
                     $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>)
target_include_directories(
  intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/include
                     ${PROJECT_SOURCE_DIR}/include)
# if the CANN_PATHS not empty
if(CANN_PATHS)
  # if the arch is aarch64, add the include path
  if(${ARCH} STREQUAL "aarch64")
    target_include_directories(intf_pub
                               INTERFACE ${CANN_PATHS}/aarch64-linux/include)
    target_link_directories(intf_pub INTERFACE
                            ${CANN_PATHS}/aarch64-linux/lib64)
  else()
    target_include_directories(intf_pub
                               INTERFACE ${CANN_PATHS}/x86_64-linux/include)
    target_link_directories(intf_pub INTERFACE ${CANN_PATHS}/x86_64-linux/lib64)
  endif()
endif()

target_link_options(
  intf_pub
  INTERFACE
  $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
  $<$<CONFIG:Release>:-s>
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack)
