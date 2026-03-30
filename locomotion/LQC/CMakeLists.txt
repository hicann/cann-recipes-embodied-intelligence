# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)
project(lltk)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lltk/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lltk/bin)

# Dependencies
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(FastNoise2 CONFIG REQUIRED)

# CMake optimization options
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DLLTK_DEBUG_MODE=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -DEIGEN_NO_DEBUG")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

if (NOT DEFINED LLTK_BACKEND)
    set(LLTK_BACKEND "mujoco")
endif ()
option(LLTK_BUILD_DEBUGGER "build debugger for lltk" OFF)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGENRAND_BUILD_TEST OFF)
set(EIGENRAND_BUILD_BENCHMARK OFF)
add_subdirectory(extern/EigenRand)

# Build and link Python interface
add_library(lltk SHARED
    lltk/src/proto/actuator.cpp
    lltk/src/proto/biped_env.cpp
    lltk/src/proto/configurations.cpp
    lltk/src/proto/expression.cpp
    lltk/src/proto/legged_env.cpp
    lltk/src/proto/quadruped_env.cpp
    lltk/src/proto/terrain.cpp
    lltk/src/utils/extensions.cpp
    lltk/src/utils/math.cpp
    lltk/src/utils/reward.cpp
    lltk/src/utils/rhythm.cpp
    lltk/src/utils/symmetry.cpp
    lltk/src/vector/vector_gym_env.cpp
    lltk/src/vector/vector_quadruped_env.cpp
)
target_include_directories(lltk PUBLIC
    lltk/include
    extern
)
target_link_libraries(lltk PUBLIC
    atomic
    Eigen3::Eigen
    EigenRand::EigenRand
    FastNoise2::FastNoise
    OpenMP::OpenMP_CXX
    yaml-cpp
)
target_compile_definitions(lltk PUBLIC
    EIGEN_DONT_PARALLELIZE
    LLTK_BUILD_TYPE=${CMAKE_BUILD_TYPE}
)

# Choose simulation engine: raisim / mujoco
if ("${LLTK_BACKEND}" STREQUAL "raisim")
    message(STATUS "lltk: RaiSim as backend")
    target_compile_definitions(lltk PUBLIC LLTK_BACKEND=LLTK_RAISIM)
    set(RAISIM_OS linux)
    find_package(raisim CONFIG REQUIRED)
    target_link_libraries(lltk PUBLIC raisim::raisim)
elseif ("${LLTK_BACKEND}" STREQUAL "mujoco")
    message(STATUS "lltk: MuJoCo as backend")
    target_compile_definitions(lltk PUBLIC LLTK_BACKEND=LLTK_MUJOCO)
    set(OOMJ_BUILD_EXAMPLE ${LLTK_BUILD_DEBUGGER})
    add_subdirectory(extern/oomj)
    target_link_libraries(lltk PUBLIC oomj::oomj)
else ()
    message(FATAL_ERROR "Unknown backend: ${LLTK_BACKEND}")
endif ()

# Choose whether to build actuator network
if (DEFINED ORT_DIR)
    message(STATUS "Building with actuator network")
    target_compile_definitions(lltk PUBLIC LLTK_BUILD_AN)
    target_include_directories(lltk PUBLIC ${ORT_DIR}/include)
    target_link_libraries(lltk PUBLIC ${ORT_DIR}/lib/libonnxruntime.so)
endif ()

# OpenMP parallelism options
foreach (omp_option OMP_SCHEDULE OMP_LW_SCHEDULE)
    if (DEFINED ${omp_option})
        message(STATUS "${omp_option}: ${${omp_option}}")
        target_compile_definitions(lltk PUBLIC ${omp_option}=${${omp_option}})
    endif ()
endforeach ()

if (${LLTK_BUILD_DEBUGGER})
    message(STATUS "Build debugger for lltk")
    add_executable(lltk_debugger lltk/src/debugger/debugger.cpp)
    target_link_libraries(lltk_debugger PRIVATE lltk)
endif ()

pybind11_add_module(lltk_py
    lltk/src/binding/lltk.cpp
    lltk/src/binding/extensions.cpp
    lltk/src/binding/submodules.cpp
)
target_link_libraries(lltk_py PRIVATE lltk)


find_package(glfw3 REQUIRED)
include_directories(${GLFW3_INCLUDE_DIRS})
target_link_libraries(lltk PUBLIC ${GLFW3_LIBRARIES})
