#!/usr/bin/env python3
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

import argparse
import logging
import os
import sys

import pybind11


class BuildError(Exception):
    """Custom exception for build errors."""
    pass


def check_conda_prefix():
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        raise BuildError(
            'Error: CONDA_PREFIX environment variable not detected. '
            'Please activate the lltk conda environment first!'
        )
    return conda_prefix


def run_command(command):
    """Run a shell command and return the exit code."""
    return (os.system(command) & 0xFF00) >> 8


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    conda_prefix = check_conda_prefix()

    os.environ['mujoco_DIR'] = conda_prefix
    os.environ['FastNoise2_DIR'] = conda_prefix
    os.environ['glfw3_DIR'] = os.path.join(conda_prefix, 'lib', 'cmake', 'glfw3')

    parser = argparse.ArgumentParser()
    parser.add_argument('-cc', help='C Compiler')
    parser.add_argument('-cxx', help='C++ Compiler')
    parser.add_argument('-cv', '--compiler-version', type=int, help='Compiler version')
    parser.add_argument('--clang', action='store_true', help='Use clang as compiler')
    parser.add_argument('-b', '--build-type', default='Release', help='CMAKE_BUILD_TYPE (default: Release)')
    parser.add_argument('--omp-schedule', default='nonmonotonic:guided', help='Macro OMP_SCHEDULE')
    parser.add_argument('--omp-lw-schedule', default='static,1', help='Macro OMP_LW_SCHEDULE')
    parser.add_argument('--backend', default='mujoco', help='Simulation backend (raisim / mujoco)')
    parser.add_argument('--onnxruntime', help='Specify onnxruntime directory to build with actuator network')
    parser.add_argument('--cmake-args', default='', help='Additional cmake arguments')
    parser.add_argument('-j', type=int, default=min(os.cpu_count(), 16))
    parser.add_argument('--compiler-args', default='', help='Additional compiler arguments')
    parser.add_argument('--incremental', action='store_true', help='Incremental build')
    parser.add_argument('--build-debugger', action='store_true', help='Build debugger')
    args = parser.parse_args()

    project_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(project_dir)

    if args.incremental:
        exit_code = run_command(f"cmake . -Bbuild && cmake --build build -- -j{args.j}")
        return exit_code

    cmake_rpath = f'{conda_prefix}/lib'

    cmake_args = [
        f'-Dpybind11_DIR={pybind11.get_cmake_dir()}',
        f'-DCMAKE_BUILD_TYPE={args.build_type}',
        f'-DOMP_SCHEDULE={args.omp_schedule}',
        f'-DOMP_LW_SCHEDULE={args.omp_lw_schedule}',
        f'-DLLTK_BACKEND={args.backend}',
        f'-DLLTK_BUILD_DEBUGGER={args.build_debugger}',
        f'-Dglfw3_DIR={os.environ["glfw3_DIR"]}',
        f'-DCMAKE_CXX_FLAGS="-fPIC -march=armv8-a -O3 -DNDEBUG -Wno-dev"',
        f'-DCMAKE_C_FLAGS="-fPIC -march=armv8-a -O3 -DNDEBUG -Wno-dev"',
        f'-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON',
        f'-DCMAKE_INSTALL_RPATH={cmake_rpath}',
    ]

    if args.backend == 'mujoco':
        cmake_args += [
            f'-DOOMJ_BUILD_EXAMPLE={args.build_debugger}',
            f'-DOOMJ_BUILD_GENERATE_SIMPLIFIED_MODEL={args.build_debugger}'
        ]
    if args.clang:
        args.cc = 'clang'
        args.cxx = 'clang++'
    if args.compiler_version is not None:
        if args.cc is None:
            args.cc = 'gcc'
        if args.cxx is None:
            args.cxx = 'g++'
        args.cc += f'-{args.compiler_version}'
        args.cxx += f'-{args.compiler_version}'

    if args.cc is not None:
        cmake_args.append(f'-DCMAKE_C_COMPILER={args.cc}')
    if args.cxx is not None:
        cmake_args.append(f'-DCMAKE_CXX_COMPILER={args.cxx}')
    if args.onnxruntime is not None:
        cmake_args.append(f'-DORT_DIR={args.onnxruntime}')
    if args.cmake_args is not None:
        cmake_args.extend(args.cmake_args.split())

    logger.info(
        '─' * 10 + ' [CMake Options] ' + '─' * 10 +
        ''.join(['\n ' + arg for arg in cmake_args]) + '\n' +
        '─' * 36 + '\n'
    )

    exit_code = run_command(
        f'cmake . -Bbuild -GNinja {" ".join(cmake_args)} && '
        f'cmake --build build -- -j{args.j}'
    )
    return exit_code


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except BuildError as e:
        logging.error(str(e))
        sys.exit(-1)