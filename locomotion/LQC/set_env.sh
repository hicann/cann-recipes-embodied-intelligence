#!/bin/bash
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


set -e # Exit immediately if any command fails

# --- 1. Basic Configuration ---
export CONDA_INSTALL_PATH=$HOME/miniconda3
export CMAKE_VERSION=3.30.0
export HUMANOID_DIR=$(pwd)
export CONDA_ENV_NAME="lltk"

# --- 2. Privilege & Package Manager Detection ---
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

if command -v apt &> /dev/null; then
    PM="apt"
    $SUDO apt update
elif command -v dnf &> /dev/null; then
    PM="dnf"
elif command -v yum &> /dev/null; then
    PM="yum"
else
    echo "Unsupported package manager. Please install dependencies manually."
    exit 1
fi

# --- 3. Helper Function: Add Line to bashrc ---
add_to_bashrc() {
    local line="$1"
    if ! grep -Fxq "$line" ~/.bashrc; then
        echo "$line" >> ~/.bashrc
    fi
}

# --- 4. Install System Dependencies ---
echo "Installing system dependencies ($PM)..."
if [ "$PM" = "apt" ]; then
    $SUDO apt install -y ninja-build libeigen3-dev libyaml-cpp-dev wget curl git build-essential \
                         libx11-dev libgl1-mesa-dev libglu1-mesa-dev libxext-dev libxinerama-dev \
                         libxcursor-dev libxi-dev libxrandr-dev wayland-protocols libwayland-dev libxkbcommon-dev
else
    # For EulerOS (dnf/yum)
    $SUDO $PM install -y ninja-build eigen3-devel yaml-cpp-devel wget curl git gcc-c++ make \
                         libX11-devel mesa-libGL-devel mesa-libGLU-devel libXext-devel libXinerama-devel \
                         libXcursor-devel libXi-devel libXrandr-devel wayland-devel libxkbcommon-devel
fi

# --- 5. Install CMake (AArch64) ---
if ! command -v cmake &> /dev/null || [[ "$(cmake --version)" != *"$CMAKE_VERSION"* ]]; then
    echo "Installing CMake $CMAKE_VERSION..."
    wget --no-check-certificate https://cmake.org/files/v${CMAKE_VERSION%.*}/cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz
    tar -zxvf cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz
    $SUDO mv cmake-${CMAKE_VERSION}-linux-aarch64 /usr/local/
    add_to_bashrc "export PATH=/usr/local/cmake-${CMAKE_VERSION}-linux-aarch64/bin:\$PATH"
    export PATH=/usr/local/cmake-${CMAKE_VERSION}-linux-aarch64/bin:$PATH
fi

# --- 6. Install Miniconda ---
if [ ! -d "$CONDA_INSTALL_PATH" ]; then
    echo "Installing Miniconda..."
    wget --no-check-certificate https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
    bash miniconda.sh -b -p $CONDA_INSTALL_PATH
    add_to_bashrc "export PATH=$CONDA_INSTALL_PATH/bin:\$PATH"
    export PATH=$CONDA_INSTALL_PATH/bin:$PATH
    conda init bash
fi

# Configure Conda Tsinghua Mirror
cat > ~/.condarc << EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# --- 7. Create & Configure Conda Environment ---
echo "Setting up Conda environment: $CONDA_ENV_NAME..."
source "$CONDA_INSTALL_PATH/etc/profile.d/conda.sh"

conda create -n $CONDA_ENV_NAME python=3.11 -y || true
conda activate $CONDA_ENV_NAME

conda install -c conda-forge libgcc-ng=14.2.0 libgomp=14.2.0 libstdcxx-ng=14.2.0 -y
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda install -y pyqt=5.15 -c conda-forge

# --- 8. Install & Configure MuJoCo ---
echo "Installing MuJoCo..."
conda install -c conda-forge mujoco=3.2.7 -y
add_to_bashrc "export MUJOCO_PATH=\$CONDA_PREFIX"
add_to_bashrc "export LD_LIBRARY_PATH=\$MUJOCO_PATH/lib:\$LD_LIBRARY_PATH"
add_to_bashrc "export CPATH=\$MUJOCO_PATH/include:\$CPATH"
add_to_bashrc "export CMAKE_PREFIX_PATH=\$MUJOCO_PATH:\$CMAKE_PREFIX_PATH"

# --- 9. Build & Install GLFW ---
echo "Building and installing GLFW..."
cd $HUMANOID_DIR
if [ ! -d "glfw" ]; then
    git clone https://github.com/glfw/glfw.git
fi
cd glfw && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DBUILD_SHARED_LIBS=ON \
         -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF -DGLFW_PLATFORM=X11
make -j$(nproc)
make install
add_to_bashrc "export PKG_CONFIG_PATH=\"\$CONDA_PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH\""
add_to_bashrc "export glfw3_DIR=\"\$CONDA_PREFIX/lib/cmake/glfw3\""

# --- 10. Build & Install FastNoise2 (AArch64 Optimized) ---
echo "Building and installing FastNoise2..."
cd $HUMANOID_DIR
if [ ! -d "FastNoise2" ]; then
    git clone https://github.com/Auburn/FastNoise2.git
fi
cd FastNoise2
git checkout ba93f17ec40a9d09066c8d07b3e72b789e5b5657

# Apply Modifications
sed -i 's/option(FASTNOISE2_NOISETOOL "Build NoiseTool application" ${FASTNOISE2_STANDALONE_PROJECT})/option(FASTNOISE2_NOISETOOL "Build NoiseTool application" OFF)/g' CMakeLists.txt
echo 'set_property(TARGET FastNoise PROPERTY POSITION_INDEPENDENT_CODE ON)' >> src/CMakeLists.txt
sed -i 's/option(FASTNOISE2_TOOLS "Build \"Node Editor\" executable" OFF)/option(FASTNOISE2_TOOLS "Build \\"Node Editor\\" executable" OFF)/g' CMakeLists.txt
sed -i '/-mno-vzeroupper/s/^/#/' src/CMakeLists.txt

mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-march=armv8-a -O3 -DNDEBUG -std=gnu++17 -fPIC -Wno-ignored-attributes -fno-stack-protector -ffast-math" \
         -DCMAKE_C_FLAGS="-march=armv8-a -O3 -DNDEBUG -fPIC" -DFASTNOISE2_TOOLS=OFF -DBUILD_SHARED_LIBS=ON

# Fix ARM NEON Header
sed -i '39s/return a.GetNative();/return FS::Register<U, 4, SIMD>(reinterpret_cast<typename FS::Register<U, 4, SIMD>::NativeType>(a.GetNative()));/' _deps/fastsimd-src/include/FastSIMD/ToolSet/ARM/NEON.h

make -j$(nproc)
make install

# Set FastNoise2 Environment Variables
add_to_bashrc "export CPATH=\$CONDA_PREFIX/include:\$CPATH"
add_to_bashrc "export LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LIBRARY_PATH"
add_to_bashrc "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
add_to_bashrc "export FastNoise2_DIR=\$CONDA_PREFIX"

# --- 11. Completed ---
echo "-------------------------------------------------------"
echo "Environment setup completed successfully!"
echo "Please run: source ~/.bashrc"
echo "Then run: conda activate $CONDA_ENV_NAME"
echo "-------------------------------------------------------"