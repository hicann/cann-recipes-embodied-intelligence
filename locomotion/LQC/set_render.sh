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

export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
chown root:root $XDG_RUNTIME_DIR

echo "export XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" >> /etc/profile
echo "export XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" >> ~/.bashrc
echo "export XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" >> /root/.bashrc

# 1. Permission and System Check
if [ "$EUID" -ne 0 ]; then
  SUDO="sudo"
else
  SUDO=""
fi

if ! command -v apt &> /dev/null; then
    echo "Error: This script currently only supports APT-based systems (Ubuntu/Debian)."
    exit 1
fi

echo "--- Starting X11/OpenGL Rendering Environment Configuration ---"

# 2. Interactive Input for Windows IP Address
read -p "Please enter your Windows host IP address: " win_ip

if [[ -z "$win_ip" ]]; then
    echo "Error: IP address cannot be empty."
    exit 1
fi

# 3. Install Basic Dependencies and MobaXterm Compatible Drivers
echo "Installing system dependencies..."
$SUDO apt update
$SUDO apt install -y libgl1-mesa-glx libglu1-mesa freeglut3-dev x11-utils wget x11-apps mesa-utils
$SUDO apt install -y libgl1-mesa-dri libegl1-mesa libgles2-mesa libglvnd0 libglvnd-dev libglx0 libegl1
$SUDO apt install -y libglfw3 libglfw3-dev

# 4. Configure Environment Variables
echo "Configuring environment variables..."

ENV_VARS=(
    "export LIBGL_ALWAYS_INDIRECT=1"
    "export DISPLAY=${win_ip}:0.0"
    "export __GLX_VENDOR_LIBRARY_NAME=mesa"
    "export LIBGL_ALWAYS_SOFTWARE=1"
    "export GALLIUM_DRIVER=llvmpipe"
    "export MESA_GL_VERSION_OVERRIDE=3.0"
    "export MESA_GLSL_VERSION_OVERRIDE=130"
)

for var in "${ENV_VARS[@]}"; do
    if ! grep -Fxq "$var" ~/.bashrc; then
        echo "$var" >> ~/.bashrc
    fi
    if ! grep -Fxq "$var" /root/.bashrc; then
        echo "$var" >> /root/.bashrc
    fi
    if ! grep -Fxq "$var" /etc/profile; then
        echo "$var" >> /etc/profile
    fi
    eval "$var"
done

echo "--- Environment Configuration Completed ---"
echo "Note: Environment variables have been written to ~/.bashrc and will take effect in new terminals."
echo "Note: Please ensure MobaXterm has X Server enabled and Access Control is disabled."

# 5. Test Section
echo "The test will start next..."

read -p "Run MuJoCo test (NOTE: Temporarily sets LIBGL_ALWAYS_INDIRECT=0)? (y/n): " confirm_mujoco
if [[ "$confirm_mujoco" == "y" ]]; then
    export LIBGL_ALWAYS_INDIRECT=0
    python3 -m mujoco.viewer
fi