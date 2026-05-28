# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
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
from setuptools import setup, find_packages

# Hardware compatibility note:
# - This project is tested on Ascend NPU + CANN environments.
# - torch.compile availability depends on your PyTorch build/backend.
#   If torch.compile is unavailable or unstable on your device, run without it.

setup(
    name="humanoid-retarget",
    version="0.1.0",
    description="Neural motion retargeting from SMPL-X to Unitree G1 humanoid robot",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "mmengine>=0.10.0",
        "numpy>=1.23,<2.0",
        "scipy",
        "joblib",
        "pin",
        "imageio[ffmpeg]",
        "opencv-python",
        "matplotlib",
        "tqdm",
    ],
)
