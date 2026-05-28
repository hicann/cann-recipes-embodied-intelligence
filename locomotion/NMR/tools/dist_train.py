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
import os
import sys
import subprocess


base_dir = os.path.dirname(os.path.abspath(__file__))  # .../NMR/tools
sh_path = os.path.join(base_dir, "dist_train.sh")

if not os.path.isfile(sh_path):
    raise FileNotFoundError(f"dist_train.sh not found: {sh_path}")


def main() -> int:
    ret = subprocess.run([sh_path, *sys.argv[1:]], check=False)
    return ret.returncode


if __name__ == "__main__":
    raise SystemExit(main())