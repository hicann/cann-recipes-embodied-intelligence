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

import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, '../../'))
sys.path.append(root_dir)

from adaptor_patches import dit_patch
from adaptor_patches import gr00t_n1d6_patch
from adaptor_patches import gr00t_policy_patch
from adaptor_patches import modeling_siglip2_patch