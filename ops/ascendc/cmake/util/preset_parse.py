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

import json
import sys
import os


def get_config_opts(file):
    src_dir = os.path.abspath(os.path.dirname(file))
    opts = ''
    with open(file, 'r') as fd:
        config = json.load(fd)
        for conf in config:
            if conf == 'configurePresets':
                for node in config[conf]:
                    macros = node.get('cacheVariables')
                    if macros is not None:
                        for key in macros:
                            opts += '-D{}={} '.format(key, macros[key]['value'])
    opts = opts.replace('${sourceDir}', src_dir)
    print(opts)


if __name__ == "__main__":
    get_config_opts(sys.argv[1])
