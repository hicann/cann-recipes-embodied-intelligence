# -*- coding: utf-8 -*-
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

"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""
import json
import os
import sys
import stat
import const_var


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(sys.argv)
        print('argv error, inert_op_info.py your_op_file lib_op_file')
        sys.exit(2)

    with open(sys.argv[1], 'r') as load_f:
        insert_operator = json.load(load_f)

    all_operators = {}
    if os.path.exists(sys.argv[2]):
        if os.path.getsize(sys.argv[2]) != 0:
            with open(sys.argv[2], 'r') as load_f:
                all_operators = json.load(load_f)

    for k in insert_operator.keys():
        if k in all_operators.keys():
            print('replace op:[', k, '] success')
        else:
            print('insert op:[', k, '] success')
        all_operators[k] = insert_operator[k]

    with os.fdopen(os.open(sys.argv[2], const_var.WFLAGS, const_var.WMODES), 'w') as json_file:
        json_file.write(json.dumps(all_operators, indent=4))
