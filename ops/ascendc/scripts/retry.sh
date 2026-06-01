#!/bin/bash
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

MAX_RETRIES=3
RETRY_DELAY=2
COUNTER=0

while [ $COUNTER -lt $MAX_RETRIES ]; do
  eval "$@" && break
  COUNTER=$((COUNTER+1))
  if [ $COUNTER -lt $MAX_RETRIES ]; then
    echo "Command failed. Retrying in $RETRY_DELAY seconds..."
    sleep $RETRY_DELAY
  else
    echo "Command failed after $COUNTER attempts."
    exit 1
  fi
done

