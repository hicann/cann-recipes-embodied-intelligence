# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/65042cc501d1cdbefaf1d6f61a9a47575eec8c71/examples/benchmarks/basic.sh
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# The code snippet comes from gsplat.
#
# Copyright (c) 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

for SCENE in $SCENE_LIST;
do
    DATA_FACTOR=4

    echo "Running $SCENE"

    # train without eval
    ASCEND_RT_VISIBLE_DEVICES=0 python train.py --eval_steps -1 --data_factor $DATA_FACTOR \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        ASCEND_RT_VISIBLE_DEVICES=0 python train.py --data_factor $DATA_FACTOR \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done


for SCENE in $SCENE_LIST;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done