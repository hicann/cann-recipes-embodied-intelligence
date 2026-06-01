# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
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

import tyro
import torch

from gsplat.distributed import cli
from rasterization.config import Config
from rasterization.runner import Runner


def main(local_rank: int, world_rank, world_size: int, cfg: Config):

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
    else:
        runner.train()


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single NPU training
    python -m train --data_dir data/360_v2/garden --result_dir results/garden
    """

    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)