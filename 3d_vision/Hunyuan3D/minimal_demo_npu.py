# Copyright 2007 Free Software Foundation Inc. https: fsf.org
#
# This file is part of MyProject.
#
# MyProject is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 only,
# as published by the Free Software Foundation.
#
# MyProject is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MyProject.  If not, see <https://www.gnu.org/licenses/>.

# coding=utf-8
# Adapted from
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/minimal_demo.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This code is based on Tencent-Hunyuan's Hunyuan3D-2 library and the Hunyuan3D-2
# implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to Hunyuan3D-2 used by Tencent-Hunyuan team that trained the model.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import argparse
import os
import time
import logging

import torch
import torch_npu
import torchair
import torch._dynamo
from PIL import Image
from torch_npu.contrib import transfer_to_npu

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.cache import first_block_forward, double_block_forward, single_block_forward
import hy3dgen.cache.cache_block
from module.dit_cache.cache_method import cache_manager

logging.basicConfig(level=logging.NOTSET)

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def local_path(*parts):
    return os.path.join(CURRENT_DIR, *parts)


def main():
    parser = argparse.ArgumentParser(
        description='Hunyuan3D'
    )
    
    parser.add_argument('--model_path', type=str, default='tencent/Hunyuan3D-2', help="模型路径")
    parser.add_argument('--multiview', action='store_true', default=False, help="多视角输入")
    parser.add_argument('--face_reduce', action='store_true', default=False, help="减少mesh面片数量")
    parser.add_argument('--full_graph', action='store_true', default=False, help="开启图模式")
    parser.add_argument('--multi_thread', action='store_true', default=False, help="开启delighting和render，bake的多线程并行")
    parser.add_argument('--use_render_npu', action='store_true', default=False, help="开启render_npu")
    parser.add_argument('--save_render', action='store_true', default=False, help="开启保存光栅化结果")
    parser.add_argument('--cache_config', type=str, default=local_path('hy3dgen', 'cache', 'cache_config.json'),
    help="cache_config路径")
    args = parser.parse_args()
    os.environ['MULTI_THREAD'] = 'true' if args.multi_thread else 'false'
    os.environ['USE_RENDER_NPU'] = 'true' if args.use_render_npu else 'false'
    os.environ['SAVE_RENDER'] = 'true' if args.save_render else 'false'


    if args.full_graph:
        config = torchair.CompilerConfig()
        config.experimental_config.keep_inference_input_mutations = True
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
    if args.multiview:
        images = {
            "front": local_path('assets', 'example_mv_images', '1', 'front.png'),
            "left": local_path('assets', 'example_mv_images', '1', 'left.png'),
            "back": local_path('assets', 'example_mv_images', '1', 'back.png')
        }

        for key in images:
            image = Image.open(images[key]).convert("RGBA")
            if image.mode == 'RGB':
                rembg = BackgroundRemover()
                image = rembg(image)
            images[key] = image

        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model_path + "mv",
            subfolder='hunyuan3d-dit-v2-mv',
            variant='fp16'
        )
        double_stream_layers = len(pipeline_shapegen.model.double_blocks)
        single_stream_layers = len(pipeline_shapegen.model.single_blocks)
        cache_params = {
            "num_steps": 100,
            "double_stream_layers": double_stream_layers,
            "single_stream_layers": single_stream_layers,   
        }
        cache_manager.from_config(args.cache_config, cache_params=cache_params)
        if cache_manager.cache_method.cache_name == "TaylorSeer":
            for block in pipeline_shapegen.model.double_blocks:
                block.forward = double_block_forward.__get__(block, type(block))
            for block in pipeline_shapegen.model.single_blocks:
                block.forward = single_block_forward.__get__(block, type(block))
        else:
            cache_block = pipeline_shapegen.model.double_blocks[0]
            cache_block.forward = first_block_forward.__get__(cache_block, type(cache_block))
        if args.full_graph and cache_manager.cache_method.cache_name != "NoCache":
            logging.info("Cannot enable both graph mode and DIT-Cache simultaneously. graph mode has been disabled")
            args.full_graph = False
        if args.full_graph:
            pipeline_shapegen.model = torch.compile(pipeline_shapegen.model, 
            dynamic=False, backend=npu_backend, fullgraph=True)
            pipeline_shapegen.vae.geo_decoder = torch.compile(pipeline_shapegen.vae.geo_decoder, 
            dynamic=False, backend=npu_backend, fullgraph=True)
        mesh = pipeline_shapegen(
            image=images,
            num_inference_steps=100,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]
        cache_manager.cache_method.print_statistics()
    else:
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)
        double_stream_layers = len(pipeline_shapegen.model.double_blocks)
        single_stream_layers = len(pipeline_shapegen.model.single_blocks)
        cache_params = {
            "num_steps": 100,
            "double_stream_layers": double_stream_layers,
            "single_stream_layers": single_stream_layers,   
        }
        cache_manager.from_config(args.cache_config, cache_params=cache_params)
        if cache_manager.cache_method.cache_name == "TaylorSeer":
            for block in pipeline_shapegen.model.double_blocks:
                block.forward = double_block_forward.__get__(block, type(block))
            for block in pipeline_shapegen.model.single_blocks:
                block.forward = single_block_forward.__get__(block, type(block))
        else:
            cache_block = pipeline_shapegen.model.double_blocks[0]
            cache_block.forward = first_block_forward.__get__(cache_block, type(cache_block))
        if args.full_graph and cache_manager.cache_method.cache_name != "NoCache":
            logging.info("Cannot enable both graph mode and DIT-Cache simultaneously. graph mode has been disabled")
            args.full_graph = False
        if args.full_graph:
            pipeline_shapegen.model = torch.compile(pipeline_shapegen.model, 
            dynamic=False, backend=npu_backend, fullgraph=True)
            pipeline_shapegen.vae.geo_decoder = torch.compile(pipeline_shapegen.vae.geo_decoder, 
            dynamic=False, backend=npu_backend, fullgraph=True)
        image_path = local_path('assets', 'demo.png')
        image = Image.open(image_path).convert("RGBA")
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)

        mesh = pipeline_shapegen(
            image=image,
            num_inference_steps=100,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]
        cache_manager.cache_method.print_statistics()
    if args.face_reduce:
        mesh = mesh.simplify_quadric_decimation(face_count=20000) 
    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(args.model_path, subfolder='hunyuan3d-paint-v2-0')
    torch.npu.synchronize()
    texgen_time = time.time()
    os.makedirs(local_path('outputs'), exist_ok=True)
    if args.multiview:
        mesh = pipeline_texgen(mesh, image=list(images.values()))
        torch.npu.synchronize()
        texgen_final_time = time.time() - texgen_time
        mesh.export(local_path('outputs', 'demo_mv.glb'))
    else:
        mesh = pipeline_texgen(mesh, image=image)
        torch.npu.synchronize()
        texgen_final_time = time.time() - texgen_time
        mesh.export(local_path('outputs', 'demo.glb'))
    logging.info(f"DIT扩散时长为{pipeline_shapegen.last_step_time:.3f}s")
    vol_decoder = pipeline_shapegen.vae.volume_decoder
    logging.info(f"VAE解码时长为{vol_decoder.last_step_time_vae:.3f}s")
    logging.info(f"texgen渲染时长为{texgen_final_time:.3f}s")
if __name__ == '__main__':
    main()
