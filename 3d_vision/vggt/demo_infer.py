# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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

import os
import argparse
import contextlib
import time
import logging

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu

from vggt.models.vggt import VGGT
from vggt.sp import SPConfig
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.cast_weight import cast_model_weight
from eval.general_utils import fix_random_seed
from quant.vggt_utils import replace_linear_in_vggt, set_ignore_quantize
from quant.vggt_linear import LinearW8A8

logging.basicConfig(level=logging.INFO)


class EmptyContextManager(contextlib.nullcontext):
    """Empty context manager that is used when profiling is disabled."""

    @staticmethod
    def step():
        pass


def define_profiler(enable_profiler=False, profile_save_path="prof", rank=0):
    """
    Define profiler based on configuration.
    
    Args:
        enable_profiler: Whether to enable profiling
        profile_save_path: Directory to save profiling results
        rank: Rank ID for multi-card scenario
    
    Returns:
        Profiler context manager
    """
    if enable_profiler:
        os.makedirs(profile_save_path, exist_ok=True)
        
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False
        )
        
        profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU,
                torch_npu.profiler.ProfilerActivity.CPU,
            ],
            with_stack=False,
            record_shapes=True,
            profile_memory=True,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=2,
                active=1,
                repeat=1,
                skip_first=0
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path),
            with_modules=False,
            with_flops=False,
        )
        logging.info(f"Profiler enabled, results will be saved to {profile_save_path}")
    else:
        profiler = EmptyContextManager()
    
    return profiler


def setup_distributed():
    """Initialize distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        try:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        except ValueError as exc:
            raise RuntimeError("Invalid distributed environment variables.") from exc
    else:
        raise RuntimeError("Distributed environment not set up properly")
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='hccl')
    
    logging.info(f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    return rank, world_size, local_rank


def setup_sequence_parallel_groups(ulysses_degree, ring_degree):
    """
    Create process groups for sequence parallel.
    
    Args:
        ulysses_degree: Degree of Ulysses parallelism
        ring_degree: Degree of Ring Attention parallelism
    
    Returns:
        sp_config, ulysses_group, ring_group, global_group
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size != ulysses_degree * ring_degree:
        raise ValueError(
            f"world_size ({world_size}) must equal ulysses_degree ({ulysses_degree}) "
            f"* ring_degree ({ring_degree})"
        )
    
    # Create Ulysses groups
    ulysses_groups = []
    for i in range(ring_degree):
        ranks = [i * ulysses_degree + j for j in range(ulysses_degree)]
        group = dist.new_group(ranks)
        ulysses_groups.append(group)
    
    ulysses_rank = rank % ulysses_degree
    ulysses_group_idx = rank // ulysses_degree
    ulysses_group = ulysses_groups[ulysses_group_idx]
    
    # Create Ring groups
    ring_groups = []
    for i in range(ulysses_degree):
        ranks = [i + j * ulysses_degree for j in range(ring_degree)]
        group = dist.new_group(ranks)
        ring_groups.append(group)
    
    ring_group = ring_groups[ulysses_rank]
    
    global_group = dist.group.WORLD
    
    sp_config = SPConfig(
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        use_ring_overlap=True,
    )
    
    logging.info(f"Sequence Parallel setup: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}")
    
    return sp_config, ulysses_group, ring_group, global_group


def get_all_files_paths(dir_path):
    """Get all file paths in a directory."""
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def sync_and_get_time(start_time=None, use_syn=True, log_result=False):
    """Synchronize NPU and get timestamp."""
    if use_syn:
        torch.npu.synchronize()
    timestamp = time.time()
    if start_time is not None:
        timestamp -= start_time
        if log_result:
            logging.info(f"VGGT inference time cost is: {timestamp*1000:.2f} ms")
        return timestamp
    return timestamp


def run_inference_with_sp(args):
    """Main inference function with sequence parallel support."""
    fix_random_seed(42)
    
    rank, world_size, local_rank = setup_distributed()
    device = f"npu:{local_rank}"
    
    if args.enable_sp:
        sp_config, ulysses_group, ring_group, global_group = setup_sequence_parallel_groups(
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree
        )
    else:
        sp_config = None
        ulysses_group = None
        ring_group = None
        global_group = None
    
    dtype = torch.bfloat16
    checkpoint_path = args.ckpt
    
    # Load model with SP support
    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        sp_config=sp_config,
        sp_ulysses_group=ulysses_group,
        sp_ring_group=ring_group,
        sp_global_group=global_group,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(dtype)
    model.to(device)
    model.eval()
    model = cast_model_weight(model)
    
    logging.info(f"Model loaded successfully on rank {rank}")
    
    # Load images
    image_paths = args.images_path
    image_names = get_all_files_paths(image_paths)
    image_names = sorted(image_names)
    
    images = load_and_preprocess_images(image_names).to(device)
    
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    
    logging.info(f"Loaded {len(image_names)} images, shape: {images.shape}")
    
    # Warmup
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            _ = model(images)
    
    dist.barrier()
    logging.info("Warmup completed")
    
    # Setup profiler
    profile_dir = os.path.join(args.profile_dir, f"rank_{rank}") if args.enable_profiling else "prof"
    profiler = define_profiler(
        enable_profiler=args.enable_profiling,
        profile_save_path=profile_dir,
        rank=rank
    )
    
    num_runs = args.num_runs
    exec_time_list = []
    
    # Run inference
    with profiler:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                for step in range(num_runs):
                    dist.barrier()
                    start_time = sync_and_get_time()
                    predictions = model(images)
                    dist.barrier()
                    exec_time = sync_and_get_time(start_time, log_result=(rank == 0 and step >= 2))
                    exec_time_list.append(exec_time)
                    profiler.step()
    
    dist.barrier()
    
    if rank == 0:
        # Skip first 2 warmup runs
        valid_times = exec_time_list[2:]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        logging.info(f"Execution times (ms): {[t*1000 for t in exec_time_list]}")
        logging.info(f"Average inference time (excluding warmup): {avg_time*1000:.2f} ms")
    
    dist.destroy_process_group()


def quick_start(args):
    """Single GPU inference with optional quantization."""
    fix_random_seed(42)
    
    # Device check
    device = "npu:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("NPU is not available. Check your environment.")
    
    logging.info(f"Using device: {device}")
    
    dtype = torch.bfloat16
    checkpoint_path = args.ckpt
    
    # Load model with quantization support
    if args.enableW8A8:
        logging.info("Loading W8A8 quantized model...")
        model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.to(device).eval()
        logging.info("W8A8 quantized model loaded successfully")
    else:
        logging.info("Loading standard model...")
        model = VGGT()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model = model.to(dtype)
        model.to(device).eval()
        model = cast_model_weight(model)
        logging.info("Standard model loaded successfully")
        
        # Build W8A8 quantized model if requested
        if args.buildW8A8:
            logging.info("Building W8A8 quantized model...")
            set_ignore_quantize(model, ignore_quantize=True)
            replace_linear_in_vggt(model, device=device)
            save_path = os.path.join(os.getcwd(), "VGGT_model_W8A8.pt")
            torch.save(model, save_path)
            logging.info(f"W8A8 model saved to {save_path}")
            return
    
    # Load images
    image_paths = args.images_path
    image_names = get_all_files_paths(image_paths)
    image_names = sorted(image_names)
    
    logging.info(f"Loading {len(image_names)} images from {image_paths}")
    
    images = load_and_preprocess_images(image_names).to(device)
    
    # Run inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Warmup
            predictions = model(images)
            
            exec_time_list = []
            for step in range(6):
                start_time = sync_and_get_time()
                predictions = model(images)
                exec_time = sync_and_get_time(start_time, log_result=True)
                exec_time_list.append(exec_time)
            
            avg_time = sum(exec_time_list) / len(exec_time_list)
            logging.info(f"Execution times (ms): {[t*1000 for t in exec_time_list]}")
            logging.info(f"Average inference time: {avg_time*1000:.2f} ms ({avg_time:.4f} s)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("VGGT Inference", add_help=True)
    
    # Basic arguments
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--images_path", default="examples/kitchen/images", help="Dataset location")
    
    # Quantization arguments
    parser.add_argument("--buildW8A8", action="store_true", help="Build W8A8 quantized model")
    parser.add_argument("--enableW8A8", action="store_true", help="Use W8A8 quantized model")
    
    # Sequence Parallel arguments
    parser.add_argument("--enable_sp", action="store_true", help="Enable sequence parallel")
    parser.add_argument("--ulysses_degree", type=int, default=2, help="Ulysses parallelism degree")
    parser.add_argument("--ring_degree", type=int, default=2, help="Ring attention degree")
    
    # Profiling arguments
    parser.add_argument("--enable_profiling", action="store_true", help="Enable NPU profiling")
    parser.add_argument("--profile_dir", default="prof_sp", help="Profiling results directory")
    
    # Performance arguments
    parser.add_argument("--num_runs", type=int, default=6, help="Number of inference runs")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if running in distributed mode
    if args.enable_sp or ('RANK' in os.environ and 'WORLD_SIZE' in os.environ):
        logging.info("Running in distributed mode with sequence parallel")
        run_inference_with_sp(args)
    else:
        logging.info("Running in single GPU mode")
        quick_start(args)


if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=False)
    main()
    logging.info("Run all examples success")