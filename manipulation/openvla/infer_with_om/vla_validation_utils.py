# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
Common utilities for OpenVLA model validation and testing.

This module provides shared functions for:
- Comparing model outputs (PyTorch vs ONNX/OM)
- Creating dummy inputs for testing
- Post-processing actions from generated tokens
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


# ==================== Data Classes ====================

@dataclass
class ComparisonConfig:
    """Container for output comparison configuration."""
    rtol: float = 1e-2
    atol: float = 1e-2
    mean_diff_threshold: Optional[float] = 1e-2


@dataclass
class DummyInputConfig:
    """Container for dummy input preparation configuration."""
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    seed: int = 0
    dtype: torch.dtype = torch.float16


# ==================== Output Comparison Functions ====================

def compare_outputs(
    ref: np.ndarray,
    out: np.ndarray,
    name: str = "",
    config: Optional[ComparisonConfig] = None,
) -> bool:
    """
    Compare two numpy arrays and report differences.

    Args:
        ref: Reference array (PyTorch output)
        out: Output array (ONNX/OM output)
        name: Name for logging
        config: Comparison configuration (optional, uses defaults if None)

    Returns:
        True if arrays match within tolerance
    """
    if config is None:
        config = ComparisonConfig()
    
    abs_diff = np.abs(ref - out)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    match = np.allclose(ref, out, rtol=config.rtol, atol=config.atol)
    if config.mean_diff_threshold is not None:
        match = match and mean_diff < config.mean_diff_threshold
    
    logging.info(f"{name}: max abs diff = {max_diff:.6e}")
    logging.info(f"{name}: mean abs diff = {mean_diff:.6e}")
    
    if config.mean_diff_threshold is not None:
        match_status = '✓ MATCH' if match else '✗ MISMATCH'
        logging.info(
            f"{name}: {match_status} (rtol={config.rtol}, atol={config.atol}, "
            f"mean_diff_threshold={config.mean_diff_threshold})"
        )
    else:
        logging.info(
            f"{name}: {'MATCH' if match else 'MISMATCH'} "
            f"(rtol={config.rtol}, atol={config.atol})"
        )
    
    return match


def print_diff_only(
    ref: np.ndarray,
    out: np.ndarray,
    name: str = "",
):
    """
    Print only diff statistics without match judgment.

    Args:
        ref: Reference array (PyTorch output)
        out: Output array (ONNX/OM output)
        name: Name for logging
    """
    abs_diff = np.abs(ref - out)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    logging.info(f"{name}: max abs diff = {max_diff:.6e}")
    logging.info(f"{name}: mean abs diff = {mean_diff:.6e}")


def show_compare(
    name: str,
    torch_output: np.ndarray,
    onnx_output: np.ndarray,
    n_print: int = 7,
):
    """
    Show comparison of first N elements from two outputs.

    Args:
        name: Name for logging
        torch_output: PyTorch output array
        onnx_output: ONNX/OM output array
        n_print: Number of elements to print
    """
    logging.info(f"\n{name} - Pytorch output (first {n_print} elements):")
    logging.info(torch_output.flatten()[:n_print])
    logging.info(f"{name} - ONNX/OM output (first {n_print} elements):")
    logging.info(onnx_output.flatten()[:n_print])


# ==================== Dummy Input Preparation Functions ====================

def get_image_size_from_processor(processor) -> Tuple[int, int]:
    """
    Get image size from processor config, with fallback to 224x224.

    Args:
        processor: Hugging Face processor

    Returns:
        Tuple of (height, width)
    """
    try:
        image_size = processor.image_processor.size
        height = image_size.get("height", 224)
        width = image_size.get("width", 224)
        return height, width
    except AttributeError:
        logging.warning("Warning: Could not determine image size from processor config. Using default 224x224.")
        return 224, 224


def make_dummy_image(width: int, height: int, seed: int = 0) -> Image.Image:
    """
    Create a dummy random image for testing.

    Args:
        width: Image width
        height: Image height
        seed: Random seed (0 = do not set seed)

    Returns:
        PIL Image
    """
    if seed:
        np.random.seed(seed)
    dummy_image_array = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(dummy_image_array)


def prepare_dummy_inputs(
    processor: AutoProcessor,
    prompt: str,
    config: Optional[DummyInputConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Prepare dummy inputs for model testing.

    Args:
        processor: Hugging Face processor
        prompt: Text prompt
        config: Dummy input configuration (optional, uses defaults if None)

    Returns:
        Dictionary of input tensors
    """
    if config is None:
        config = DummyInputConfig()
    
    # Get image size
    h0, w0 = get_image_size_from_processor(processor)
    height = (
        config.image_height
        if config.image_height is not None and config.image_height > 0
        else h0
    )
    width = (
        config.image_width
        if config.image_width is not None and config.image_width > 0
        else w0
    )

    # Create dummy image
    dummy_image = make_dummy_image(width=width, height=height, seed=config.seed)

    # Process inputs
    inputs = processor(
        text=prompt, images=dummy_image, return_tensors="pt"
    ).to(dtype=config.dtype)

    return inputs


# ==================== Action Post-processing Functions ====================

def post_process_action(
    generated_ids: np.ndarray,
    action_dim: int,
    vocab_size: int,
    bin_centers: np.ndarray,
    action_norm_stats: dict,
) -> np.ndarray:
    """
    Post-process generated token IDs to obtain actions.

    Args:
        generated_ids: Generated token IDs (batch_size, seq_len)
        action_dim: Action dimension
        vocab_size: Vocabulary size
        bin_centers: Bin centers for action discretization
        action_norm_stats: Action normalization statistics

    Returns:
        Post-processed actions
    """
    # Extract action tokens (last action_dim tokens)
    predicted_action_token_ids = generated_ids[0, -action_dim:]

    # Convert tokens to discretized actions
    discretized_actions = vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1
    )

    # Map to bin centers
    normalized_actions = bin_centers[discretized_actions]

    # Denormalize using action stats
    mask = action_norm_stats.get(
        "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
    )
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])

    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    return actions

