# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit

from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu


@dataclass
class FrequencyConfig:
    """Configuration for frequency components computation."""
    dim: int
    seq_len: int
    device: torch.device
    dtype: torch.dtype
    input_positions: Optional[torch.Tensor] = None
    height: Optional[int] = None
    width: Optional[int] = None


class PositionGetter:
    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        # 2: 2D coordinates (x, y)
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.cos_sin_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, 
        config: FrequencyConfig,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        """
        Unified frequency components computation for both legacy and modern paths.
        
        Legacy mode (config.input_positions=None):
            Returns (cos_components, sin_components) for simple RoPE application
            
        Modern mode (config.input_positions provided):
            Returns [vertical_cos, vertical_sin, horizontal_cos, horizontal_sin]
            for 2D position encoding with optional sequence parallel support
        
        Args:
            config: FrequencyConfig containing all computation parameters
            
        Returns:
            Legacy mode: (cos, sin) tuple
            Modern mode: [v_cos, v_sin, h_cos, h_sin] list
        """
        cache_key = (config.dim, config.seq_len, config.device, config.dtype)
        
        # Compute base frequency components (shared logic)
        if cache_key not in self.frequency_cache:
            # Compute inverse frequencies: freq_i = 1 / (base^(i/dim))
            # 0, dim, 2: Generate even indices [0, 2, 4, ...] for frequency computation
            exponents = torch.arange(0, config.dim, 2, device=config.device).float() / config.dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            
            # Generate position indices [0, 1, 2, ..., seq_len-1]
            positions = torch.arange(config.seq_len, device=config.device, dtype=inv_freq.dtype)
            
            # Compute angles: angles[i,j] = position[i] * inv_freq[j]
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(config.dtype)
            
            # dim=-1: Duplicate angles along last dimension for full feature dimension
            angles = torch.cat((angles, angles), dim=-1)
            
            # Compute cos and sin components
            cos_components = angles.cos().to(config.dtype)
            sin_components = angles.sin().to(config.dtype)
            
            self.frequency_cache[cache_key] = (cos_components, sin_components)
        
        cos_components, sin_components = self.frequency_cache[cache_key]
        
        # Legacy mode: return raw cos/sin components
        if config.input_positions is None:
            return cos_components, sin_components
        
        # Modern mode: apply to 2D positions
        # 0: Extract batch size from first dimension
        batch_size = config.input_positions.shape[0]
        
        if config.height is not None and config.width is not None and batch_size is not None:
            sub_cache_key = (config.height, config.width, batch_size, config.seq_len)
            
            if sub_cache_key not in self.cos_sin_cache:
                # ..., 0: Extract y-coordinate (vertical position)
                # ..., 1: Extract x-coordinate (horizontal position)
                # [:, None, :, :]: Add head dimension at index 1
                vertical_cos = F.embedding(config.input_positions[..., 0], cos_components)[:, None, :, :]
                vertical_sin = F.embedding(config.input_positions[..., 0], sin_components)[:, None, :, :]
                horizontal_cos = F.embedding(config.input_positions[..., 1], cos_components)[:, None, :, :]
                horizontal_sin = F.embedding(config.input_positions[..., 1], sin_components)[:, None, :, :]
                self.cos_sin_cache[sub_cache_key] = (vertical_cos, vertical_sin, horizontal_cos, horizontal_sin)
            
            vertical_cos, vertical_sin, horizontal_cos, horizontal_sin = self.cos_sin_cache[sub_cache_key]
        else:
            # No caching for variable batch sizes
            vertical_cos = F.embedding(config.input_positions[..., 0], cos_components)[:, None, :, :]
            vertical_sin = F.embedding(config.input_positions[..., 0], sin_components)[:, None, :, :]
            horizontal_cos = F.embedding(config.input_positions[..., 1], cos_components)[:, None, :, :]
            horizontal_sin = F.embedding(config.input_positions[..., 1], sin_components)[:, None, :, :]
        
        return [vertical_cos, vertical_sin, horizontal_cos, horizontal_sin]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Rotate features by 90 degrees (swap and negate)."""
        feature_dim = x.shape[-1]
        # Split into two halves and rotate: [x1, x2] -> [-x2, x1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply 1D rotary position embedding using NPU-optimized operation."""
        # 'half': Apply rotation to half the features (RoPE2D splits features for x/y)
        return torch_npu.npu_rotary_mul(tokens, cos, sin, rotary_mode='half')

    def forward(
        self, 
        tokens: torch.Tensor, 
        positions: torch.Tensor, 
        height: Optional[int] = None, 
        width: Optional[int] = None,
        global_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Unified forward pass for 2D rotary position embeddings.
        
        Supports both legacy mode (height/width=None) and SP mode (height/width provided).
        
        Args:
            tokens: [B, num_heads, N, head_dim] - Input tokens
            positions: [B, N, 2] - 2D positions (x, y coordinates)
            height: Image height (None for legacy mode)
            width: Image width (None for legacy mode)
            global_seq_len: Global sequence length (auto-calculated if None)
        
        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.
        """
        # 2: Feature dimension must be even for RoPE (split into two halves)
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        # 3: Positions must be 3D [batch, tokens, coordinates]
        # 2: Last dimension must be 2 for 2D coordinates (x, y)
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # //2: Split feature dimension in half for vertical/horizontal encoding
        feature_dim = tokens.size(-1) // 2
        
        # Legacy mode: height and width not provided
        if height is None and width is None:
            # +1: Convert max position (0-indexed) to sequence length
            max_position = int(positions.max()) + 1
            
            # Get base frequency components (legacy mode)
            config = FrequencyConfig(
                dim=feature_dim,
                seq_len=max_position,
                device=tokens.device,
                dtype=tokens.dtype,
                input_positions=None
            )
            cos_comp, sin_comp = self._compute_frequency_components(config)
            
            # 2: Split features into two halves for vertical and horizontal
            vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

            # ..., 0: Extract y-coordinate for vertical encoding
            # ..., 1: Extract x-coordinate for horizontal encoding
            # [:, None, :, :]: Add head dimension for broadcasting
            vertical_cos = F.embedding(positions[..., 0], cos_comp)[:, None, :, :]
            vertical_sin = F.embedding(positions[..., 0], sin_comp)[:, None, :, :]
            horizontal_cos = F.embedding(positions[..., 1], cos_comp)[:, None, :, :]
            horizontal_sin = F.embedding(positions[..., 1], sin_comp)[:, None, :, :]

            # Apply rotary embeddings to each half
            vertical_features = self._apply_1d_rope(vertical_features, vertical_cos, vertical_sin)
            horizontal_features = self._apply_1d_rope(horizontal_features, horizontal_cos, horizontal_sin)

            # dim=-1: Concatenate along feature dimension
            return torch.cat((vertical_features, horizontal_features), dim=-1)
        
        # Modern mode with sequence parallel support
        # Auto-calculate global sequence length if not provided
        if global_seq_len is None:
            global_seq_len = height * width
        
        # Get frequency components with 2D position application (modern mode)
        config = FrequencyConfig(
            dim=feature_dim,
            seq_len=global_seq_len,
            device=tokens.device,
            dtype=tokens.dtype,
            input_positions=positions,
            height=height,
            width=width
        )
        cos_sin_output = self._compute_frequency_components(config)
        vertical_cos, vertical_sin, horizontal_cos, horizontal_sin = cos_sin_output
        
        # Split and apply rotary embeddings
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        
        vertical_features = self._apply_1d_rope(vertical_features, vertical_cos, vertical_sin)
        horizontal_features = self._apply_1d_rope(horizontal_features, horizontal_cos, horizontal_sin)
        
        return torch.cat((vertical_features, horizontal_features), dim=-1)