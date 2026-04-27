# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor

logger = logging.getLogger(__name__)


class TrackHead(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    
    Supports sequence parallel
    
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
        sp_config: Optional['SPConfig'] = None,
        sp_global_group: Optional[dist.ProcessGroup] = None,
        allgather_output: bool = False,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in: Input dimension of tokens from the backbone
            patch_size: Size of image patches used in the vision transformer
            features: Number of feature channels in the feature extractor output
            iters: Number of refinement iterations for tracking predictions
            predict_conf: Whether to predict confidence scores for tracked points
            stride: Stride value for the tracker predictor
            corr_levels: Number of correlation pyramid levels
            corr_radius: Radius for correlation computation
            hidden_size: Size of hidden layers in the tracker network
            sp_config: Sequence parallel configuration
            sp_global_group: Process group for SP
            allgather_output: Whether to allgather output to full sequence
        """
        super().__init__()

        self.patch_size = patch_size

        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,
            down_ratio=2,
            pos_embed=False,
            sp_config=sp_config,
            sp_global_group=sp_global_group,
            allgather_output=False,
        )

        self.tracker = BaseTrackerPredictor(
            latent_dim=features,
            predict_conf=predict_conf,
            stride=stride,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_size=hidden_size,
        )

        self.iters = iters
        
        self.sp_config = sp_config
        self.sp_global_group = sp_global_group
        self.sp_enabled = sp_config is not None and sp_config.sp_degree > 1
        self.allgather_output = allgather_output
        
        if self.sp_enabled:
            self.sp_world_size = sp_config.sp_degree

    def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list: List of aggregated tokens from the backbone
                                   Single-device: [batch_size, seq_len, num_patches, channel]
                                   SP mode: [batch_size, shard_seq_len, num_patches, channel]
            images: Input images
                   Single-device: [batch_size, seq_len, channel, height, width]
                   SP mode: [batch_size, shard_seq_len, channel, height, width]
            patch_start_idx: Starting index for patch tokens
            query_points: Initial query points to track [batch_size, num_queries, 2]
            iters: Number of refinement iterations

        Returns:
            tuple: (coord_preds, vis_scores, conf_scores)
                  Single-device: [batch_size, seq_len, num_queries, *]
                  SP mode (default): [batch_size, shard_seq_len, num_queries, *]
                  SP mode (allgather_output=True): [batch_size, seq_len, num_queries, *]
        """
        batch_size, shard_seq_len, _, height, width = images.shape

        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)

        if iters is None:
            iters = self.iters

        coord_preds, vis_scores, conf_scores = self.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)
        
        if self.sp_enabled and self.allgather_output:
            coord_preds = [self._allgather_sequence(coord) for coord in coord_preds]
            vis_scores = self._allgather_sequence(vis_scores)
            conf_scores = self._allgather_sequence(conf_scores)

        return coord_preds, vis_scores, conf_scores
    
    def _allgather_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Allgather tensor from all ranks in SP mode.
        
        Args:
            tensor: [batch_size, shard_seq_len, ...]
        
        Returns:
            tensor_full: [batch_size, seq_len, ...]
        """
        shape = tensor.shape
        batch_size, shard_seq_len = shape[0], shape[1]
        remaining_dims = shape[2:]
        
        tensor_flat = tensor.reshape(batch_size, shard_seq_len, -1)
        
        tensor_gathered = torch.empty(
            batch_size, self.sp_world_size * shard_seq_len, tensor_flat.shape[-1],
            dtype=tensor.dtype,
            device=tensor.device
        )
        
        dist.all_gather_into_tensor(
            tensor_gathered,
            tensor_flat.contiguous(),
            group=self.sp_global_group
        )
        
        tensor_full = tensor_gathered.reshape(batch_size, -1, *remaining_dims)
        
        return tensor_full