# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse
import logging

import torch
import numpy as np

from dataset_utils.data_io import read_pfm, save_pfm

logging.basicConfig(level=logging.INFO)


def align_pred_to_gt(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_valid_pixels: int = 100,
) -> tuple[float, float, np.ndarray]:
    """
    Aligns a predicted depth map to a ground truth depth map using scale and shift.
    The alignment is: gt_aligned_to_pred ≈ scale * pred_depth + shift.

    Args:
        pred_depth (np.ndarray): The HxW predicted depth map.
        gt_depth (np.ndarray): The HxW ground truth depth map.
        min_gt_depth (float): Minimum valid depth value for GT.
        max_gt_depth (float): Maximum valid depth value for GT.
        min_pred_depth (float): Minimum valid depth value for predictions.
        min_valid_pixels (int): Minimum number of valid overlapping pixels.

    Returns:
        tuple[float, float, np.ndarray]:
            - scale (float): The calculated scale factor. (NaN if alignment failed)
            - shift (float): The calculated shift offset. (NaN if alignment failed)
            - aligned_pred_depth (np.ndarray): The HxW predicted depth map after
                                               applying scale and shift. (Original pred_depth
                                               if alignment failed)
    """
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"Predicted depth shape {pred_depth.shape} must match GT depth shape {gt_depth.shape}"
        )

    # Extract valid depth values
    gt_masked = gt_depth[valid_mask]
    pred_masked = pred_depth[valid_mask]

    if len(gt_masked) < min_valid_pixels:
        logging.warning(
            f"Warning: Not enough valid pixels ({len(gt_masked)} < {min_valid_pixels}) to align. "
            "Using all pixels."
        )
        gt_masked = gt_depth.reshape(-1)
        pred_masked = pred_depth.reshape(-1)


    # Handle case where pred_masked has no variance (e.g., all zeros or a constant value)
    if np.std(pred_masked) < 1e-6: # Small epsilon to check for near-constant values
        logging.warning(
            "Warning: Predicted depth values in the valid mask have near-zero variance. "
            "Scale is ill-defined. Setting scale=1 and solving for shift only."
        )
        scale = 1.0
        shift = np.mean(gt_masked) - np.mean(pred_masked) # or np.median(gt_masked) - np.median(pred_masked)
    else:
        tmp_pred_masked = np.vstack([pred_masked, np.ones_like(pred_masked)]).T
        try:
            x, residuals, rank, s_values = np.linalg.lstsq(tmp_pred_masked, gt_masked, rcond=None)
            scale, shift = x[0], x[1]
        except np.linalg.LinAlgError as e:
            logging.warning(f"Warning: Least squares alignment failed ({e}). Returning original prediction.")
            return np.nan, np.nan, pred_depth.copy()


    aligned_pred_depth = scale * pred_depth + shift
    return scale, shift, aligned_pred_depth



def get_args():
    parser = argparse.ArgumentParser("Align predicted depth to ground truth depth", add_help=False)
    parser.add_argument("--depth_conf_thres", type=int, default=3, 
        help="the range of depth conf is [1, +inf]. Ususally > 3 or > 5 should be good enough")
    parser.add_argument('--testlist', help='data list for testing')
    parser.add_argument('--output_path', help='root dir for the predection depth that will be aligned')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    with open(os.path.join(args.testlist)) as f:
        scans = [line.rstrip() for line in f.readlines()]
    for scan in scans:
        for view_id in range(49):
            path_str = f"{scan}/{{}}/{view_id:08d}.pfm"
            scan_name = os.path.join(args.output_path, path_str)
            gt_depth_filename = scan_name.format("depth_gt")
            est_depth_filename = scan_name.format("depth_est")
            depth_conf_filename = scan_name.format("confidence")
            gt_depth = torch.from_numpy(np.array(read_pfm(gt_depth_filename)[0], dtype=np.float32))
            est_depth = torch.from_numpy(np.array(read_pfm(est_depth_filename)[0], dtype=np.float32))
            depth_conf = torch.from_numpy(np.array(read_pfm(depth_conf_filename)[0], dtype=np.float32))

            valid_mask = torch.logical_and(
                gt_depth.squeeze().cpu() > 1e-3,     # filter out black background
                depth_conf.squeeze() > args.depth_conf_thres
            )
            valid_mask = valid_mask.numpy()  # Take first item in batch
  

            align_mask = valid_mask.copy()
  
            scale, shift, aligned_pred_depth = align_pred_to_gt(
                est_depth.squeeze().numpy(), 
                gt_depth.squeeze().numpy(), 
                align_mask
            )
            
            aligned_depth_filename = scan_name.format("aligned_depth_est")

            os.makedirs(aligned_depth_filename.rsplit('/', 1)[0], exist_ok=True)
            save_pfm(aligned_depth_filename, aligned_pred_depth)
        logging.info(f"Alignment Scan: {scan} Finished!")