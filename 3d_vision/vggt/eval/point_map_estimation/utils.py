# coding=utf-8
# Adapted from
# https://github.com/wzzheng/StreamVGGT/blob/main/src/eval/mv_recon/launch.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright [2025–present] StreamVGGT. All rights reserved.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License.  
#
# --------------------------------------------------------
import os
from typing import Optional
import numpy as np
import torch
import open3d as o3d


def transfer_single_view(view, data_type, device, ignore_keys):
    for name in view.keys():
        if data_type == "input":
            if name in ignore_keys:
                continue
        if isinstance(view[name], tuple) or isinstance(
            view[name], list
        ):
            view[name] = [x.to(device, non_blocking=False) for x in view[name]]
        else:
            view[name] = view[name].to(device, non_blocking=False)


def transfer_data_between_devices(data, data_type, device):
    ignore_keys = set(
        [
            "depthmap",
            "dataset",
            "label",
            "instance",
            "idx",
            "true_shape",
            "rng",
        ]
    )
    for view in data:
        transfer_single_view(view, data_type, device, ignore_keys)


def get_pcd(pts_all_masked, pts_gt_all_masked, images_all_masked):
    threshold = 0.1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        pts_all_masked.reshape(-1, 3)
    )
    pcd.colors = o3d.utility.Vector3dVector(
        images_all_masked.reshape(-1, 3)
    )
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(
        pts_gt_all_masked.reshape(-1, 3)
    )
    pcd_gt.colors = o3d.utility.Vector3dVector(
        images_all_masked.reshape(-1, 3)
    )
    trans_init = np.eye(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd,
        pcd_gt,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    transformation = reg_p2p.transformation

    pcd = pcd.transform(transformation)

    return pcd, pcd_gt


def write_pcd(pcd, pcd_gt, save_path, scene_id):
    from pathlib import Path
    out_dir = Path(save_path)
    out_dir.mkdir(exist_ok=True)
    o3d.io.write_point_cloud(
        os.path.join(
            save_path, f"{scene_id.replace('/', '_')}-mask.ply"
        ),
        pcd,
     )
    o3d.io.write_point_cloud(
        os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                pcd_gt,
    )


def denormalize_image(data, dtype):
    with torch.cuda.amp.autocast(dtype=dtype):
        if isinstance(data, dict) and "img" in data:
            data["img"] = (data["img"] + 1.0) / 2.0
        elif isinstance(data, list) and all(isinstance(v, dict) and "img" in v for v in data):
            for view in data:
                view["img"] = (view["img"] + 1.0) / 2.0


def extract_pts3d(criterion, data, pred_result, use_proj):
    
    preds = pred_result[0]
    point_map_by_unprojection = pred_result[1]
    depth_conf = pred_result[2]

    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
            criterion.get_all_pts3d_t(data, preds)
        )
    in_camera1 = None
    pts_all, pts_gt_all, images_all, masks_all, conf_all = [], [], [], [], []
    for j, view in enumerate(data):
        if in_camera1 is None:
            in_camera1 = view["camera_pose"][0].cpu()

        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
        mask = view["valid_mask"].cpu().numpy()[0]

        if use_proj:
            pts = point_map_by_unprojection[j]
            conf = depth_conf[0, j].to(torch.float).cpu().data.numpy()
        else:
            pts = pred_pts[j].to(torch.float).cpu().numpy()[0]
            conf = preds[j]["conf"].to(torch.float).cpu().data.numpy()[0]

        pts_gt = gt_pts[j].detach().to(torch.float).cpu().numpy()[0]

        image_height, image_width = image.shape[:2]
        cx = image_width // 2
        cy = image_height // 2
        l, t = cx - 112, cy - 112
        r, b = cx + 112, cy + 112
        image = image[t:b, l:r]
        mask = mask[t:b, l:r]
        pts = pts[t:b, l:r]
        pts_gt = pts_gt[t:b, l:r]

        images_all.append(image[None, ...])
        pts_all.append(pts[None, ...])
        pts_gt_all.append(pts_gt[None, ...])
        masks_all.append(mask[None, ...])
        conf_all.append(conf[None, ...])
    images_all = np.concatenate(images_all, axis=0)
    pts_all = np.concatenate(pts_all, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    pts_all_masked = pts_all[masks_all > 0]
    pts_gt_all_masked = pts_gt_all[masks_all > 0]
    images_all_masked = images_all[masks_all > 0]

    mask = np.isfinite(pts_all_masked)  
    pts_all_masked = pts_all_masked[mask]
    pts_gt_all_masked = pts_gt_all_masked[mask]

    return pts_all_masked, pts_gt_all_masked, images_all_masked


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    assert src.shape == dst.shape
    number, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma = dst_c.T @ src_c / number  # (3,3)

    u, d, vt = np.linalg.svd(sigma) 

    s = np.eye(dim)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1

    r = u @ s @ vt

    if with_scale:
        var_src = (src_c ** 2).sum() / number
        final_s = (d * s.diagonal()).sum() / var_src
    else:
        final_s = 1.0

    t = mu_dst - final_s * r @ mu_src

    return final_s, r, t


def projection_alignment(pts_all_masked, pts_gt_all_masked):
    pts_all_masked = pts_all_masked.reshape(-1, 3)
    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
    s, r, t = umeyama_alignment(pts_all_masked, pts_gt_all_masked, with_scale=True)
    pts_all_aligned = (s * (r @ pts_all_masked.T)).T + t  # (N,3)
    pts_all_masked = pts_all_aligned

    return pts_all_masked, pts_gt_all_masked, pts_all_aligned