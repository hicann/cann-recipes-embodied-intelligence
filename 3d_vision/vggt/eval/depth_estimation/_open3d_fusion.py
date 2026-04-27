# coding=utf-8
# Adapted from
# https://github.com/doubleZ0108/GeoMVSNet/blob/master/fusions/dtu/_open3d.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2022 Zhenxing Mi. All rights reserved.
# License under Apache.
# =======================================================================
import sys
import os
import argparse
import gc
import errno
import logging

from PIL import Image
import open3d as o3d
import numpy as np
import cv2


import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu

from dataset_utils.data_io import read_pfm, write_ply, parse_cameras



parser = argparse.ArgumentParser(description='Depth fusion with consistency check.')
parser.add_argument('--depth_path', type=str, default='')
parser.add_argument('--data_list', type=str, default='./datasets/lists/test.txt')
parser.add_argument('--ply_path', type=str, default='outputs/dtu/open3d_fusion_plys')
parser.add_argument('--dist_thresh', type=float, default=1.0)
parser.add_argument('--prob_thresh', type=float, default=5)
parser.add_argument('--num_consist', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(-1, 1, 1, height * width)  # [B, 3, 1, H*W]

        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, height, width)

    return warped_src_fea


def warp_colors(src_rgb, src_proj, ref_proj, depth_values):
    return homo_warping(src_rgb, src_proj, ref_proj, depth_values)   # re‑use same grid


def generate_points_from_depth(depth, proj):
    '''
    :param depth: (B, 1, H, W)
    :param proj: (B, 4, 4)
    :return: point_cloud (B, 3, H, W)
    '''
    batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
    inv_proj = torch.inverse(proj)

    rot = inv_proj[:, :3, :3]  # [B,3,3]
    trans = inv_proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1)
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
    proj_xyz = proj_xyz.view(batch, 3, height, width)

    return proj_xyz



def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    '''
    :param ref_depth: (1, 1, H, W)
    :param src_depths: (B, 1, H, W)
    :param ref_proj: (1, 4, 4)
    :param src_proj: (B, 4, 4)
    :return: ref_pc: (1, 3, H, W), aligned_pcs: (B, 3, H, W), dist: (B, 1, H, W)
    '''

    ref_pc = generate_points_from_depth(ref_depth, ref_proj)
    src_pcs = generate_points_from_depth(src_depths, src_projs)

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0])**2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1])**2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2])**2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist



def load_data(depth_path, scene_name, thresh):
    # Check if the scene directory exists
    scene_dir = f"{depth_path}/{scene_name}"
    if not os.path.exists(scene_dir):
        logging.warning(f"Scene directory {scene_dir} does not exist, skipping...")
        return None, None, None

    depths = []
    projs = []
    rgbs = []

    for view in range(49):
        img_filename = "{}/{}/images/{:08d}.jpg".format(depth_path, scene_name, view)
        cam_filename = "{}/{}/cams/{:08d}_cam.txt".format(depth_path, scene_name, view)
        depth_filename = "{}/{}/aligned_depth_est/{:08d}.pfm".format(depth_path, scene_name, view)
        confidence_filename = "{}/{}/confidence/{:08d}.pfm".format(depth_path, scene_name, view)

        extr_mat, intr_mat = parse_cameras(cam_filename)
        proj_mat = np.eye(4)
        proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], extr_mat[:3, :4])
        projs.append(torch.from_numpy(proj_mat))

        dep_map, _ = read_pfm(depth_filename)
        h, w = dep_map.shape
        conf_map, _ = read_pfm(confidence_filename)
        conf_map = cv2.resize(conf_map, (w, h), interpolation=cv2.INTER_LINEAR)

        dep_map = dep_map * (conf_map > thresh).astype(np.float32)
        depths.append(torch.from_numpy(dep_map).unsqueeze(0))

        rgb = np.array(Image.open(img_filename))
        rgbs.append(rgb)

    depths = torch.stack(depths).float()
    projs = torch.stack(projs).float()

    # NEW – pack RGBs into one tensor, [B,3,H,W] in [0,1]
    rgb_tensors = [torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1)
                   for img in rgbs]
    rgbs = torch.stack(rgb_tensors)           # (B,3,H,W)

    if args.device == 'cuda' and torch.cuda.is_available():
        depths = depths.cuda()
        projs = projs.cuda()
        rgbs = rgbs.cuda()

    return depths, projs, rgbs               # <- now a tensor, not list


def extract_points(pc, mask, rgb):
    # Convert CUDA tensors to CPU before converting to numpy
    pc = pc.cpu()
    mask = mask.cpu()
    rgb = rgb.cpu()

    # Now convert to numpy
    pc = pc.numpy()
    mask = mask.numpy()
    rgb = rgb.numpy()

    mask = np.reshape(mask, (-1,))
    pc = np.reshape(pc, (-1, 3))
    rgb = np.reshape(rgb, (-1, 3))

    points = pc[np.where(mask)]
    colors = rgb[np.where(mask)]

    points_with_color = np.concatenate([points, colors], axis=1)

    return points_with_color


def process_scene(input_args, input_scene):
    # Check if the scene directory exists
    scene_dir = f"{input_args.depth_path}/{input_scene}"
    if not os.path.exists(scene_dir):
        logging.warning(f"Scene directory {scene_dir} does not exist, skipping...")
        # continue
        return
    depths, projs, rgbs = load_data(input_args.depth_path, input_scene, input_args.prob_thresh)
    tot_frame = depths.shape[0]
    height, width = depths.shape[2], depths.shape[3]
    points = []

    logging.info('Scene: {} total: {} frames'.format(input_scene, tot_frame))
    for idx in range(tot_frame):
        pc_buff = torch.zeros((3, height, width), device=depths.device, dtype=depths.dtype)
        val_cnt = torch.zeros((1, height, width), device=depths.device, dtype=depths.dtype)
        j = 0
        batch_size = 20

        while True:
            # ---- depth part ----
            ref_pc, pcs, dist = filter_depth(
                ref_depth=depths[idx:idx + 1],
                src_depths=depths[j:min(j + batch_size, tot_frame)],
                ref_proj=projs[idx:idx + 1],
                src_projs=projs[j:min(j + batch_size, tot_frame)]
            )

            depth_mask = (dist < input_args.dist_thresh).float()

            # Initialize masks with depth consistency results
            masks = depth_mask

            
            masked_pc = pcs * masks
            pc_buff += masked_pc.sum(dim=0, keepdim=False)
            val_cnt += masks.sum(dim=0, keepdim=False)

            j += batch_size
            if j >= tot_frame:
                break

        final_mask = (val_cnt >= input_args.num_consist).squeeze(0)
        avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0)

        final_pc = extract_points(avg_points, final_mask, rgbs[idx])
        points.append(final_pc)
        logging.info('Processing {} {}/{} ...'.format(input_scene, idx + 1, tot_frame))

    ply_id = int(input_scene[4:])
    write_ply('{}/{:03d}.ply'.format(input_args.ply_path, ply_id), np.concatenate(points, axis=0))
    logging.info('Save {}/{:03d}.ply successful.'.format(input_args.ply_path, ply_id))
    del points, depths, rgbs, projs
    return 



def open3d_filter():
    with torch.no_grad():
        mkdir_p(args.ply_path)
        with open(args.data_list, 'r') as f:
            all_scenes = f.readlines()
            all_scenes = list(map(str.strip, all_scenes))

        for i, scene in enumerate(all_scenes):
            logging.info(f"{i + 1}/{len(all_scenes)} Processing {scene}:")
            is_successed = process_scene(args, scene)
            if is_successed == False:
                continue
            gc.collect()



if __name__ == '__main__':
    open3d_filter()
