# coding=utf-8
# Adapted from
# https://github.com/alibaba/cascade-stereo/blob/master/CasMVSNet/datasets/general_eval.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) 2019 Alibaba. All rights reserved.
# License under MIT. 
# =======================================================================
import os

import cv2
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset

from dataset_utils.data_io import read_pfm


class DTUDataset(Dataset):
    def __init__(self, root_dir, list_file, n_views, **kwargs):
        super(DTUDataset, self).__init__()
        
        self.root_dir = root_dir
        self.list_file = list_file
        self.n_views = n_views


        self.total_depths = 192
        self.interval_scale = 1.06

        self.max_wh = kwargs.get("max_wh", (1600, 1200))

        self.metas = self.build_metas()

    

    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        scan, light_idx, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views - 1]
        scale_ratio = 1

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # @Note image & cam
            
            img_filename = os.path.join(self.root_dir, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.root_dir, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_wh[0], self.max_wh[1])

            imgs.append(img.transpose(2, 0, 1))

            # reference view
            if i == 0:
                # @Note depth values
                diff = 0.5 
                depth_max = depth_interval * (self.total_depths - diff) + depth_min
                depth_values = np.array([depth_min * scale_ratio, depth_max * scale_ratio], dtype=np.float32)

                depth_filename = os.path.join(self.root_dir, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
                depth = self.read_scale_depth(depth_filename, scale_ratio, self.max_wh[0], self.max_wh[1])
                

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            
        proj_matrices = np.stack(proj_matrices)
        intrinsics = np.stack(intrinsics)

        pjmats = proj_matrices.copy()
        pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        ins = intrinsics.copy()
        ins[:2, :] = intrinsics[:2, :] * 4.0
        proj_matrices = pjmats
        intrinsics_matrices = ins

        sample = {
            "imgs": imgs,
            "proj_matrices": proj_matrices,
            "intrinsics_matrices": intrinsics_matrices,
            "depth_values": depth_values,
            "depth": depth,
            "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
        }

        return sample

    def build_metas(self):
        metas = []

        with open(os.path.join(self.list_file)) as f:
            scans = [line.rstrip() for line in f.readlines()]
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            f = open(os.path.join(self.root_dir, pair_file))
            num_viewpoint = int(f.readline())

            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                if len(src_views) < self.n_views:
                    src_views += [src_views[0]] * (self.n_views - len(src_views))
                metas.append((scan, 3, ref_view, src_views))
            f.close()
        return metas



    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        intrinsics[:2, :] /= 4.0

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        
        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.total_depths

        depth_interval *= self.interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval


    def read_img(self, filename):
        with Image.open(filename) as img:
            np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    
    
    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=14):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def read_scale_depth(self, filename, scale, max_w, max_h, base=14):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        h, w = depth.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        depth = cv2.resize(depth, (int(new_w), int(new_h)))
        return depth

    