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
import time
import argparse
import logging

import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from sklearn import neighbors as skln
import open3d as o3d
from scipy.spatial import KDTree as KDTree

from dataset_utils.data_io import read_ply


logging.basicConfig(level=logging.INFO)


def max_dist_cp(qto: np.ndarray, qfrom: np.ndarray, max_dist: float) -> np.ndarray:
    tree = KDTree(qto)
    dist, _ = tree.query(qfrom, k=1,
                        distance_upper_bound=max_dist,
                        workers=-1)
    dist[np.isinf(dist)] = max_dist
    return dist


def reduce_pts_o3d(points: np.ndarray, radius: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd_down = pcd.voxel_down_sample(voxel_size=radius)
    return np.asarray(pcd_down.points)


def comput_one_scan(scanid,             # the scan id to be computed 
                    pred_ply,           # predict points cloud file path, such as "./mvsnet001_l3.ply"
                    gt_ply,             # ground truth points cloud file path, such as "./stl001_total.ply"
                    mask_file,          # obsmask file path, decide which parts of 3D space should be used 
                    plane_file,         # plane file path, used to destinguise which Stl points are 'used'
                    ):
    '''Compute accuracy(mm), completeness(mm), overall(mm) for one scan 

        scanid:         the scan id to be computed 
        pred_ply:       predict points cloud file path, such as "./mvsnet001_l3.ply"
        gt_ply:         ground truth points cloud file path, such as "./stl001_total.ply"
        mask_file:      obsmask file path, decide which parts of 3D space should be used for evaluation
        plane_file:     plane file path, used to destinguise which Stl points are 'used'
    '''
    # downsample density, Min dist between points when reducing
    down_dense = 0.2
    # patch size
    patch = 60
    # outlier thresshold of 20 mm
    max_dist = 20
    thresh = down_dense
    pbar = tqdm(total=6)
    pbar.set_description(f'[scan{scanid}] read data pcd')
    data_pcd = read_ply(pred_ply)


    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] downsample pcd')
    
    data_down = reduce_pts_o3d(data_pcd, thresh)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] masking data pcd')
    obs_mask_file = loadmat(mask_file)
    obs_mask, bb_matrix, res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    bb_matrix = bb_matrix.astype(np.float32)

    inbound = ((data_down >= bb_matrix[:1] - patch) & (data_down < bb_matrix[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - bb_matrix[:1]) / res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(obs_mask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = obs_mask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] read STL pcd')
    stl = read_ply(gt_ply)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute data2stl')
    max_dist = max_dist
    ddata = max_dist_cp(stl, data_in_obs, max_dist)
    mean_d2s = np.mean(ddata[ddata < max_dist])

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute stl2data')
    ground_plane = loadmat(plane_file)['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    dstl = max_dist_cp(data_in, stl_above, max_dist)
    mean_s2d = np.mean(dstl[dstl < max_dist])

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    logging.info(f"\t\t\tacc.(mm):{mean_d2s:.4f}, comp.(mm):{mean_s2d:.5f}, overall(mm):{over_all:.4f}")
    return mean_d2s, mean_s2d, over_all


def compute_scans(scans, pred_dir, gt_dir, **kargs):
    t1 = time.time()
    acc, comp, overall = [], [], []
    for scan in scans:
        scanid = int(scan[4:])
        pred_ply = os.path.join(pred_dir, f"{scanid:03}.ply")   
        gt_ply = os.path.join(gt_dir, f"Points/stl/stl{scanid:03}_total.ply")
        mask_file = os.path.join(gt_dir, f'ObsMask/ObsMask{scanid}_10.mat')
        plane_file = os.path.join(gt_dir, f'ObsMask/Plane{scanid}.mat')
        if not os.path.exists(pred_ply):
            raise ValueError(f"File '{pred_ply}' not found")
        if not os.path.exists(gt_ply):
            raise ValueError(f"File '{gt_ply}' not found")
        if not os.path.exists(mask_file):
            raise ValueError(f"File '{mask_file}' not found")
        if not os.path.exists(plane_file):
            raise ValueError(f"File '{plane_file}' not found")
        
        result = comput_one_scan(scanid, pred_ply, gt_ply, mask_file, plane_file)
        acc.append(result[0])
        comp.append(result[1])
        overall.append(result[2])
    mean_acc = np.mean(acc)
    mean_comp = np.mean(comp)
    mean_overall = np.mean(overall)
    t2 = time.time()
    logging.info(f"Finished, total time cost: {t2-t1:.2f}s")
    return mean_acc, mean_comp, mean_overall


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, default='./datasets/lists/test.txt')
    parser.add_argument('--pred_dir', type=str, default='./Predict/mvsnet', help="predict result ply file path")
    parser.add_argument('--gt_dir', type=str, default='./SampleSet/MVS Data', help="groud truth ply file path")
    parser.add_argument('--down_dense', type=float, default=0.2, 
        help="downsample density, Min dist between points when reducing")
    parser.add_argument('--patch', type=float, default=60, help="patch size")
    parser.add_argument('--max_dist', type=float, default=20, help="outlier thresshold of 20 mm")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    with open(args.data_list, 'r') as f:
        all_scenes = f.readlines()
        scans = list(map(str.strip, all_scenes))

    scans = scans
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir

    exclude = ["scans", "data_list", "pred_dir", "gt_dir"]
    args = vars(args)
    args = {key: args[key] for key in args if key not in exclude}
    acc, comp, overall = compute_scans(scans, pred_dir, gt_dir, **args)
    logging.info(f"mean acc:{acc:>12.4f}\nmean comp:{comp:>11.4f}\nmean overall:{overall:>8.4f}")