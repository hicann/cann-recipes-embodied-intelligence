# coding=utf-8
# # Adapted from https://github.com/doubleZ0108/GeoMVSNet/blob/master/datasets/data_io.py & 
#           https://github.com/doubleZ0108/GeoMVSNet/blob/master/models/utils/utils.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2022 Zhenxing Mi. All rights reserved.
# License under Apache.
import sys
import re
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import torch


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + \
         ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


def read_ply(file):
    data = PlyData.read(file)
    vertex = data['vertex']
    data_pcd = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    return data_pcd
    

def write_ply(file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(file, pcd, write_ascii=False)


def parse_cameras(path):
    with open(path, 'r') as file:
        cam_txt = file.readlines()
        
    def parse_matrix(xs):
        return list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = parse_matrix(cam_txt[1:5])
    intr_mat = parse_matrix(cam_txt[7:10])

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(input_vars):
        if isinstance(input_vars, list):
            return [wrapper(x) for x in input_vars]
        elif isinstance(input_vars, tuple):
            return tuple([wrapper(x) for x in input_vars])
        elif isinstance(input_vars, dict):
            return {k: wrapper(v) for k, v in input_vars.items()}
        else:
            return func(input_vars)
    return wrapper
    
    
@make_recursive_func
def tocuda(input_vars):
    if isinstance(input_vars, torch.Tensor):
        return input_vars.to(torch.device("cuda"))
    elif isinstance(input_vars, str):
        return input_vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(input_vars)))