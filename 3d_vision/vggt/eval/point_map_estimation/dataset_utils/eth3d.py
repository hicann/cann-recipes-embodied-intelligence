# coding=utf-8
# Applied from https://github.com/naver/dust3r/blob/main/dust3r/datasets/base/base_stereo_view_dataset.py
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
import os
import random
from collections import deque
import numpy as np
from einops import rearrange

import torch
import cv2
import PIL

from third_party.dust3r.utils.image import imread_cv2
from third_party.dust3r.utils.image import ImgNorm
from third_party.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dataset_utils.cropping as cropping


class ETH3D:
    def __init__(
        self,
        root_dir,
        resolution,
        num_seq=1,
        num_frames=10,
        transform=ImgNorm,
        shuffle_seed=-1,
    ):
        self.root_dir = root_dir
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.shuffle_seed = shuffle_seed
        self._load_all_scenes(root_dir)
        self._set_resolutions(resolution)
        self.transform = transform
        if isinstance(transform, str):
            transform = eval(transform)
        # set-up the rng
        seed = torch.initial_seed()
        self._rng = np.random.default_rng(seed=seed)

    def __len__(self):
        return len(self.scenes)

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            if len(self._resolutions) != 1:
                raise ValueError("The length of resolutions must be 1 when idx is not a tuple.")
            ar_idx = 0
        # over-loaded code
        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)

        # check data-types
        for v, view in enumerate(views):
            if 'pts3d' in view:
                raise ValueError(f"pts3d should not be there, they will be computed afterwards \
                        based on intrinsics+depthmap for view {view_name(view)}")
            
            view["idx"] = v

            # encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = self.transform(view["img"])

            if "camera_intrinsics" not in view:
                raise ValueError("camera_intrinciss is missing in view.")

            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                if not np.isfinite(view["camera_pose"]).all():
                    raise ValueError(f"NaN in camera pose for view {view_name(view)}")
            if "pts3d" in view:
                raise ValueError(f"pts3d should not be there, they will be computed afterwards \
                        based on intrinsics+depthmap for view {view_name(view)}")
            if "valid_mask" in view:
                raise ValueError(f"valid_mask should not be there, they will be computed \
                    afterwards based on intrinsics+depthmap for view {view_name(view)}")
            if not np.isfinite(view["depthmap"]).all():
                raise ValueError(f"NaN in depthmap for view {view_name(view)}")
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                if not res:
                    raise TypeError(f"{err_msg} with {key}={val} for view {view_name(view)}")
            view["update"] = True
            view["reset"] = False

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        return views


    @staticmethod
    def _read_cameras(path):
        cams = {}
        with open(path) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                toks = line.strip().split()
                cam_id = int(toks[0])
                w, h = map(int, toks[2:4])
                params = list(map(float, toks[4:]))
                fx, fy, cx, cy = params[:4]
                cam_intrins_k = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], np.float32)
                cams[cam_id] = dict(K=cam_intrins_k, size=(w, h))
        return cams

    @staticmethod
    def _qvec2rotmat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ], np.float32)

    @staticmethod
    def _read_images(path):
        extrinsics = {}      # image_name -> (T_world2cam, cam_id)
        with open(path) as f:
            lines = [line for line in f.readlines() if line and not line.startswith('#')]
        for i in range(0, len(lines), 2):
            toks = lines[i].strip().split()
            q = list(map(float, toks[1:5]))
            t = list(map(float, toks[5:8]))
            cam_id = int(toks[8])
            name = toks[9]
            cam_r = ETH3D._qvec2rotmat(q)
            t = np.asarray(t, np.float32).reshape(3, 1)
            tw2c = np.eye(4, dtype=np.float32)
            tw2c[:3, :3] = cam_r
            tw2c[:3, 3] = t[:, 0]
            extrinsics[name] = (tw2c, cam_id)
        return extrinsics

    @staticmethod
    def _read_depth_raw(path, shape):
        h, w = shape
        depth = np.fromfile(path, "<f4")
        depth = rearrange(depth, "(h w) -> h w", h=h, w=w)
        depth[np.isinf(depth)] = 0
        depth[np.isnan(depth)] = 0
        return depth

    @staticmethod
    def _crop_resize_if_necessary(
        image, depth_map, camera_intrinsics, resolution, rng=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        image_w, image_h = image.size
        cx, cy = camera_intrinsics[:2, 2].round().astype(int)

        # calculate min distance to margin
        min_margin_x = min(cx, image_w - cx)
        min_margin_y = min(cy, image_h - cy)
        if min_margin_x <= image_w / 5:
            raise ValueError(f"Bad principal point")
        if min_margin_y <= image_h / 5:
            raise ValueError(f"Bad principal point")

        ## Center crop
        # Crop on the principal point, make it always centered
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)

        image, depth_map, camera_intrinsics = cropping.crop_image_depthmap(
            image, depth_map, camera_intrinsics, crop_bbox
        )

        # # transpose the resolution if necessary
        image_w, image_h = image.size  # new size
        if resolution[0] < resolution[1]:
            raise ValueError("Resolution width must be greater than or equal to resolution height")
        if image_h > 1.1 * image_w:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < image_h / image_w < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        target_resolution = np.array(resolution)
        image, depth_map, camera_intrinsics = cropping.rescale_image_depthmap(
            image, depth_map, camera_intrinsics, target_resolution
        )
        camera_intrinsics2 = cropping.camera_matrix_of_crop(
            camera_intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            camera_intrinsics, camera_intrinsics2, resolution
        )
        image, depth_map, camera_intrinsics = cropping.crop_image_depthmap(
            image, depth_map, camera_intrinsics, crop_bbox
        )
        return image, depth_map, camera_intrinsics

    def get_stats(self):
        return f"{len(self)} pairs"
        
    def _load_all_scenes(self, root):
        self.scene_list = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.scenes = self.scene_list

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]  
        scene_dir = os.path.join(self.root_dir, scene_id)
        calib_dir = os.path.join(scene_dir, 'dslr_calibration_jpg')

        cams = self._read_cameras(os.path.join(calib_dir, 'cameras.txt'))
        exts = self._read_images(os.path.join(calib_dir, 'images.txt'))

        img_names = sorted(exts.keys())
        rng.shuffle(img_names)
        img_names = img_names[: self.num_frames]

        views = []
        for name in img_names:
            img_path = os.path.join(scene_dir, 'images', name)
            depth_fname = os.path.basename(name)
            depth_path = os.path.join(scene_dir, 'ground_truth_depth',
                                      'dslr_images', depth_fname)

            # intrinsics & extrinsics
            tw2c, cam_id = exts[name]
            tc2w = np.linalg.inv(tw2c)
            cam_intrincis_k = cams[cam_id]['K']
            w, h = cams[cam_id]['size']

            # load RGB
            rgb = imread_cv2(img_path)                           # BGR uint8
            if rgb.shape[0] != h or rgb.shape[1] != w:
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

            # load depth
            depth = self._read_depth_raw(depth_path, (h, w))     # float32, metres
            rgb, depth, cam_intrincis_k_ = self._crop_resize_if_necessary(
                    rgb, depth, cam_intrincis_k, resolution, rng=rng
                )

            views.append(dict(
                img=rgb,
                depthmap=depth,
                camera_pose=tc2w,
                camera_intrinsics=cam_intrincis_k_,
                dataset='eth3d',
                label=f"{scene_id}/{name}",
                instance=img_path,
            ))
        return views


    def _set_resolutions(self, resolutions):
        """Set the resolution(s) of the dataset.
        Params:
            - resolutions: int or tuple or list of tuples
        """
        if resolutions is None:
            raise ValueError("undefined resolution")

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            if not isinstance(width, int):
                raise ValueError(f"Bad type for {width=} {type(width)=}, should be int")
            if not isinstance(height, int):
                raise ValueError(f"Bad type for {height=} {type(height)=}, should be int")
            if width < height:
                raise ValueError(f"Width must be greater than or equal to height, got width={width}, height={height}")
            self._resolutions.append((width, height))
    
    
    

def is_good_type(key, v):
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        if view["img"].shape != (3, height, width):
            raise ValueError(f"Unexpected shape for img: {view['img'].shape}, expected: (3, {height}, {width})")
        view["img"] = view["img"].swapaxes(1, 2)

        if view["valid_mask"].shape != (height, width):
            raise ValueError(f"Unexpected shape for valid_mask: {view['valid_mask'].shape}, \
                expected: ({height}, {width})")
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        if view["depthmap"].shape != (height, width):
            raise ValueError(f"Unexpected shape for depthmap: {view['depthmap'].shape}, expected: ({height}, {width})")
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        if view["pts3d"].shape != (height, width, 3):
            raise ValueError(f"Unexpected shape for pts3d: {view['pts3d'].shape}, expected: ({height}, {width}, 3)")
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)


        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]