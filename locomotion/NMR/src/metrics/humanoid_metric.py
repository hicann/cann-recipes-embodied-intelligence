# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import NamedTuple, Optional

import joblib
import numpy as np
import pinocchio as pin
from scipy import signal as sp_signal
import torch
from mmengine.registry import METRICS

from ..utils.rotation_conversions import matrix_to_quaternion, rotation_6d_to_matrix
from .smplx_metric import SMPLXReconsMetric


def _compute_mpja(q_np, fps):
    """Mean Per-Joint Acceleration: mean(abs(d2q / dt2))."""
    if q_np.shape[0] < 3:
        return np.nan
    accel = np.diff(q_np, n=2, axis=0) * (fps**2)
    return float(np.mean(np.abs(accel)))


def _compute_mpjj(q_np, fps):
    """Mean Per-Joint Jerk: mean(abs(d3q / dt3))."""
    if q_np.shape[0] < 4:
        return np.nan
    jerk = np.diff(q_np, n=3, axis=0) * (fps**3)
    return float(np.mean(np.abs(jerk)))


joint_mapping = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
body_mapping = [
    4,
    20,
    34,
    6,
    22,
    36,
    8,
    24,
    38,
    10,
    26,
    46,
    62,
    12,
    28,
    48,
    64,
    14,
    30,
    50,
    66,
    52,
    68,
    54,
    70,
    56,
    72,
    58,
    74,
]


class RobotMotionParts(NamedTuple):
    trans: torch.Tensor
    rot_quat: torch.Tensor
    dof: torch.Tensor
    joints: torch.Tensor


class PoseMetricBuckets(NamedTuple):
    w_mpjpe: list
    pampjpe: list
    accel: list
    mpjpe: list


@METRICS.register_module()
class HumanoidReconsMetric(SMPLXReconsMetric):

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: str | None = None,
        collect_dir: str | None = None,
    ) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        urdf_path = (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "GMR"
            / "assets"
            / "unitree_g1"
            / "g1_custom_collision_29dof.urdf"
        )
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"未找到 URDF 文件: {urdf_path}\n请确认已正确下载 GMR 子模块并放置到上述路径。详见 README 安装指引。"
            )
        self.model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())

    @staticmethod
    def _append_dof_jitter_metrics(gt_dof, pred_dof, dof_accel_ratios, dof_jerk_ratios):
        frame_rate = 30
        pred_dof_np = pred_dof.cpu().numpy()
        gt_dof_np = gt_dof.cpu().numpy()
        pred_mpja = _compute_mpja(pred_dof_np, frame_rate)
        gt_mpja = _compute_mpja(gt_dof_np, frame_rate)
        pred_mpjj = _compute_mpjj(pred_dof_np, frame_rate)
        gt_mpjj = _compute_mpjj(gt_dof_np, frame_rate)
        dof_accel_ratios.append(
            pred_mpja / gt_mpja
            if (not np.isnan(gt_mpja) and abs(gt_mpja) > 1e-12)
            else np.nan
        )
        dof_jerk_ratios.append(
            pred_mpjj / gt_mpjj
            if (not np.isnan(gt_mpjj) and abs(gt_mpjj) > 1e-12)
            else np.nan
        )

    @staticmethod
    def _build_sample(gt_motion, gt_parts, pred_parts, source_path):
        return dict(
            pred_g1_root_ori_quat=pred_parts.rot_quat.cpu(),
            pred_g1_dof=pred_parts.dof.cpu(),
            pred_g1_trans=pred_parts.trans.cpu(),
            gt_g1_root_ori_quat=gt_parts.rot_quat.cpu(),
            gt_g1_dof=gt_parts.dof.cpu(),
            gt_g1_trans=gt_parts.trans.cpu(),
            max_root_vel=torch.norm(gt_motion[:, :2], dim=-1).max().item(),
            source_path=source_path,
        )

    @staticmethod
    def _filter_prediction_parts(pred_trans, pred_rot_quat, pred_dof):
        _b, _a = sp_signal.butter(4, 5 / (30.0 / 2), btype="low")
        min_len = 3 * max(len(_a), len(_b))
        if pred_trans.shape[0] >= min_len:
            pred_trans = torch.from_numpy(
                sp_signal.filtfilt(_b, _a, pred_trans.cpu().numpy(), axis=0).copy()
            ).to(pred_trans.dtype)
        if pred_rot_quat.shape[0] >= min_len:
            pred_rot_quat_np = sp_signal.filtfilt(
                _b, _a, pred_rot_quat.cpu().numpy(), axis=0
            ).copy()
            pred_rot_quat_np /= np.linalg.norm(pred_rot_quat_np, axis=-1, keepdims=True)
            pred_rot_quat = torch.from_numpy(pred_rot_quat_np).to(pred_rot_quat.dtype)
        if pred_dof.shape[0] >= min_len:
            pred_dof = torch.from_numpy(
                sp_signal.filtfilt(_b, _a, pred_dof.cpu().numpy(), axis=0).copy()
            ).to(pred_dof.dtype)
        return pred_trans, pred_rot_quat, pred_dof

    def get_position_from_dof(self, dof, rot_quat, transl):
        # build pinocchio model with free-flyer floating base
        data = self.model.createData()

        mapped_joint = torch.zeros((dof.shape[0], len(joint_mapping)))
        mapped_joint[:, joint_mapping] = dof

        # 计算正运动学并将世界坐标系位置信息保存到一个新的数组中
        num_frames = dof.shape[0]

        joint_positions = torch.zeros(
            (num_frames, len(body_mapping), 3)
        )  # 1 root + 29 DOF joints
        for t in range(num_frames):
            q = np.zeros(self.model.nq)
            q[0:3] = transl[t].numpy()  # 根位置
            q[3:7] = rot_quat[
                t, [1, 2, 3, 0]
            ].numpy()  # 根旋转（四元数，pinocchio使用xyzw顺序）
            q[7:] = mapped_joint[t].numpy()  # 关节角度

            pin.forwardKinematics(self.model, data, q)
            pin.updateFramePlacements(self.model, data)

            for lab_idx, pino_idx in enumerate(body_mapping):
                joint_positions[t, lab_idx] = torch.from_numpy(
                    data.oMf[pino_idx].translation
                )

        return joint_positions  # (T, njoints, 3)

    @torch.no_grad()
    def process(self, data_batch, data_samples):
        pred_motions, indices = data_samples
        device = (
            pred_motions.device
            if hasattr(pred_motions, "device")
            else pred_motions[0].device
        )
        gt_motions = data_batch["motion"].to(device)
        motion_lengths = data_batch["motion_length"].to(device)
        mean, std = data_batch["mean"].to(device), data_batch["std"].to(device)
        mpjpe, pampjpe, accel, w_mpjpe, samples = [], [], [], [], []
        pose_buckets = PoseMetricBuckets(w_mpjpe, pampjpe, accel, mpjpe)
        dof_accel_ratios, dof_jerk_ratios = [], []
        for i, pred_motion_item in enumerate(pred_motions):
            seq_len = motion_lengths[i]
            pred_motion = (
                pred_motion_item[:seq_len].squeeze() * std[i][None] + mean[i][None]
            )
            gt_motion = gt_motions[i, :seq_len] * std[i] + mean[i]

            gt_parts = self._motion_to_robot_parts(gt_motion, seq_len)
            pred_parts = self._build_predicted_parts(pred_motion, seq_len)

            self._append_pose_metrics(gt_parts.joints, pred_parts.joints, pose_buckets)
            self._append_dof_jitter_metrics(
                gt_parts.dof, pred_parts.dof, dof_accel_ratios, dof_jerk_ratios
            )

            source_path = data_batch.get("source_paths", [""] * len(pred_motions))[i]
            samples.append(
                self._build_sample(gt_motion, gt_parts, pred_parts, source_path)
            )

        self.results.append(
            {
                "w_mpjpe": w_mpjpe,
                "mpjpe": mpjpe,
                "pampjpe": pampjpe,
                "accel": accel,
                "dof_accel_ratio": dof_accel_ratios,
                "dof_jerk_ratio": dof_jerk_ratios,
                "count": len(mpjpe),
                "samples": samples,
            }
        )

    def compute_metrics(self, results):
        all_samples = [s for item in results for s in item["samples"]]
        all_samples = all_samples[:200]
        joblib.dump(all_samples, "test_results_wiz_filter_2.pkl")

        count = sum(item["count"] for item in results)
        mpjpe = sum(i for item in results for i in item["mpjpe"])
        pampjpe = sum(i for item in results for i in item["pampjpe"])
        accel = sum(i for item in results for i in item["accel"])
        w_mpjpe = sum(i for item in results for i in item["w_mpjpe"])
        dof_accel_ratios = []
        for item in results:
            for value in item["dof_accel_ratio"]:
                if not np.isnan(value):
                    dof_accel_ratios.append(value)

        dof_jerk_ratios = []
        for item in results:
            for value in item["dof_jerk_ratio"]:
                if not np.isnan(value):
                    dof_jerk_ratios.append(value)

        return dict(
            mpjpe=mpjpe / count,
            pampjpe=pampjpe / count,
            accel=accel / count,
            w_mpjpe=w_mpjpe / count,
            dof_accel_ratio=(
                float(np.mean(dof_accel_ratios)) if dof_accel_ratios else float("nan")
            ),
            dof_jerk_ratio=(
                float(np.mean(dof_jerk_ratios)) if dof_jerk_ratios else float("nan")
            ),
        )

    def _build_predicted_parts(self, pred_motion, seq_len):
        pred_parts = self._motion_to_robot_parts(pred_motion, seq_len)
        pred_trans, pred_rot_quat, pred_dof = self._filter_prediction_parts(
            pred_parts.trans, pred_parts.rot_quat, pred_parts.dof
        )
        pred_joints = self.get_position_from_dof(
            pred_dof.cpu(), pred_rot_quat.cpu(), pred_trans.cpu()
        )
        return RobotMotionParts(pred_trans, pred_rot_quat, pred_dof, pred_joints)

    def _motion_to_robot_parts(self, motion, seq_len):
        rot_mat = (
            torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).float().to(motion.device)
        )
        trans = motion[:, 8:8 + 30 * 3].reshape(seq_len, -1, 3)[:, 0]
        trans[:, [0, 2]] += torch.cumsum(motion[:, :2], dim=0)
        trans = torch.einsum("ij,tj->ti", rot_mat, trans)
        rot_mat_seq = rotation_6d_to_matrix(motion[:, 2:8])
        rot_mat_seq = torch.einsum("ij,tjk->tik", rot_mat, rot_mat_seq)
        rot_quat = matrix_to_quaternion(rot_mat_seq)
        dof = motion[:, -29:]
        joints = self.get_position_from_dof(dof.cpu(), rot_quat.cpu(), trans.cpu())
        return RobotMotionParts(trans, rot_quat, dof, joints)

    def _append_pose_metrics(self, gt_joints, pred_joints, buckets):
        buckets.w_mpjpe.append(
            self.calc_w_mpjpe(gt_joints * 1000, pred_joints * 1000).mean()
        )
        buckets.pampjpe.append(
            self.calc_pampjpe(gt_joints * 1000, pred_joints * 1000).mean()
        )
        buckets.accel.append(
            self.calc_accel(gt_joints * 1000, pred_joints * 1000).mean()
        )
        buckets.mpjpe.append(
            self.calc_mpjpe(gt_joints * 1000, pred_joints * 1000).mean()
        )
