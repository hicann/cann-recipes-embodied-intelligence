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

import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from ..utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_quaternion


@METRICS.register_module()
class SMPLXReconsMetric(BaseMetric):

    def align_by_parts(self, joints, align_inds=None):
        if align_inds is None:
            return joints
        anchor = joints[:, align_inds].mean(dim=1, keepdim=True)
        return joints - anchor

    def calc_mpjpe(self, preds, target, align_inds=(0,), sample_wise=True, trans=None):
        valid_mask = target[:, :, 0] != -2.0

        if align_inds is not None:
            preds_aligned = self.align_by_parts(preds, align_inds=align_inds)
            if trans is not None:
                preds_aligned += trans
            target_aligned = self.align_by_parts(target, align_inds=align_inds)
        else:
            preds_aligned, target_aligned = preds, target
        mpjpe_each = self._compute_mpjpe(
            preds_aligned,
            target_aligned,
            valid_mask=valid_mask,
            sample_wise=sample_wise,
        )
        return mpjpe_each

    def calc_w_mpjpe(self, gt_joints, pred_joints):
        gt_centered = gt_joints - gt_joints[:1, [0]]
        pred_centered = pred_joints - pred_joints[:1, [0]]
        return torch.linalg.norm(pred_centered - gt_centered, dim=-1).mean(dim=-1)

    def batch_compute_similarity_transform_torch(self, s1, s2):
        """Solve batched Procrustes alignment and return transformed points."""
        transposed = s1.shape[0] not in (2, 3)
        if transposed:
            s1 = s1.permute(0, 2, 1)
            s2 = s2.permute(0, 2, 1)

        if s1.shape[1] != s2.shape[1]:
            raise ValueError(
                f"S1 and S2 must have same point count, got {s1.shape[1]} vs {s2.shape[1]}"
            )

        src_mean = s1.mean(dim=-1, keepdim=True)
        dst_mean = s2.mean(dim=-1, keepdim=True)
        src_centered = s1 - src_mean
        dst_centered = s2 - dst_mean

        covariance = src_centered.bmm(dst_centered.transpose(1, 2))
        u_mat, _, vh_mat = torch.linalg.svd(covariance, full_matrices=False)
        v_mat = vh_mat.transpose(1, 2)

        reflect_fix = torch.eye(
            u_mat.shape[1], device=s1.device, dtype=s1.dtype
        ).unsqueeze(0)
        reflect_fix = reflect_fix.repeat(u_mat.shape[0], 1, 1)
        det_sign = torch.sign(torch.det(v_mat.bmm(u_mat.transpose(1, 2))))
        reflect_fix[:, -1, -1] = det_sign

        rot = v_mat.bmm(reflect_fix.bmm(u_mat.transpose(1, 2)))
        src_var = src_centered.pow(2).sum(dim=(1, 2)).clamp_min(1e-8)
        trace_term = torch.einsum("bii->b", rot.bmm(covariance))
        scale = trace_term / src_var
        trans = dst_mean - scale[:, None, None] * rot.bmm(src_mean)
        aligned = scale[:, None, None] * rot.bmm(s1) + trans

        if transposed:
            aligned = aligned.permute(0, 2, 1)

        return aligned, (scale, rot, trans)

    def calc_pampjpe(self, preds, target, sample_wise=True, return_transform_mat=False):
        target, preds = target.float(), preds.float()

        preds_transformed, pa_transform = self.batch_compute_similarity_transform_torch(
            preds, target
        )
        pa_mpjpe_each = self._compute_mpjpe(
            preds_transformed, target, sample_wise=sample_wise
        )
        return pa_mpjpe_each

    def calc_accel(self, preds, target):
        """Compute mean acceleration error over joints for each sequence."""
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must share shape, got {preds.shape} vs {target.shape}"
            )
        if preds.dim() != 3:
            raise ValueError(f"preds must be 3D tensor, got dim={preds.dim()}")

        gt_accel = target[2:] - 2 * target[1:-1] + target[:-2]
        pred_accel = preds[2:] - 2 * preds[1:-1] + preds[:-2]
        return torch.linalg.norm(pred_accel - gt_accel, dim=-1).mean(dim=1)

    @torch.no_grad()
    def process(self, data_batch, data_samples):
        pred_motions, indices = data_samples
        device = (
            pred_motions[0].device
            if len(pred_motions) > 0
            else data_batch["motion"].device
        )
        gt_motions = data_batch["motion"].to(device)
        motion_lengths = data_batch["motion_length"].to(device)
        mean, std = data_batch["mean"].to(device), data_batch["std"].to(device)

        mpjpe, pampjpe, accel, w_mpjpe = [], [], [], []
        for i, pred_motion_item in enumerate(pred_motions):
            seq_len = motion_lengths[i]
            pred_motion = (
                pred_motion_item[:seq_len].squeeze() * std[i][None] + mean[i][None]
            )
            gt_motion = gt_motions[i, :seq_len] * std[i] + mean[i]

            pred_joints = pred_motion[:, 8:8 + 22 * 3].reshape(seq_len, -1, 3)
            pred_joints[:, :, [0, 2]] += torch.cumsum(
                pred_motion[:, :2], dim=0
            ).unsqueeze(1)

            gt_joints = gt_motion[:, 8:8 + 22 * 3].reshape(seq_len, -1, 3)
            gt_joints[:, :, [0, 2]] += torch.cumsum(gt_motion[:, :2], dim=0).unsqueeze(
                1
            )

            w_mpjpe.append(
                self.calc_w_mpjpe(gt_joints * 1000, pred_joints * 1000).mean()
            )
            pampjpe.append(
                self.calc_pampjpe(gt_joints * 1000, pred_joints * 1000).mean()
            )
            accel.append(self.calc_accel(gt_joints * 1000, pred_joints * 1000).mean())
            mpjpe.append(self.calc_mpjpe(gt_joints * 1000, pred_joints * 1000).mean())

        self.results.append(
            {
                "w_mpjpe": w_mpjpe,
                "mpjpe": mpjpe,
                "pampjpe": pampjpe,
                "accel": accel,
                "count": len(mpjpe),
            }
        )

    def compute_metrics(self, results):
        count = sum(item["count"] for item in results)
        mpjpe = sum([i for item in results for i in item["mpjpe"]])
        pampjpe = sum([i for item in results for i in item["pampjpe"]])
        accel = sum([i for item in results for i in item["accel"]])
        w_mpjpe = sum([i for item in results for i in item["w_mpjpe"]])

        return dict(
            mpjpe=mpjpe / count,
            pampjpe=pampjpe / count,
            accel=accel / count,
            w_mpjpe=w_mpjpe / count,
        )

    def _compute_mpjpe(
        self, preds, target, valid_mask=None, pck_joints=None, sample_wise=True
    ):
        """Compute per-joint Euclidean error with optional masking/subset selection."""
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must share shape, got {preds.shape} vs {target.shape}"
            )
        joint_error = torch.linalg.norm(preds - target, ord=2, dim=-1)

        if pck_joints is not None:
            return joint_error[:, pck_joints]

        if sample_wise:
            if valid_mask is None:
                return joint_error.mean(dim=-1)
            valid = valid_mask.float()
            valid_count = valid.sum(dim=-1).clamp_min(1.0)
            return (joint_error * valid).sum(dim=-1) / valid_count

        return joint_error[valid_mask] if valid_mask is not None else joint_error


@METRICS.register_module()
class SMPLXReconsMetricV1(SMPLXReconsMetric):

    def recover_motion(self, motion):
        root_ori_quat = matrix_to_quaternion(
            rotation_6d_to_matrix(motion[:, 3 - 3:9 - 3])
        )
        joints = motion[:, 9 - 3:9 - 3 + 14 * 3].reshape(motion.shape[0], -1, 3)
        return 0, root_ori_quat, joints
