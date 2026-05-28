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
"""
端到端推理脚本：SMPL/SMPL-X → G1 DOF/root_trans/root_rot

支持输入格式：
  - SMPL-X NPZ: 含 global_orient, body_pose, transl
  - SMPL PKL: 含 fullpose(72维), trans

用法:
    PYTHONPATH=. python tools/inference.py <config> <checkpoint> --src <file_or_dir> [--output-dir <dir>] [--no-filter]
"""
import argparse
from copy import deepcopy
from dataclasses import dataclass
import glob
import logging
import os
import sys
import time

import joblib
from mmengine.config import Config
from mmengine.registry import MODELS
import numpy as np
from scipy import signal as sp_signal
from smplx import SMPLX
import src
from src.utils.rotation_conversions import (
    axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d,
    rotation_6d_to_matrix, matrix_to_quaternion,
)
import torch

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

# PyTorch 2.6 兼容 patch
_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    return _torch_load(
        *args,
        weights_only=kwargs.pop('weights_only', False),
        **kwargs,
    )


torch.load = _patched_torch_load


logger = logging.getLogger(__name__)


def load_smpl_data(file_path):
    """从 NPZ 或 PKL 加载 SMPL/SMPL-X 运动参数。

    支持两种格式：
    - NPZ: 含 global_orient(T,3), body_pose(T,63), transl(T,3)
    - PKL: 含 fullpose(T,72), trans(T,3)  (SMPL格式，取前21个body关节)
    """
    if file_path.endswith('.pkl'):
        data = joblib.load(file_path)
        transl = torch.from_numpy(np.array(data['trans'])).float()
        fullpose = torch.from_numpy(np.array(data['fullpose'])).float()
        global_orient = fullpose[:, :3]
        body_pose = fullpose[:, 3:66]  # 21 joints, 丢弃 SMPL 的手部关节

        # PKL 数据是 Z-up，转换到 Y-up（绕 X 轴旋转 -90°）
        rot_zup_to_yup = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).float()
        transl = torch.einsum('ij,tj->ti', rot_zup_to_yup, transl)
        # global_orient 是 axis-angle，转换为矩阵后左乘坐标变换
        go_mat = axis_angle_to_matrix(global_orient)  # (T, 3, 3)
        go_mat = torch.einsum('ij,tjk->tik', rot_zup_to_yup, go_mat)
        global_orient = matrix_to_axis_angle(go_mat)

        # PKL 数据是 120 FPS，降采样到 30 FPS
        transl = transl[::4]
        global_orient = global_orient[::4]
        body_pose = body_pose[::4]

        return transl, global_orient, body_pose
    else:
        data = np.load(file_path)
        transl = torch.from_numpy(data['transl']).float()
        global_orient = torch.from_numpy(data['global_orient']).float()
        body_pose = torch.from_numpy(data['body_pose']).float()
        return transl, global_orient, body_pose


def preprocess_smpl(file_path, smplx_model, betas, device):
    """将 SMPL/SMPL-X 运动文件转换为 (T, 140) 运动特征向量。

    逻辑与 data/process.py:get_smplx_motion() 一致。
    """
    transl, global_orient, body_pose = load_smpl_data(file_path)

    num_frames = transl.shape[0]
    frame_params = dict(
        transl=transl.to(device),
        global_orient=global_orient.to(device),
        body_pose=body_pose.to(device),
        betas=betas.unsqueeze(0).repeat(num_frames, 1).float().to(device),
        leye_pose=torch.zeros((num_frames, 3), device=device),
        reye_pose=torch.zeros((num_frames, 3), device=device),
        left_hand_pose=torch.zeros((num_frames, 45), device=device),
        right_hand_pose=torch.zeros((num_frames, 45), device=device),
        jaw_pose=torch.zeros((num_frames, 3), device=device),
        expression=torch.zeros((num_frames, 100), device=device),
    )

    with torch.no_grad():
        output = smplx_model(**frame_params)
    position_data = output.joints.detach().cpu()[:, :22]  # (T, 22, 3)

    global_orient_mat = axis_angle_to_matrix(global_orient)  # (T, 3, 3)

    position_val_data = position_data[1:] - position_data[:-1]

    root_idx = 0
    y_min = torch.min(position_data[:, :, 1])
    ori = deepcopy(position_data[0, root_idx])
    ori[1] = y_min
    position_data = position_data - ori

    velocities_root = position_data[1:, root_idx, :] - position_data[:-1, root_idx, :]
    position_data_cp = deepcopy(position_data)
    position_data[:, :, 0] -= position_data_cp[:, 0:1, 0]
    position_data[:, :, 2] -= position_data_cp[:, 0:1, 2]

    num_frames, njoint, _ = position_data.shape
    final_x = torch.zeros((num_frames, 2 + 6 + njoint * 3 + njoint * 3))
    final_x[1:, 0] = velocities_root[:, 0]
    final_x[1:, 1] = velocities_root[:, 2]
    final_x[:, 2:8] = matrix_to_rotation_6d(global_orient_mat)
    final_x[:, 8:8 + njoint * 3] = position_data.flatten(1, 2)
    final_x[1:, 8 + njoint * 3:8 + njoint * 6] = position_val_data.flatten(1, 2)

    return final_x  # (T, 140)


def postprocess_g1(pred_motion, apply_filter=True):
    """从 G1 217 维运动向量提取 DOF/root_trans/root_rot。

    逻辑与 src/metrics/humanoid_metric.py:92-118 一致。
    """
    num_frames = pred_motion.shape[0]
    rot_mat = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).float()

    pred_trans = pred_motion[:, 8:8 + 30 * 3].reshape(num_frames, -1, 3)[:, 0]
    pred_trans[:, [0, 2]] += torch.cumsum(pred_motion[:, :2], dim=0)
    pred_trans = torch.einsum('ij,tj->ti', rot_mat, pred_trans)

    pred_rot_mat = rotation_6d_to_matrix(pred_motion[:, 2:8])
    pred_rot_mat = torch.einsum('ij,tjk->tik', rot_mat, pred_rot_mat)
    pred_rot_quat = matrix_to_quaternion(pred_rot_mat)

    pred_dof = pred_motion[:, -29:]

    if apply_filter and num_frames >= 13:  # filtfilt 需要足够长的序列
        _b, _a = sp_signal.butter(4, 5 / (30.0 / 2), btype='low')
        pred_trans_device, pred_trans_dtype = pred_trans.device, pred_trans.dtype
        pred_rot_quat_device, pred_rot_quat_dtype = pred_rot_quat.device, pred_rot_quat.dtype
        pred_dof_device, pred_dof_dtype = pred_dof.device, pred_dof.dtype
        pred_trans = torch.from_numpy(
            sp_signal.filtfilt(_b, _a, pred_trans.detach().cpu().numpy(), axis=0).copy()
        ).to(device=pred_trans_device, dtype=pred_trans_dtype)

        pred_rot_quat_np = sp_signal.filtfilt(
            _b, _a, pred_rot_quat.detach().cpu().numpy(), axis=0
        ).copy()
        pred_rot_quat_np /= np.linalg.norm(pred_rot_quat_np, axis=-1, keepdims=True)
        pred_rot_quat = torch.from_numpy(pred_rot_quat_np).to(
            device=pred_rot_quat_device, dtype=pred_rot_quat_dtype
        )

        pred_dof = torch.from_numpy(
            sp_signal.filtfilt(_b, _a, pred_dof.detach().cpu().numpy(), axis=0).copy()
        ).to(device=pred_dof_device, dtype=pred_dof_dtype)

    return pred_dof, pred_rot_quat, pred_trans


def _find_path(
    cfg,
    key,
    fallback_sections=('test_dataloader.dataset', 'val_dataloader.dataset', 'train_dataloader.dataset'),
):
    """从配置中查找路径，优先用顶层，不存在则从 dataloader.dataset 中取。"""
    path = cfg.get(key, None)
    if path and os.path.exists(path):
        return path
    for section in fallback_sections:
        obj = cfg
        for part in section.split('.'):
            obj = obj.get(part, {})
        path = obj.get(key, None)
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f'找不到配置项 {key} 对应的文件')


@dataclass
class ModelBundle:
    model: torch.nn.Module
    cfg: Config
    smplx_mean: torch.Tensor
    smplx_std: torch.Tensor
    g1_mean: torch.Tensor
    g1_std: torch.Tensor


@dataclass
class InferChunkContext:
    model: torch.nn.Module
    smplx_mean: torch.Tensor
    smplx_std: torch.Tensor
    g1_mean: torch.Tensor
    g1_std: torch.Tensor
    device: torch.device


@dataclass
class InferSingleContext:
    chunk_context: InferChunkContext
    smplx_model: SMPLX
    betas: torch.Tensor


def load_model(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(device)

    smplx_mean = torch.from_numpy(np.load(_find_path(cfg, 'smplx_mean_path'))).float()
    smplx_std = torch.from_numpy(np.load(_find_path(cfg, 'smplx_std_path'))).float()
    g1_mean = torch.from_numpy(np.load(_find_path(cfg, 'g1_mean_path'))).float()
    g1_std = torch.from_numpy(np.load(_find_path(cfg, 'g1_std_path'))).float()

    return ModelBundle(
        model=model,
        cfg=cfg,
        smplx_mean=smplx_mean,
        smplx_std=smplx_std,
        g1_mean=g1_mean,
        g1_std=g1_std,
    )


FPS = 30
CHUNK_SECONDS = 4
CHUNK_FRAMES = (FPS * CHUNK_SECONDS // 4) * 4  # 120，对齐到4的倍数
OVERLAP_FRAMES = 32  # 重叠帧数（约1秒），必须是4的倍数
STRIDE_FRAMES = CHUNK_FRAMES - OVERLAP_FRAMES  # 滑动步长


def _make_y_rot(theta):
    """构建绕 Y 轴旋转 theta 弧度的 3x3 矩阵。"""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def _extract_yaw(rot_6d):
    """从 6D 旋转表示提取 Y 轴旋转角（yaw）。rot_6d: (6,)"""
    rot_matrix = rotation_6d_to_matrix(rot_6d.unsqueeze(0))[0]  # (3, 3)
    forward = rot_matrix[:, 2]  # Z 轴列
    return torch.atan2(forward[0], forward[2])


def _rotate_motion_features(motion, rot_mat_y, n_joints, rotate_6d=True):
    """将运动特征绕 Y 轴旋转。适用于 SMPLX(140维, 22关节) 和 G1(217维, 30关节)。

    特征布局: [0:2] root XZ vel | [2:8] 6D rot | [8:8+nj*3] pos | [...+nj*3] vel | (可选 DOF)
    """
    result = motion.clone()
    # root XZ velocity
    vx, vz = motion[:, 0], motion[:, 1]
    result[:, 0] = rot_mat_y[0, 0] * vx + rot_mat_y[0, 2] * vz
    result[:, 1] = rot_mat_y[2, 0] * vx + rot_mat_y[2, 2] * vz
    # 6D rotation
    if rotate_6d:
        rot_mat = rotation_6d_to_matrix(motion[:, 2:8])
        rot_mat = torch.einsum('ij,tjk->tik', rot_mat_y, rot_mat)
        result[:, 2:8] = matrix_to_rotation_6d(rot_mat)
    # joint positions
    pos_start, pos_end = 8, 8 + n_joints * 3
    pos = motion[:, pos_start:pos_end].reshape(-1, n_joints, 3)
    pos = torch.einsum('ij,tnj->tni', rot_mat_y, pos)
    result[:, pos_start:pos_end] = pos.reshape(-1, n_joints * 3)
    # joint velocities
    vel_start, vel_end = pos_end, pos_end + n_joints * 3
    if vel_end <= motion.shape[1]:
        vel = motion[:, vel_start:vel_end].reshape(-1, n_joints, 3)
        vel = torch.einsum('ij,tnj->tni', rot_mat_y, vel)
        result[:, vel_start:vel_end] = vel.reshape(-1, n_joints * 3)
    return result


def _infer_chunk(smplx_motion, context):
    """对单段 smplx_motion (T, 140) 做旋转规范化→标准化→推理→反标准化→旋转还原，返回 (T, 217)。"""
    model = context.model
    smplx_mean = context.smplx_mean
    smplx_std = context.smplx_std
    g1_mean = context.g1_mean
    g1_std = context.g1_std
    device = context.device

    # 提取首帧 yaw，旋转到规范朝向（yaw=0）
    yaw = _extract_yaw(smplx_motion[0, 2:8])
    r_canon = _make_y_rot(-yaw)  # 旋转到规范方向
    r_restore = _make_y_rot(yaw)  # 旋转回原方向

    smplx_motion = _rotate_motion_features(smplx_motion, r_canon, n_joints=22)

    smplx_motion = (smplx_motion - smplx_mean) / smplx_std
    smplx_input = smplx_motion.unsqueeze(0).float().to(device)
    motion_length = torch.tensor([smplx_motion.shape[0]]).to(device)

    with torch.no_grad():
        pred_motions, _ = model(smplx_motion=smplx_input, motion_length=motion_length, mode='predict')

    pred_motion = pred_motions[0].cpu()
    pred_motion = pred_motion * g1_std + g1_mean

    # 将 G1 输出旋转回原始朝向（DOF 不受旋转影响）
    pred_motion = _rotate_motion_features(pred_motion, r_restore, n_joints=30)
    return pred_motion


def infer_single(file_path, context, apply_filter=True):
    t0 = time.time()
    smplx_motion = preprocess_smpl(
        file_path,
        context.smplx_model,
        context.betas,
        context.chunk_context.device,
    )

    total_frames = smplx_motion.shape[0]
    aligned_frames = (total_frames // 4) * 4
    if aligned_frames < 4:
        logger.info("  跳过 %s: 序列太短 (%d 帧)", file_path, total_frames)
        return None
    smplx_motion = smplx_motion[:aligned_frames]
    t_preprocess = time.time() - t0

    t1 = time.time()
    pred_motion = _infer_aligned_motion(smplx_motion, aligned_frames, context.chunk_context)
    t_infer = time.time() - t1

    t2 = time.time()
    pred_dof, pred_rot_quat, pred_trans = postprocess_g1(pred_motion, apply_filter=apply_filter)
    t_postprocess = time.time() - t2

    t_total = time.time() - t0
    timing = dict(preprocess=t_preprocess, infer=t_infer, postprocess=t_postprocess, total=t_total)

    return dict(
        dof=pred_dof.numpy(),              # (T, 29)
        root_trans=pred_trans.numpy(),      # (T, 3)
        root_rot_quat=pred_rot_quat.numpy(),  # (T, 4) wxyz
        source_path=file_path,
    ), timing


def _infer_aligned_motion(smplx_motion, aligned_frames, context):
    if aligned_frames <= CHUNK_FRAMES:
        return _infer_chunk(smplx_motion, context)

    chunks, starts = _infer_overlapping_chunks(smplx_motion, aligned_frames, context)
    return _blend_overlapping_chunks(chunks, starts)


def _infer_overlapping_chunks(smplx_motion, aligned_frames, context):
    chunks = []
    starts = []
    for start in range(0, aligned_frames, STRIDE_FRAMES):
        end = min(start + CHUNK_FRAMES, aligned_frames)
        seg_len = (end - start) // 4 * 4
        if seg_len < 4:
            break
        chunk = smplx_motion[start:start + seg_len]
        chunks.append(_infer_chunk(chunk, context))
        starts.append(start)
    return chunks, starts


def _blend_overlapping_chunks(chunks, starts):
    pred_motion = chunks[0]
    for i in range(1, len(chunks)):
        overlap = starts[i - 1] + len(chunks[i - 1]) - starts[i]
        if overlap > 0:
            w = torch.linspace(0, 1, overlap).unsqueeze(1)
            prev_tail = pred_motion[-overlap:]
            curr_head = chunks[i][:overlap]
            blended = prev_tail * (1 - w) + curr_head * w
            pred_motion = torch.cat([pred_motion[:-overlap], blended, chunks[i][overlap:]], dim=0)
        else:
            pred_motion = torch.cat([pred_motion, chunks[i]], dim=0)
    return pred_motion


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _validate_model_paths(args)

    t_load_start = time.time()
    logger.info('加载模型...')
    infer_context = _load_infer_context(args, device)
    t_load = time.time() - t_load_start
    logger.info('完成 (%.2fs)', t_load)

    input_files, base_dir = _collect_input_files(args.src)
    if not input_files:
        logger.info('未找到输入文件: %s', args.src)
        return

    logger.info('找到 %d 个输入文件', len(input_files))
    os.makedirs(args.output_dir, exist_ok=True)
    n_processed, t_all = _run_inference_batch(input_files, base_dir, args, infer_context)

    logger.info('%s', '=' * 40)
    avg = t_all / n_processed if n_processed > 0 else 0
    logger.info('模型加载: %.2fs', t_load)
    logger.info('完成! 共 %d 个文件, 推理总耗时 %.2fs, 平均 %.2fs/文件', n_processed, t_all, avg)


def _parse_args():
    parser = argparse.ArgumentParser(description='SMPL-X → G1 推理脚本')
    parser.add_argument('config', help='mmengine 配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')
    parser.add_argument('--src', required=True, help='输入文件(NPZ/PKL)或目录')
    parser.add_argument('--output-dir', default='inference_output', help='输出目录')
    parser.add_argument('--no-filter', action='store_true', help='禁用低通滤波')
    parser.add_argument(
        '--smplx-model-path',
        default='checkpoints/human_model/SMPLX_NEUTRAL.npz',
        help='SMPLX 模型文件路径 (npz)',
    )
    parser.add_argument('--betas-path', default='checkpoints/humanoid_model/g1/betas.npy', help='G1 betas 文件路径 (npy)')
    return parser.parse_args()


def _validate_model_paths(args):
    if not os.path.exists(args.smplx_model_path):
        raise FileNotFoundError(
            f"SMPLX 模型文件不存在: {args.smplx_model_path}\n请参考 README 下载并放置到指定路径。"
        )
    if not os.path.exists(args.betas_path):
        raise FileNotFoundError(
            f"G1 betas 文件不存在: {args.betas_path}\n请参考 README 下载并放置到指定路径。"
        )


def _load_infer_context(args, device):
    model_bundle = load_model(args.config, args.checkpoint, device)
    model = model_bundle.model
    smplx_mean = model_bundle.smplx_mean
    smplx_std = model_bundle.smplx_std
    g1_mean = model_bundle.g1_mean
    g1_std = model_bundle.g1_std

    # 加载 SMPLX body model
    smplx_model = SMPLX(
        model_path=args.smplx_model_path,
        use_pca=False, num_expression_coeffs=100, num_betas=10, ext='npz'
    ).to(device).eval()
    betas = torch.from_numpy(np.load(args.betas_path)).float()
    return InferSingleContext(
        chunk_context=InferChunkContext(
            model=model,
            smplx_mean=smplx_mean,
            smplx_std=smplx_std,
            g1_mean=g1_mean,
            g1_std=g1_std,
            device=device,
        ),
        smplx_model=smplx_model,
        betas=betas,
    )


def _collect_input_files(src_path):
    if os.path.isfile(src_path):
        input_files = [src_path]
        base_dir = os.path.dirname(src_path)
    else:
        npz_files = sorted(glob.glob(os.path.join(src_path, '**', '*.npz'), recursive=True))
        pkl_files = sorted(glob.glob(os.path.join(src_path, '**', '*.pkl'), recursive=True))
        input_files = npz_files + pkl_files
        base_dir = src_path
    return input_files, base_dir


def _run_inference_batch(input_files, base_dir, args, infer_context):
    t_all_start = time.time()
    n_processed = 0
    for file_path in input_files:
        if _process_input_file(file_path, base_dir, args, infer_context):
            n_processed += 1
    return n_processed, time.time() - t_all_start


def _process_input_file(file_path, base_dir, args, infer_context):
    rel_path = os.path.relpath(file_path, base_dir)
    output_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + '.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info('处理: %s', rel_path)
    ret = infer_single(file_path, infer_context, apply_filter=not args.no_filter)
    if ret is None:
        return False

    result, timing = ret
    joblib.dump(result, output_path)
    logger.info(
        '  预处理: %.2fs | 推理: %.2fs | 后处理: %.2fs | 总计: %.2fs',
        timing['preprocess'],
        timing['infer'],
        timing['postprocess'],
        timing['total'],
    )
    logger.info(
        '  → %s  (dof: %s, trans: %s)',
        output_path,
        result['dof'].shape,
        result['root_trans'].shape,
    )
    return True


if __name__ == '__main__':
    main()
