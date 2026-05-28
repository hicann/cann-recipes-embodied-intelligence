# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Author: NJU-3DV
# Copyright (c) 2023, ankile. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: MIT AND BSD-3-Clause AND Apache-2.0
#
# Portions are adapted from ankile/robust-rearrangement under MIT and
# Meta/Facebook PyTorch3D-style rotation utilities under BSD-3-Clause.
#
# This file is not Apache-2.0-only. Use and redistribution of third-party
# portions are subject to their upstream licenses. NMR modifications are
# licensed under Apache-2.0. See THIRD_PARTY_LICENSES.md.
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


from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np


"""Utilities for converting among quaternion, matrix, Euler and axis-angle."""


def _half_angle_ratio_torch(angles: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    half = 0.5 * angles
    out = torch.empty_like(angles)
    mask = angles.abs() >= eps
    out[mask] = torch.sin(half[mask]) / angles[mask]
    out[~mask] = 0.5 - (angles[~mask] * angles[~mask]) / 48
    return out


def _half_angle_ratio_np(angles: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    half = 0.5 * angles
    out = np.empty_like(angles)
    mask = np.abs(angles) >= eps
    out[mask] = np.sin(half[mask]) / angles[mask]
    out[~mask] = 0.5 - (angles[~mask] * angles[~mask]) / 48.0
    return out


def quaternion_to_matrix(quaternions):
    q = quaternions
    norm = (q * q).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    s = 2.0 / norm
    qw, qx, qy, qz = torch.unbind(q, dim=-1)

    xx = qx * qx * s[..., 0]
    yy = qy * qy * s[..., 0]
    zz = qz * qz * s[..., 0]
    xy = qx * qy * s[..., 0]
    xz = qx * qz * s[..., 0]
    yz = qy * qz * s[..., 0]
    wx = qw * qx * s[..., 0]
    wy = qw * qy * s[..., 0]
    wz = qw * qz * s[..., 0]

    m00 = 1.0 - yy - zz
    m01 = xy - wz
    m02 = xz + wy
    m10 = xy + wz
    m11 = 1.0 - xx - zz
    m12 = yz - wx
    m20 = xz - wy
    m21 = yz + wx
    m22 = 1.0 - xx - yy

    return torch.stack(
        [
            torch.stack((m00, m01, m02), dim=-1),
            torch.stack((m10, m11, m12), dim=-1),
            torch.stack((m20, m21, m22), dim=-1),
        ],
        dim=-2,
    )


def quaternion_to_matrix_np(quaternions):
    norm = np.sum(quaternions * quaternions, axis=-1, keepdims=True)
    scale = 2.0 / np.clip(norm, 1e-12, None)
    qw, qx, qy, qz = [quaternions[..., i] for i in range(4)]
    s = scale[..., 0]

    xx = qx * qx * s
    yy = qy * qy * s
    zz = qz * qz * s
    xy = qx * qy * s
    xz = qx * qz * s
    yz = qy * qz * s
    wx = qw * qx * s
    wy = qw * qy * s
    wz = qw * qz * s

    mat = np.empty(quaternions.shape[:-1] + (3, 3), dtype=quaternions.dtype)
    mat[..., 0, 0] = 1.0 - yy - zz
    mat[..., 0, 1] = xy - wz
    mat[..., 0, 2] = xz + wy
    mat[..., 1, 0] = xy + wz
    mat[..., 1, 1] = 1.0 - xx - zz
    mat[..., 1, 2] = yz - wx
    mat[..., 2, 0] = xz - wy
    mat[..., 2, 1] = yz + wx
    mat[..., 2, 2] = 1.0 - xx - yy
    return mat


def _copysign(a, b):
    return torch.sign(b) * torch.abs(a)


def _safe_sqrt(x):
    return torch.sqrt(torch.clamp(x, min=0.0))


def _normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    return q / denom


def _axis_quaternion(axis: str, angle: torch.Tensor) -> torch.Tensor:
    half = angle * 0.5
    c = torch.cos(half)
    s = torch.sin(half)
    z = torch.zeros_like(c)
    if axis == 'X':
        return torch.stack((c, s, z, z), dim=-1)
    if axis == 'Y':
        return torch.stack((c, z, s, z), dim=-1)
    if axis == 'Z':
        return torch.stack((c, z, z, s), dim=-1)
    raise ValueError(f'Invalid axis {axis}')


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    tr = m00 + m11 + m22
    quat = torch.zeros(matrix.shape[:-2] + (4,), dtype=matrix.dtype, device=matrix.device)

    pos_trace = tr > 0
    if pos_trace.any():
        s = torch.sqrt(tr[pos_trace] + 1.0) * 2.0
        quat[pos_trace, 0] = 0.25 * s
        quat[pos_trace, 1] = (m21[pos_trace] - m12[pos_trace]) / s
        quat[pos_trace, 2] = (m02[pos_trace] - m20[pos_trace]) / s
        quat[pos_trace, 3] = (m10[pos_trace] - m01[pos_trace]) / s

    branch_x = (~pos_trace) & (m00 > m11) & (m00 > m22)
    if branch_x.any():
        s = torch.sqrt(1.0 + m00[branch_x] - m11[branch_x] - m22[branch_x]) * 2.0
        quat[branch_x, 0] = (m21[branch_x] - m12[branch_x]) / s
        quat[branch_x, 1] = 0.25 * s
        quat[branch_x, 2] = (m01[branch_x] + m10[branch_x]) / s
        quat[branch_x, 3] = (m02[branch_x] + m20[branch_x]) / s

    branch_y = (~pos_trace) & (~branch_x) & (m11 > m22)
    if branch_y.any():
        s = torch.sqrt(1.0 + m11[branch_y] - m00[branch_y] - m22[branch_y]) * 2.0
        quat[branch_y, 0] = (m02[branch_y] - m20[branch_y]) / s
        quat[branch_y, 1] = (m01[branch_y] + m10[branch_y]) / s
        quat[branch_y, 2] = 0.25 * s
        quat[branch_y, 3] = (m12[branch_y] + m21[branch_y]) / s

    branch_z = (~pos_trace) & (~branch_x) & (~branch_y)
    if branch_z.any():
        s = torch.sqrt(1.0 + m22[branch_z] - m00[branch_z] - m11[branch_z]) * 2.0
        quat[branch_z, 0] = (m10[branch_z] - m01[branch_z]) / s
        quat[branch_z, 1] = (m02[branch_z] + m20[branch_z]) / s
        quat[branch_z, 2] = (m12[branch_z] + m21[branch_z]) / s
        quat[branch_z, 3] = 0.25 * s

    return standardize_quaternion(_normalize_quaternion(quat))


def _axis_angle_rotation(axis: str, angle):
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    mat = torch.zeros(angle.shape + (3, 3), dtype=angle.dtype, device=angle.device)
    mat[..., 0, 0] = 1.0
    mat[..., 1, 1] = 1.0
    mat[..., 2, 2] = 1.0
    if axis == 'X':
        mat[..., 1, 1] = cos_angle
        mat[..., 1, 2] = -sin_angle
        mat[..., 2, 1] = sin_angle
        mat[..., 2, 2] = cos_angle
    elif axis == 'Y':
        mat[..., 0, 0] = cos_angle
        mat[..., 0, 2] = sin_angle
        mat[..., 2, 0] = -sin_angle
        mat[..., 2, 2] = cos_angle
    elif axis == 'Z':
        mat[..., 0, 0] = cos_angle
        mat[..., 0, 1] = -sin_angle
        mat[..., 1, 0] = sin_angle
        mat[..., 1, 1] = cos_angle
    else:
        raise ValueError(f'Invalid axis {axis}')
    return mat


def euler_angles_to_matrix(euler_angles, convention: str):
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    angles = torch.unbind(euler_angles, dim=-1)
    q = _axis_quaternion(convention[0], angles[0])
    q = quaternion_raw_multiply(q, _axis_quaternion(convention[1], angles[1]))
    q = quaternion_raw_multiply(q, _axis_quaternion(convention[2], angles[2]))
    q = standardize_quaternion(_normalize_quaternion(q))
    return quaternion_to_matrix(q)


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    i1, i2 = {'X': (2, 1), 'Y': (0, 2), 'Z': (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    is_even_cycle = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == is_even_cycle:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _validate_euler_inputs(matrix: torch.Tensor, convention: str):
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")


def _index_from_letter(letter: str):
    lut = {'X': 0, 'Y': 1, 'Z': 2}
    if letter not in lut:
        raise ValueError(f"Invalid axis letter: {letter}")
    return lut[letter]


def matrix_to_euler_angles(matrix, convention: str):
    _validate_euler_inputs(matrix, convention)
    first_axis = _index_from_letter(convention[0])
    third_axis = _index_from_letter(convention[2])
    tait_bryan = first_axis != third_axis

    if tait_bryan:
        sign = -1.0 if first_axis - third_axis in (-1, 2) else 1.0
        middle = torch.asin(matrix[..., first_axis, third_axis] * sign)
    else:
        middle = torch.acos(matrix[..., first_axis, first_axis])

    first = _angle_from_tan(convention[0], convention[1], matrix[..., third_axis, :], False, tait_bryan)
    third = _angle_from_tan(convention[2], convention[1], matrix[..., first_axis, :], True, tait_bryan)
    return torch.stack([first, middle, third], dim=-1)


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """Sample normalized quaternions with non-negative real part."""
    q = torch.randn((n, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    lengths = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-12)
    q = q / lengths
    return standardize_quaternion(q)


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """Sample random rotation matrices of shape (n, 3, 3)."""
    quaternions = random_quaternions(
        n, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """Sample one random rotation matrix."""
    return random_rotations(1, dtype, device, requires_grad)[0]


def standardize_quaternion(quaternions):
    """Flip sign so the real part is non-negative."""
    sign = torch.where(quaternions[..., 0:1] < 0, -1.0, 1.0)
    return quaternions * sign


def quaternion_raw_multiply(a, b):
    ar, av = a[..., :1], a[..., 1:]
    br, bv = b[..., :1], b[..., 1:]
    out_r = ar * br - (av * bv).sum(dim=-1, keepdim=True)
    out_v = ar * bv + br * av + torch.cross(av, bv, dim=-1)
    return torch.cat((out_r, out_v), dim=-1)


def quaternion_multiply(a, b):
    return standardize_quaternion(quaternion_raw_multiply(a, b))


def quaternion_invert(quaternion):
    """Invert a unit quaternion rotation."""

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    q = _normalize_quaternion(quaternion)
    q_vec = q[..., 1:]
    q_w = q[..., :1]
    t = 2.0 * torch.cross(q_vec, point, dim=-1)
    return point + q_w * t + torch.cross(q_vec, t, dim=-1)


def axis_angle_to_matrix(axis_angle):
    """Axis-angle to rotation matrix."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """Rotation matrix to axis-angle."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle):
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half = angles * 0.5
    sin_half = torch.sin(half)
    inv_angles = torch.where(angles > 1e-8, 1.0 / angles, torch.zeros_like(angles))
    unit_axis = axis_angle * inv_angles
    imag = unit_axis * sin_half
    imag = torch.where(angles > 1e-8, imag, 0.5 * axis_angle)
    return torch.cat((torch.cos(half), imag), dim=-1)


def axis_angle_to_quaternion_np(axis_angle):
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    ratio = _half_angle_ratio_np(angles)
    return np.concatenate([np.cos(half_angles), axis_angle * ratio], axis=-1)



def quaternion_to_axis_angle(quaternions):
    q = _normalize_quaternion(quaternions)
    vec = q[..., 1:]
    sin_half = torch.linalg.norm(vec, dim=-1, keepdim=True)
    cos_half = q[..., :1]
    full = 2.0 * torch.atan2(sin_half, cos_half)
    inv_sin = torch.where(sin_half > 1e-8, 1.0 / sin_half, torch.zeros_like(sin_half))
    axis = vec * inv_sin
    return torch.where(sin_half > 1e-8, axis * full, 2.0 * vec)


def quaternion_to_axis_angle_np(quaternions):
    vec_norm = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(vec_norm, quaternions[..., :1])
    full_angles = 2 * half_angles
    ratio = _half_angle_ratio_np(full_angles)
    return quaternions[..., 1:] / ratio


def axis_angle_to_6d(axis_angle):
    """Axis-angle to 6D representation."""
    matrices = axis_angle_to_matrix(axis_angle)
    return matrix_to_rotation_6d(matrices)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D representation to rotation matrix via Gram-Schmidt."""

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to 6D representation (first two rows)."""
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    """6D representation to axis-angle."""
    matrices = rotation_6d_to_matrix(d6)
    return matrix_to_axis_angle(matrices)


def axis_angle_to_6d_np(axis_angle):
    """Convert axis-angle to 6D rotation representation (NumPy version)."""
    matrices = axis_angle_to_matrix_np(axis_angle)
    return matrix_to_rotation_6d_np(matrices)



def axis_angle_to_matrix_np(axis_angle):
    """NumPy axis-angle to rotation matrix."""
    return quaternion_to_matrix_np(axis_angle_to_quaternion_np(axis_angle))



def matrix_to_rotation_6d_np(matrix):
    """NumPy rotation matrix to 6D representation."""
    first_two_rows = matrix[..., :2, :]
    return first_two_rows.reshape(*matrix.shape[:-2], 6)
