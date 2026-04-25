import math
from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu

from .raster import is_frontface, triangle_signed_squared_distance
from .interp import _barycentric_coords_noperspective
from .render_settings import RenderSettings

INF_Z = math.inf


def render_npu_rasterize(
    face_verts: torch.Tensor, # (face ,3 ,3),
    pos_idx: torch.Tensor,
    image_size: tuple,
):
    device = face_verts.device

    face_verts = face_verts[0, :, :-1][pos_idx]
    face_verts[:, :, 0] = (face_verts[:, :, 0] * 0.5 + 0.5) * (image_size[0] - 1) + 0.5
    face_verts[:, :, 1] = (face_verts[:, :, 1] * 0.5 + 0.5) * (image_size[1] - 1) + 0.5
    face_verts[:, :, 2] = face_verts[:, :, 2] * 0.49999 + 0.5

    settings = RenderSettings(
        image_size=image_size,
        bin_size=128,
        cull_backfaces=True,
        max_blend_depth=1
    )

    with torch.device(face_verts.device):
        bin_size = settings.bin_size
        max_blend_depth = settings.max_blend_depth
        default_bin_frags = {
            'pix_to_face': torch.full((bin_size, bin_size, max_blend_depth), 0, dtype=torch.long),
            'bary_coords': torch.full((bin_size, bin_size, max_blend_depth, 3), 0, dtype=torch.float),
            'zbuftorch': torch.full((bin_size, bin_size, max_blend_depth), INF_Z, dtype=torch.float),
            'diststorch': torch.full((bin_size, bin_size, max_blend_depth), -1, dtype=torch.float),
        }

        return _internal_rasterize(settings, face_verts, default_bin_frags)


def _internal_rasterize(
    settings: RenderSettings,
    face_verts: torch.Tensor,
    default_bin_frags: dict,
):
    bin_size = settings.bin_size
    bin_size_inv = 1 / bin_size

    bin_w, bin_h = tuple(map(lambda x: int(math.ceil(x * bin_size_inv)), settings.image_size))

    verts_without_z = face_verts[..., :2]
    min_wh = verts_without_z.min(dim=-2).values
    max_wh = verts_without_z.max(dim=-2).values
    bin_min_wh = (min_wh * bin_size_inv).floor().long()
    bin_max_wh = (max_wh * bin_size_inv).ceil().long()

    w_grid = torch.arange(bin_w).view(bin_w, 1, 1)
    h_grid = torch.arange(bin_h).view(1, bin_h, 1)

    blur_radius_scr = settings.blur_radius_scr()
    w_min = bin_min_wh[:, 0] - blur_radius_scr
    w_max = bin_max_wh[:, 0] + blur_radius_scr
    h_min = bin_min_wh[:, 1] - blur_radius_scr
    h_max = bin_max_wh[:, 1] + blur_radius_scr

    binning_mask = (w_grid >= w_min) & (w_grid < w_max) & (h_grid >= h_min) & (h_grid < h_max)
    if settings.cull_backfaces:
        is_front = is_frontface(*verts_without_z.unbind(dim=-2),
                                            front_face=settings.front_face_direction())
        binning_mask = binning_mask & is_front
    
    pix_to_face = []
    bary_coords = []

    for bin_x in range(bin_w):
        bin_pix_to_face = []
        bin_bary_coords = []

        for bin_y in range(bin_h):
            binning_index = binning_mask[bin_x, bin_y].nonzero().squeeze(dim=-1)
            if binning_index.numel() == 0:
                bin_pix_to_face.append(default_bin_frags['pix_to_face'])
                bin_bary_coords.append(default_bin_frags['bary_coords'])
                continue
            
            samp_x_grid = torch.arange(bin_size) + (bin_x * bin_size + 0.5)
            samp_y_grid = torch.arange(bin_size) + (bin_y * bin_size + 0.5)
            samples = torch.cartesian_prod(samp_x_grid, samp_y_grid).view(bin_size, bin_size, 2)

            bin_frags_pix_to_face, bin_frags_bary_coords = _fine_rasterize_3(
                settings,
                samples,
                face_verts[binning_index],
                binning_index
            )
            bin_pix_to_face.append(bin_frags_pix_to_face)
            bin_bary_coords.append(bin_frags_bary_coords)
        
        pix_to_face.append(torch.cat(bin_pix_to_face, dim=1))
        bary_coords.append(torch.cat(bin_bary_coords, dim=1))

    pix_to_face = torch.cat(pix_to_face, dim=0)
    pix_to_face = pix_to_face[:settings.image_size[0], :settings.image_size[1]]

    bary_coords = torch.cat(bary_coords, dim=0)
    bary_coords = bary_coords[:settings.image_size[0], :settings.image_size[1]]

    pix_to_face = torch.cat((bary_coords.squeeze(2), pix_to_face), dim=-1).unsqueeze(0)

    return pix_to_face.transpose(1, 2), bary_coords


def _fine_rasterize_3(
    settings: RenderSettings,
    pix_samples: torch.Tensor,
    face_verts: torch.Tensor,
    face_idx: torch.Tensor,
):

    max_blend_depth = settings.max_blend_depth
    blur_radius = settings.blur_radius_ndc

    flattened_samples = pix_samples.view(-1, 2)

    tri_vert_tuple = face_verts[..., :2].unbind(dim=-2)
    face_sd2 = triangle_signed_squared_distance(settings, flattened_samples, *tri_vert_tuple)

    barycentrics = _barycentric_coords_noperspective(flattened_samples, *tri_vert_tuple)
    
    z_frags = (barycentrics * face_verts[..., 2]).sum(-1)
    should_write = (face_sd2 < blur_radius) & (z_frags > 0)
    
    pix_to_face = face_idx.expand(flattened_samples.shape[0], -1)
    zbuf = torch.where(should_write, z_frags, INF_Z)
    bary_coords = barycentrics
    dists = face_sd2

    if max_blend_depth == 1:
        zbuf, min_idx = zbuf.min(dim=-1, keepdim=True)
        zbuf_zero = should_write.any(dim=1).int().unsqueeze(dim=-1)
        pix_to_face = (torch.gather(pix_to_face, dim=-1, index=min_idx) + 1) * zbuf_zero
        bary_coords = torch.gather(bary_coords,
                                   dim=-2,
                                   index=min_idx.unsqueeze(dim=-1).expand(*min_idx.shape, 3)) 
                        
        bary_coords = bary_coords * zbuf_zero.unsqueeze(dim=-1).expand(*min_idx.shape, 3)
    else:
        zbuf, sorted_idx = zbuf.sort(dim=-1)

        pix_to_face = pix_to_face.gather(dim=-1, index=sorted_idx)
        bary_coords = (bary_coords.gather(dim=-2, index=sorted_idx.unsqueeze(dim=-1)
                        .expand(*sorted_idx.shape, 3)))
        dists = dists.gather(dim=-1, index=sorted_idx)

        if face_verts.shape[0] >= max_blend_depth:
            pix_to_face = pix_to_face[:, :max_blend_depth]
            bary_coords = bary_coords[:, :max_blend_depth]
        else:
            pad_len = max_blend_depth - face_verts.shape[0]
            pix_to_face = F.pad(pix_to_face, (0, pad_len), value=-1)
            bary_coords = F.pad(bary_coords, (0, 0, 0, pad_len), value=0)

    
    if settings.clip_barycentric_coords:
        bary_coords = bary_coords.clamp(0, 1)
    
    pix_to_face = pix_to_face.view(*pix_samples.shape[:-1], -1)
    bary_coords = bary_coords.view(*pix_samples.shape[:-1], -1, 3)
    
    return pix_to_face, bary_coords