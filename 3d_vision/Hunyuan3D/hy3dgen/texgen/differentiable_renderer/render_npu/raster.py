from typing import Tuple
import torch

from .render_settings import RenderSettings


def _cross2d(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _edge_function(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    p = p.unsqueeze(dim=-2)
    return _cross2d(p - a, b - a).sign()


def _edge_function_2(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    p = p.unsqueeze(dim=-2)
    a_edge = b[..., 1] - a[..., 1]
    b_edge = a[..., 0] - b[..., 0]
    c_edge = _cross2d(b, a)
    return ((torch.stack((a_edge, b_edge), dim=-1) * p).sum(dim=-1) + c_edge).sign()


def _point_segment_distance_square(
    settings: RenderSettings,
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:

    p_unsq = p.unsqueeze(dim=-2)

    ab = b - a
    t = ((p_unsq - a) * ab).sum(dim=-1) / (ab ** 2).sum(dim=-1)
    t_clamped = torch.clip(t, 0, 1)
    projection = a + t_clamped.unsqueeze(dim=-1) * ab
    d = (projection - p_unsq) * torch.tensor(settings.screen_to_ndc_scale())
    return (d ** 2).sum(dim=-1)


def _triangle_inside_direction(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    return (_cross2d(c - a, b - a) > 0).int() * 2 - 1


def _internal_is_inside_triangle(
    edge_ab: torch.Tensor,
    edge_bc: torch.Tensor,
    edge_ca: torch.Tensor,
    inside_dir,
) -> torch.Tensor:
    return (edge_ab == inside_dir) & (edge_ab == edge_bc) & (edge_ab == edge_ca)


def is_frontface(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    front_face: int = 1,
) -> torch.Tensor:
    return _triangle_inside_direction(a, b, c) == front_face


def is_inside_triangle(
    settings: RenderSettings,
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    edge_ab = _edge_function_2(p, a, b)
    edge_bc = _edge_function_2(p, b, c)
    edge_ca = _edge_function_2(p, c, a)

    if settings.cull_backfaces:
        inside_dir = settings.front_face_direction()
    else:
        inside_dir = _triangle_inside_direction(a, b, c)
    
    return _internal_is_inside_triangle(edge_ab, edge_bc, edge_ca, inside_dir)


def triangle_signed_squared_distance(
    settings: RenderSettings,
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:

    is_inside = is_inside_triangle(settings, p, a, b, c)

    dist_ab = _point_segment_distance_square(settings, p, a, b)
    dist_bc = _point_segment_distance_square(settings, p, b, c)
    dist_ca = _point_segment_distance_square(settings, p, c, a)

    dists = torch.stack([dist_ab, dist_bc, dist_ca]).min(dim=0).values
    sign = (is_inside.int() * -2) + 1

    return dists * sign