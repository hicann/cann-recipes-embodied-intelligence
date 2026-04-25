import torch


def calculate_signed_area(a, b, c):
    area = ((c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) -
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]))
    
    return area


def _barycentric_coords_noperspective(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    
    p = p.unsqueeze(-2)
    beta_tri = calculate_signed_area(a, p, c)
    gamma_tri = calculate_signed_area(a, b, p)
    area = calculate_signed_area(a, b, c) + 1e-10

    tri_inv = 1.0 / area
    beta = beta_tri * tri_inv
    gamma = gamma_tri * tri_inv
    alpha = 1.0 - beta - gamma

    return torch.stack((alpha, beta, gamma), dim=-1)


def barycentric_coords_noperspective(
    pxy: torch.Tensor,
    axy: torch.Tensor,
    bxy: torch.Tensor,
    cxy: torch.Tensor,
) -> torch.Tensor:

    v0 = bxy - axy
    v1 = cxy - axy
    v2 = pxy.unsqueeze(-2) - axy
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    denom_inv = 1 / denom
    v = (d11 * d20 - d01 * d21) * denom_inv
    w = (d00 * d21 - d01 * d20) * denom_inv
    u = 1.0 - v - w
    return torch.stack((u, v, w), dim=-1)


def barycentric_coords_perspective_correction(
    bary_nopersp: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:

    eps = 1e-6
    az = a[..., 2]
    bz = b[..., 2]
    cz = c[..., 2]
    w0_top = bary_nopersp[..., 0] * bz * cz
    w1_top = az * bary_nopersp[..., 1] * cz
    w2_top = az * bz * bary_nopersp[..., 2]
    denom = torch.max(w0_top + w1_top + w2_top, torch.tensor(eps))
    denom_inv = 1 / denom
    w0 = w0_top * denom_inv
    w1 = w1_top * denom_inv
    w2 = w2_top * denom_inv

    return torch.stack((w0, w1, w2), dim=-1)
