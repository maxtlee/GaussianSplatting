"""Differentiable alpha-compositing rasterizer.

The renderer walks the Gaussians front to back at every pixel and applies the
standard "over" operator::

    C = sum_i c_i a_i T_i,     T_i = prod_{j < i} (1 - a_j)

Rather than a Python loop over Gaussians, the exclusive prefix product ``T`` is
computed with ``cumprod`` and the color sum with a single matmul, so the whole
compositing step is a handful of vectorized ops.

Gaussians only influence pixels near their center, so rendering processes the
image in horizontal bands and, within each band, evaluates only the Gaussians
whose support reaches that band. This keeps the work proportional to the local
overlap instead of ``pixels x all Gaussians``.
"""

import torch

from .gaussians2d import covariance_2d, gaussian_weight
from .gaussians3d import covariance_3d, project_gaussian

# A Gaussian is treated as zero beyond this many standard deviations. At 3.5
# sigma the weight is exp(-0.5 * 3.5**2) ~= 2e-3.
CUTOFF = 3.5


def pixel_grid(H, W, device="cpu", dtype=torch.float32):
    """Pixel centers as ``(H*W, 2)`` in row-major order (x, y)."""
    ys = torch.arange(H, device=device, dtype=dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def composite(alpha, color):
    """Front-to-back compositing.

    ``alpha`` is ``(P, N)`` in ``[0, 1]`` ordered front to back and ``color``
    is ``(N, 3)``. Returns the accumulated ``(P, 3)`` over a black background.
    """
    one_minus = 1.0 - alpha
    trans = torch.cumprod(one_minus, dim=1)
    # Shift right by one so T_i excludes Gaussian i's own alpha.
    trans = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], dim=1)
    return (alpha * trans) @ color


def rasterize(xy, mu, Sigma, color, opacity):
    """Rasterize already-ordered Gaussians at the points ``xy``.

    ``mu``/``Sigma``/``color``/``opacity`` must already be in front-to-back
    order. Returns ``(P, 3)``.
    """
    w = gaussian_weight(xy, mu, Sigma).clamp(0.0, 1.0)
    alpha = (opacity[None, :] * w).clamp(0.0, 1.0)
    return composite(alpha, color)


def _band_indices(mu, Sigma, y_lo, y_hi, cutoff):
    """Indices of Gaussians whose support intersects ``[y_lo, y_hi]``."""
    ry = cutoff * torch.sqrt(Sigma[:, 1, 1].clamp(min=1e-12))
    mask = (mu[:, 1] + ry >= y_lo) & (mu[:, 1] - ry <= y_hi)
    return mask.nonzero(as_tuple=True)[0]


def _band_shape(H, W, peak_n, target_elems):
    rows = max(1, target_elems // (W * max(peak_n, 1)))
    return min(H, rows) * W


def render_cloud_2d(cloud, H, W, band=None, cutoff=CUTOFF):
    """Render a :class:`GaussianCloud2D` to an ``(H, W, 3)`` image (no grad)."""
    P = H * W
    xy = pixel_grid(H, W, device=cloud.mu.device)
    if band is None:
        band = _band_shape(H, W, cloud.n, 16_000_000)
    with torch.no_grad():
        Sigma = cloud.covariance()
        color = cloud.color()
        opacity = cloud.opacity()
        mu = cloud.mu
        out = torch.zeros(P, 3, device=cloud.mu.device)
        for lo in range(0, P, band):
            hi = min(lo + band, P)
            r_lo, r_hi = lo // W, (hi - 1) // W
            idx = _band_indices(mu, Sigma, r_lo + 0.5, r_hi + 0.5, cutoff)
            if idx.numel():
                out[lo:hi] = rasterize(
                    xy[lo:hi], mu[idx], Sigma[idx], color[idx], opacity[idx]
                )
    return out.reshape(H, W, 3)


def render_chunk_2d(xy, y_lo, y_hi, cloud, cutoff=CUTOFF):
    """One band of the 2D render, building the graph from raw parameters.

    The band membership test is computed on detached values (a hard local
    support), while the rasterization itself stays differentiable through the
    selected Gaussians' parameters. Recomputing color/opacity/covariance per
    band keeps each band's graph independent so training can call ``backward``
    per band.
    """
    scale = torch.exp(cloud.log_s)
    Sigma = covariance_2d(scale, cloud.theta)
    color = torch.sigmoid(cloud.color_raw)
    opacity = torch.sigmoid(cloud.op_raw)
    with torch.no_grad():
        idx = _band_indices(cloud.mu.detach(), Sigma.detach(), y_lo, y_hi, cutoff)
    if idx.numel() == 0:
        return xy.new_zeros(xy.shape[0], 3)
    return rasterize(xy, cloud.mu[idx], Sigma[idx], color[idx], opacity[idx])


def _project_ordered_3d(cloud, cam):
    """Project every 3D Gaussian and return them in front-to-back order."""
    Sig3 = covariance_3d(cloud.scale(), cloud.quaternion())
    mu2, Sig2, depth = project_gaussian(cloud.mu, Sig3, cam.R, cam.t, cam.K)
    # Drop primitives behind the camera by giving them zero opacity.
    opacity = cloud.opacity() * (depth > 1e-4).to(cloud.mu.dtype)
    order = torch.argsort(depth)
    return mu2[order], Sig2[order], cloud.color()[order], opacity[order]


def render_chunk_3d(xy, y_lo, y_hi, cloud, cam, cutoff=CUTOFF):
    """One band of a 3D render for one camera, differentiable in the cloud."""
    mu2, Sig2, color, opacity = _project_ordered_3d(cloud, cam)
    with torch.no_grad():
        idx = _band_indices(mu2.detach(), Sig2.detach(), y_lo, y_hi, cutoff)
    if idx.numel() == 0:
        return xy.new_zeros(xy.shape[0], 3)
    return rasterize(xy, mu2[idx], Sig2[idx], color[idx], opacity[idx])


def render_cloud_3d(cloud, cam, band=None, cutoff=CUTOFF):
    """Render a :class:`GaussianCloud3D` from ``cam`` (no grad)."""
    H, W = cam.H, cam.W
    P = H * W
    xy = pixel_grid(H, W, device=cloud.mu.device)
    if band is None:
        band = _band_shape(H, W, cloud.n, 16_000_000)
    with torch.no_grad():
        mu2, Sig2, color, opacity = _project_ordered_3d(cloud, cam)
        out = torch.zeros(P, 3, device=cloud.mu.device)
        for lo in range(0, P, band):
            hi = min(lo + band, P)
            r_lo, r_hi = lo // W, (hi - 1) // W
            idx = _band_indices(mu2, Sig2, r_lo + 0.5, r_hi + 0.5, cutoff)
            if idx.numel():
                out[lo:hi] = rasterize(
                    xy[lo:hi], mu2[idx], Sig2[idx], color[idx], opacity[idx]
                )
    return out.reshape(H, W, 3)
