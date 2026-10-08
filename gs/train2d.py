"""Optimize a set of 2D Gaussians to reproduce a target image."""

import math

import torch

from .densify import densify_2d
from .device import get_device
from .gaussians2d import GaussianCloud2D
from .optim import AdamStore
from .raster import pixel_grid, render_chunk_2d, render_cloud_2d


def auto_band(H, W, peak_n, target_elems=16_000_000):
    """Rows per band so a ``(band, N)`` weight tensor stays modest."""
    rows = max(1, target_elems // (W * max(peak_n, 1)))
    return min(H, rows) * W


def psnr_from_mse(mse):
    mse = float(mse)
    return float("inf") if mse <= 0.0 else float(-10.0 * math.log10(mse))


@torch.no_grad()
def render_psnr(cloud, target, band=None):
    """Render the full image at no grad and return ``(image, psnr)``."""
    H, W = target.shape[:2]
    img = render_cloud_2d(cloud, H, W, band=band)
    mse = ((img - target.to(img.device)) ** 2).mean()
    return img, psnr_from_mse(mse)


def fit_2d(
    target,
    n=None,
    steps=2000,
    lr=1e-2,
    seed=0,
    device=None,
    chunk=None,
    densify=None,
    log_every=50,
    verbose=False,
):
    """Fit ``target`` (H, W, 3) in [0, 1] with 2D Gaussians.

    With ``densify=None`` a fixed set of ``n`` Gaussians is optimized. With a
    :class:`~gs.densify.DensifyConfig` the fit starts at ``densify.start`` and
    grows toward ``densify.budget``.

    Returns ``(cloud, history)`` where history is a list of per-log dicts.
    """
    device = get_device() if device is None else device
    target = target.to(device)
    H, W = target.shape[:2]
    P = H * W

    start_n = densify.start if densify is not None else n
    if start_n is None:
        raise ValueError("provide n or a DensifyConfig with start")

    cloud = GaussianCloud2D.init(start_n, H, W, device=device, seed=seed)
    peak = max(cloud.n, densify.budget if densify is not None else cloud.n)
    if chunk is None:
        band = auto_band(H, W, peak)
    else:
        band = max(W, (chunk // W) * W)
    opt = AdamStore(cloud.params(), lr=lr)
    coords = pixel_grid(H, W, device=device)
    tgt = target.reshape(-1, 3)

    history = []
    grad_sum = torch.zeros(cloud.n, device=device)
    since = 0
    for step in range(steps):
        opt.zero_grad(cloud.params())
        loss = 0.0
        for lo in range(0, P, band):
            hi = min(lo + band, P)
            r_lo, r_hi = lo // W, (hi - 1) // W
            C = render_chunk_2d(coords[lo:hi], r_lo + 0.5, r_hi + 0.5, cloud)
            mse = ((C - tgt[lo:hi]) ** 2).mean()
            (mse * (hi - lo) / P).backward()
            loss += float(mse.detach()) * (hi - lo) / P

        grad_sum += cloud.mu.grad.detach().norm(dim=1)
        since += 1
        opt.step(cloud.params())

        if log_every and (step % log_every == 0 or step == steps - 1):
            psnr = psnr_from_mse(loss)
            history.append(
                {"step": step, "psnr": psnr, "loss": loss, "n": cloud.n}
            )
            if verbose:
                print(
                    f"step {step:5d}  n={cloud.n:5d}  psnr={psnr:6.2f}  loss={loss:.3e}"
                )

        if (
            densify is not None
            and since > 0
            and (step + 1) % densify.every == 0
            and (step + 1) < densify.until
        ):
            score = grad_sum / since
            cloud, opt = densify_2d(cloud, opt, score, densify.budget, densify)
            grad_sum = torch.zeros(cloud.n, device=device)
            since = 0
            opt.zero_grad(cloud.params())

    return cloud, history
