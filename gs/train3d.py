"""Optimize a set of 3D Gaussians to reconstruct a scene from posed views."""

import torch

from .densify import densify_3d
from .device import get_device
from .gaussians3d import GaussianCloud3D
from .optim import AdamStore
from .raster import pixel_grid, render_chunk_3d, render_cloud_3d
from .train2d import auto_band, psnr_from_mse


@torch.no_grad()
def evaluate(cloud, cams, band=None, return_images=False, cutoff=3.5):
    """Render every camera and return ``(psnrs, images)``."""
    psnrs, images = [], []
    for cam in cams:
        img = render_cloud_3d(cloud, cam, band=band, cutoff=cutoff)
        mse = ((img - cam.image.to(img.device)) ** 2).mean()
        psnrs.append(psnr_from_mse(mse))
        if return_images:
            images.append(img)
    return psnrs, images


def fit_3d(
    train_cams,
    n=None,
    steps=1500,
    lr=1e-2,
    seed=0,
    device=None,
    band=None,
    densify=None,
    log_every=50,
    verbose=False,
    cam_seed=0,
    eval_every=0,
    eval_sets=None,
    cutoff=3.5,
):
    """Fit 3D Gaussians to posed ``train_cams`` (each with an RGB target).

    With ``densify=None`` a fixed set of ``n`` Gaussians is optimized. With a
    :class:`~gs.densify.DensifyConfig` the fit starts at ``densify.start`` and
    grows toward ``densify.budget``. Returns ``(cloud, history)``.
    """
    device = get_device() if device is None else device
    H, W = train_cams[0].H, train_cams[0].W
    P = H * W

    start_n = densify.start if densify is not None else n
    if start_n is None:
        raise ValueError("provide n or a DensifyConfig with start")

    cloud = GaussianCloud3D.init(start_n, device=device, seed=seed)
    peak = max(cloud.n, densify.budget if densify is not None else cloud.n)
    if band is None:
        band = auto_band(H, W, peak)
    opt = AdamStore(cloud.params(), lr=lr)
    xy = pixel_grid(H, W, device=device)
    # torch.randint samples the camera index on the host regardless of device.
    gen = torch.Generator().manual_seed(cam_seed)

    history = []
    grad_sum = torch.zeros(cloud.n, device=device)
    since = 0
    for step in range(steps):
        ci = int(torch.randint(len(train_cams), (1,), generator=gen).item())
        cam = train_cams[ci]
        tgt = cam.image.to(device).reshape(-1, 3)

        opt.zero_grad(cloud.params())
        loss = 0.0
        for lo in range(0, P, band):
            hi = min(lo + band, P)
            r_lo, r_hi = lo // W, (hi - 1) // W
            C = render_chunk_3d(
                xy[lo:hi], r_lo + 0.5, r_hi + 0.5, cloud, cam, cutoff=cutoff
            )
            mse = ((C - tgt[lo:hi]) ** 2).mean()
            term = mse * (hi - lo) / P
            if term.requires_grad:  # empty bands contribute no gradient
                term.backward()
            loss += float(mse.detach()) * (hi - lo) / P

        if cloud.mu.grad is not None:  # no gradient if every band was empty
            grad_sum += cloud.mu.grad.detach().norm(dim=1)
        since += 1
        opt.step(cloud.params())

        sample_psnr = psnr_from_mse(loss)
        if eval_every and eval_sets and (
            step % eval_every == 0 or step == steps - 1
        ):
            rec = {
                "step": step,
                "n": cloud.n,
                "sample_psnr": sample_psnr,
                "loss": loss,
            }
            for name, cams in eval_sets.items():
                ps, _ = evaluate(cloud, cams)
                rec[name] = float(sum(ps) / len(ps))
            history.append(rec)
            if verbose:
                msg = "  ".join(
                    f"{k}={rec[k]:.2f}" for k in eval_sets
                )
                print(f"step {step:5d}  n={cloud.n:5d}  {msg}")
        elif log_every and (step % log_every == 0 or step == steps - 1):
            history.append(
                {
                    "step": step,
                    "psnr": sample_psnr,
                    "loss": loss,
                    "n": cloud.n,
                }
            )
            if verbose:
                print(
                    f"step {step:5d}  n={cloud.n:5d}  psnr={sample_psnr:6.2f}  "
                    f"loss={loss:.3e}"
                )

        if (
            densify is not None
            and since > 0
            and (step + 1) % densify.every == 0
            and (step + 1) < densify.until
        ):
            score = grad_sum / since
            cloud, opt = densify_3d(cloud, opt, score, densify.budget, densify)
            grad_sum = torch.zeros(cloud.n, device=device)
            since = 0
            opt.zero_grad(cloud.params())

    return cloud, history
