#!/usr/bin/env python3
"""End-to-end checks: fitting, densification, and gradient flow."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.data import Camera  # noqa: E402
from gs.densify import DensifyConfig  # noqa: E402
from gs.gaussians2d import GaussianCloud2D  # noqa: E402
from gs.gaussians3d import GaussianCloud3D  # noqa: E402
from gs.raster import render_chunk_2d, render_chunk_3d  # noqa: E402
from gs.train2d import fit_2d  # noqa: E402
from gs.train3d import fit_3d  # noqa: E402


def test_2d_fit_decreases_loss():
    # Fit a smooth gradient image; the fit should improve and stay in range.
    H = W = 32
    xs = torch.linspace(0, 1, W)[None, :].repeat(H, 1)
    target = xs[..., None].repeat(1, 1, 3)
    cloud, hist = fit_2d(target, n=64, steps=120, device="cpu", log_every=20)
    assert hist[-1]["psnr"] > hist[0]["psnr"]
    assert torch.isfinite(torch.tensor(hist[-1]["psnr"]))


def test_2d_gradients_flow():
    cloud = GaussianCloud2D.init(16, 8, 8, device="cpu", seed=0)
    xy = torch.tensor([[4.0, 4.0]])
    out = render_chunk_2d(xy, 0.5, 7.5, cloud)
    out.sum().backward()
    for p in cloud.params().values():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_2d_densify_grows_and_prunes():
    H = W = 64
    target = torch.rand(H, W, 3, generator=torch.Generator().manual_seed(1))
    cfg = DensifyConfig(
        every=30, start=32, budget=256, top_fraction=0.3, size_threshold=0.02 * 64
    )
    cloud, _ = fit_2d(target, steps=130, device="cpu", densify=cfg, log_every=0)
    assert cloud.n > 32
    # Top-fraction densification grows the set; the budget still caps it.
    assert cloud.n <= 256


def test_3d_gradients_flow():
    K = torch.tensor([[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]])
    cam = Camera(torch.eye(3), torch.tensor([0.0, 0.0, 4.0]), K, torch.zeros(16, 16, 3))
    cloud = GaussianCloud3D.init(16, device="cpu", seed=0)
    xy = torch.tensor([[8.0, 8.0]])
    out = render_chunk_3d(xy, 7.5, 8.5, cloud, cam)
    out.sum().backward()
    for p in cloud.params().values():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_3d_fit_runs():
    K = torch.tensor([[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]])
    img = torch.zeros(16, 16, 3)
    img[6:10, 6:10] = 1.0
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cams = [Camera(R, torch.tensor([0.0, 0.0, 4.0]), K, img.clone())]
    cloud, hist = fit_3d(cams, n=64, steps=40, device="cpu", log_every=20)
    assert cloud.n == 64
    assert hist[-1]["psnr"] >= hist[0]["psnr"]


if __name__ == "__main__":
    test_2d_fit_decreases_loss()
    test_2d_gradients_flow()
    test_2d_densify_grows_and_prunes()
    test_3d_gradients_flow()
    test_3d_fit_runs()
    print("test_train: OK")
