#!/usr/bin/env python3
"""Checks for alpha compositing and the 2D renderer."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.gaussians2d import GaussianCloud2D  # noqa: E402
from gs.raster import composite, rasterize, render_cloud_2d  # noqa: E402


def _reference_composite(alpha, color):
    P, N = alpha.shape
    C = torch.zeros(P, 3)
    T = torch.ones(P)
    for i in range(N):
        a = alpha[:, i]
        C = C + color[i][None, :] * a[:, None] * T[:, None]
        T = T * (1.0 - a)
    return C


def test_composite_matches_loop():
    torch.manual_seed(0)
    alpha = torch.rand(37, 23)
    color = torch.rand(23, 3)
    got = composite(alpha, color)
    ref = _reference_composite(alpha, color)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_single_opaque_gaussian_paints_its_color():
    # One Gaussian, opacity 1, weight 1 at the origin -> the pixel is its color.
    xy = torch.tensor([[0.0, 0.0]])
    mu = torch.tensor([[0.0, 0.0]])
    Sigma = torch.eye(2).unsqueeze(0)
    color = torch.tensor([[0.2, 0.4, 0.6]])
    opacity = torch.tensor([1.0])
    out = rasterize(xy, mu, Sigma, color, opacity)
    assert torch.allclose(out, color, atol=1e-5)


def test_front_to_back_over_black():
    # Two Gaussians both at the same pixel: front red alpha .5, back green 1.
    # C = red*.5 + green*1*(1-.5) = (.5, .5, 0)
    xy = torch.tensor([[0.0, 0.0]])
    mu = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    Sigma = torch.eye(2).unsqueeze(0).repeat(2, 1, 1)
    color = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    opacity = torch.tensor([0.5, 1.0])
    out = rasterize(xy, mu, Sigma, color, opacity)
    assert torch.allclose(out, torch.tensor([[0.5, 0.5, 0.0]]), atol=1e-5)


def test_render_shape_and_range():
    cloud = GaussianCloud2D.init(64, H=32, W=48, seed=1)
    img = render_cloud_2d(cloud, 32, 48, band=48)
    assert img.shape == (32, 48, 3)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


if __name__ == "__main__":
    test_composite_matches_loop()
    test_single_opaque_gaussian_paints_its_color()
    test_front_to_back_over_black()
    test_render_shape_and_range()
    print("test_raster: OK")
