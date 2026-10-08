#!/usr/bin/env python3
"""Checks for 2D covariance construction and Gaussian evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.gaussians2d import GaussianCloud2D, covariance_2d, gaussian_weight  # noqa: E402


def test_covariance_is_symmetric_psd():
    torch.manual_seed(0)
    scale = torch.rand(64, 2) + 0.1
    theta = torch.rand(64) * 6.28
    Sigma = covariance_2d(scale, theta)
    assert torch.allclose(Sigma, Sigma.transpose(-1, -2), atol=1e-6)
    eig = torch.linalg.eigvalsh(Sigma)
    assert float(eig.min()) > -1e-6


def test_covariance_axis_aligned():
    # theta=0 -> diag(sx^2, sy^2)
    Sigma = covariance_2d(torch.tensor([[2.0, 3.0]]), torch.tensor([0.0]))
    assert torch.allclose(Sigma[0], torch.diag(torch.tensor([4.0, 9.0])), atol=1e-6)


def test_weight_peak_and_decay():
    mu = torch.tensor([[0.0, 0.0]])
    Sigma = torch.eye(2).unsqueeze(0)
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    w = gaussian_weight(xy, mu, Sigma)
    assert abs(float(w[0, 0]) - 1.0) < 1e-6
    assert abs(float(w[1, 0]) - torch.exp(torch.tensor(-0.5))) < 1e-6
    assert abs(float(w[2, 0]) - torch.exp(torch.tensor(-4.5))) < 1e-6


def test_weight_is_isotropic_under_rotation():
    # Evaluating a rotated Gaussian at offset d equals evaluating an
    # axis-aligned Gaussian at R^T d.
    mu = torch.zeros(1, 2)
    scale = torch.tensor([[2.0, 0.5]])
    theta = torch.tensor([0.7])
    Sigma_rot = covariance_2d(scale, theta)
    Sigma_axis = covariance_2d(scale, torch.zeros(1))
    d = torch.tensor([[1.3, 0.4]])
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor([[c, -s], [s, c]])
    d_local = (R.T @ d.T).T
    w1 = gaussian_weight(mu + d, mu, Sigma_rot)
    w2 = gaussian_weight(mu + d_local, mu, Sigma_axis)
    assert torch.allclose(w1, w2, atol=1e-5)


def test_cloud_init_spread():
    cloud = GaussianCloud2D.init(1000, H=256, W=256, seed=0)
    assert cloud.n == 1000
    mu = cloud.mu.detach()
    assert float(mu[:, 0].min()) >= 0.0
    assert float(mu[:, 0].max()) <= 256.0
    assert float(mu[:, 1].max()) <= 256.0
    assert torch.allclose(cloud.scale(), torch.full((1000, 2), 0.02 * 256), atol=1e-4)


if __name__ == "__main__":
    test_covariance_is_symmetric_psd()
    test_covariance_axis_aligned()
    test_weight_peak_and_decay()
    test_weight_is_isotropic_under_rotation()
    test_cloud_init_spread()
    print("test_gaussians2d: OK")
