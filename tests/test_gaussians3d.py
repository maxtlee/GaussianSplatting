#!/usr/bin/env python3
"""Checks for 3D covariance, quaternion rotation, and projection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.gaussians3d import (  # noqa: E402
    GaussianCloud3D,
    covariance_3d,
    project_gaussian,
    quaternion_to_rotation,
)


def test_quaternion_identity_and_orthonormal():
    R = quaternion_to_rotation(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(R[0], torch.eye(3), atol=1e-6)

    torch.manual_seed(0)
    q = torch.randn(16, 4)
    R = quaternion_to_rotation(q)
    eye = torch.eye(3).expand(16, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert abs(float(torch.linalg.det(R).mean()) - 1.0) < 1e-4


def test_covariance_is_psd_and_axis_aligned():
    scale = torch.tensor([[1.0, 2.0, 3.0]])
    Sigma = covariance_3d(scale, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(Sigma[0], torch.diag(torch.tensor([1.0, 4.0, 9.0])), atol=1e-5)
    eig = torch.linalg.eigvalsh(Sigma)
    assert float(eig.min()) > -1e-6


def test_projection_center():
    # A point on the optical axis projects to the principal point.
    K = torch.tensor([[100.0, 0.0, 80.0], [0.0, 100.0, 80.0], [0.0, 0.0, 1.0]])
    mu3 = torch.tensor([[0.0, 0.0, 0.0]])
    Sigma3 = torch.eye(3).unsqueeze(0)
    R = torch.eye(3)
    t = torch.tensor([0.0, 0.0, 4.0])
    mu2, Sigma2, depth = project_gaussian(mu3, Sigma3, R, t, K)
    assert torch.allclose(mu2[0], torch.tensor([80.0, 80.0]), atol=1e-4)
    assert abs(float(depth[0]) - 4.0) < 1e-5
    eig = torch.linalg.eigvalsh(Sigma2)
    assert float(eig.min()) > -1e-6


def test_projection_scales_with_depth():
    # A fixed offset projects closer to center as depth grows.
    K = torch.tensor([[100.0, 0.0, 80.0], [0.0, 100.0, 80.0], [0.0, 0.0, 1.0]])
    R = torch.eye(3)
    Sigma3 = torch.eye(3).unsqueeze(0)
    near = project_gaussian(torch.tensor([[1.0, 0.0, 0.0]]), Sigma3, R, torch.tensor([0.0, 0.0, 4.0]), K)[0]
    far = project_gaussian(torch.tensor([[1.0, 0.0, 0.0]]), Sigma3, R, torch.tensor([0.0, 0.0, 8.0]), K)[0]
    assert float(near[0, 0]) > float(far[0, 0]) > 80.0


def test_cloud_init_bounds():
    cloud = GaussianCloud3D.init(500, seed=0)
    assert cloud.n == 500
    assert float(cloud.mu.detach().abs().max()) <= 1.5
    assert torch.allclose(cloud.scale(), torch.full((500, 3), 0.08), atol=1e-5)


if __name__ == "__main__":
    test_quaternion_identity_and_orthonormal()
    test_covariance_is_psd_and_axis_aligned()
    test_projection_center()
    test_projection_scales_with_depth()
    test_cloud_init_bounds()
    print("test_gaussians3d: OK")
