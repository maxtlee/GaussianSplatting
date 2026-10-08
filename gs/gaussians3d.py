"""3D Gaussian primitives and projection to a camera.

A 3D Gaussian has a mean, a covariance built from a per-axis scale and a
rotation (stored as a quaternion), a color, and an opacity. To render one it is
projected through a pinhole camera: the mean goes through the projection and the
covariance is pushed through the projection's Jacobian, giving a 2D mean and a
2D covariance that the 2D rasterizer can composite.
"""

import torch


def quaternion_to_rotation(q):
    """Convert ``(N, 4)`` quaternions ``(w, x, y, z)`` to ``(N, 3, 3)`` rotations.

    The quaternions are normalized first, so the raw parameters need not be unit
    length.
    """
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def covariance_3d(scale, quat):
    """Build ``(N, 3, 3)`` covariances ``Sigma = R S S^T R^T``."""
    R = quaternion_to_rotation(quat)
    M = R * scale[:, None, :]
    return M @ M.transpose(-1, -2)


def project_gaussian(mu3, Sigma3, R_wc, t, K):
    """Project 3D Gaussians into a camera.

    Returns ``(mu2, Sigma2, depth)`` with shapes ``(N, 2)``, ``(N, 2, 2)`` and
    ``(N,)``. ``depth`` is the camera-space ``z`` used for ordering.
    """
    mu_cam = mu3 @ R_wc.T + t
    xc, yc, zc = mu_cam[:, 0], mu_cam[:, 1], mu_cam[:, 2]
    z = zc.clamp(min=1e-4)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    mu2 = torch.stack([fx * xc / z + cx, fy * yc / z + cy], dim=-1)

    J = torch.zeros(mu3.shape[0], 2, 3, device=mu3.device, dtype=mu3.dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * xc / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * yc / (z * z)

    Sigma_cam = R_wc @ Sigma3 @ R_wc.transpose(-1, -2)
    Sigma2 = J @ Sigma_cam @ J.transpose(-1, -2)
    return mu2, Sigma2, zc


class GaussianCloud3D:
    """A set of 3D Gaussians stored as raw (unconstrained) parameters."""

    def __init__(self, mu, log_s, quat_raw, color_raw, op_raw):
        self.mu = mu
        self.log_s = log_s
        self.quat_raw = quat_raw
        self.color_raw = color_raw
        self.op_raw = op_raw

    @property
    def n(self):
        return self.mu.shape[0]

    def params(self):
        return {
            "mu": self.mu,
            "log_s": self.log_s,
            "quat_raw": self.quat_raw,
            "color_raw": self.color_raw,
            "op_raw": self.op_raw,
        }

    def scale(self):
        return torch.exp(self.log_s)

    def color(self):
        return torch.sigmoid(self.color_raw)

    def opacity(self):
        return torch.sigmoid(self.op_raw)

    def quaternion(self):
        return self.quat_raw / self.quat_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def covariance(self):
        return covariance_3d(self.scale(), self.quaternion())

    @classmethod
    def init(cls, n, device="cpu", seed=None, extent=1.5, scale=0.08):
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
        mu = (torch.rand(n, 3, device=device, generator=gen) * 2 - 1) * extent
        log_s = torch.log(
            scale * torch.ones(n, 3, device=device, dtype=torch.float32)
        )
        quat_raw = torch.zeros(n, 4, device=device, dtype=torch.float32)
        quat_raw[:, 0] = 1.0
        color_raw = torch.zeros(n, 3, device=device, dtype=torch.float32)
        op_raw = torch.full((n,), -2.0, device=device, dtype=torch.float32)
        for p in (mu, log_s, quat_raw, color_raw, op_raw):
            p.requires_grad_(True)
        return cls(mu, log_s, quat_raw, color_raw, op_raw)
