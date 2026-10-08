"""2D Gaussian primitives.

A 2D Gaussian is a center ``mu``, a covariance ``Sigma``, an RGB color, and an
opacity. The covariance is never optimized directly: it is built from a pair of
scales and a rotation angle, which keeps it positive semi-definite by
construction.
"""

import torch


def covariance_2d(scale, theta):
    """Build a batch of 2x2 covariances from scales and rotation angles.

    ``scale`` is ``(N, 2)`` (positive) and ``theta`` is ``(N,)`` radians.
    Returns ``(N, 2, 2)`` with ``Sigma = R S S^T R^T``.
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    # R(theta) with columns (c, s) and (-s, c).
    r00, r10 = c, s
    r01, r11 = -s, c
    # M = R @ diag(scale): scale each column.
    m00 = r00 * scale[:, 0]
    m10 = r10 * scale[:, 0]
    m01 = r01 * scale[:, 1]
    m11 = r11 * scale[:, 1]
    sigma = torch.empty(scale.shape[0], 2, 2, device=scale.device, dtype=scale.dtype)
    sigma[:, 0, 0] = m00 * m00 + m01 * m01
    sigma[:, 0, 1] = m00 * m10 + m01 * m11
    sigma[:, 1, 0] = sigma[:, 0, 1]
    sigma[:, 1, 1] = m10 * m10 + m11 * m11
    return sigma


def gaussian_weight(xy, mu, Sigma):
    """Evaluate every Gaussian at every pixel.

    ``xy`` is ``(P, 2)``, ``mu`` is ``(N, 2)`` and ``Sigma`` is ``(N, 2, 2)``.
    Returns ``(P, N)`` with
    ``w = exp(-0.5 (xy - mu)^T Sigma^-1 (xy - mu))``.
    """
    a = Sigma[:, 0, 0]
    b = Sigma[:, 0, 1]
    c = Sigma[:, 1, 0]
    d = Sigma[:, 1, 1]
    det = torch.clamp(a * d - b * c, min=1e-12)
    inv00 = d / det
    inv01 = -b / det
    inv10 = -c / det
    inv11 = a / det

    dx = xy[:, 0:1] - mu[None, :, 0]
    dy = xy[:, 1:2] - mu[None, :, 1]
    quad = (
        inv00[None, :] * dx * dx
        + (inv01 + inv10)[None, :] * dx * dy
        + inv11[None, :] * dy * dy
    )
    return torch.exp(-0.5 * quad)


class GaussianCloud2D:
    """A set of 2D Gaussians stored as raw (unconstrained) parameters.

    Scales live in log space, colors and opacities go through a sigmoid, and
    the rotation is a raw angle. Use :meth:`params` for the leaf tensors that
    the optimizer updates.
    """

    def __init__(self, mu, log_s, theta, color_raw, op_raw):
        self.mu = mu
        self.log_s = log_s
        self.theta = theta
        self.color_raw = color_raw
        self.op_raw = op_raw

    @property
    def n(self):
        return self.mu.shape[0]

    def params(self):
        return {
            "mu": self.mu,
            "log_s": self.log_s,
            "theta": self.theta,
            "color_raw": self.color_raw,
            "op_raw": self.op_raw,
        }

    def scale(self):
        return torch.exp(self.log_s)

    def color(self):
        return torch.sigmoid(self.color_raw)

    def opacity(self):
        return torch.sigmoid(self.op_raw)

    def covariance(self):
        return covariance_2d(self.scale(), self.theta)

    @classmethod
    def init(cls, n, H, W, device="cpu", seed=None):
        """Spread-out initialization across an ``H x W`` image."""
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
        mu = torch.rand(n, 2, device=device, generator=gen) * torch.tensor(
            [W, H], device=device, dtype=torch.float32
        )
        log_s = torch.log(
            0.02 * max(H, W) * torch.ones(n, 2, device=device, dtype=torch.float32)
        )
        theta = torch.zeros(n, device=device, dtype=torch.float32)
        color_raw = torch.zeros(n, 3, device=device, dtype=torch.float32)
        op_raw = torch.full((n,), -2.0, device=device, dtype=torch.float32)
        for p in (mu, log_s, theta, color_raw, op_raw):
            p.requires_grad_(True)
        return cls(mu, log_s, theta, color_raw, op_raw)
