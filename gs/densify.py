"""Adaptive density control.

Periodically grow the Gaussian set where the fit is under-reconstructed and
shrink it where primitives have gone transparent. A Gaussian is scored by the
mean magnitude of the loss gradient with respect to its position, accumulated
over the steps since the last pass. Under-fit small Gaussians are cloned in
place; under-fit large ones are split into two smaller children offset along
their major axis. Nearly transparent Gaussians are pruned.

The same rule drives the 2D and 3D fits; only the representation differs
(position/scale rank, rotation type, size units, and the split offset).
"""

from dataclasses import dataclass

import torch


@dataclass
class DensifyConfig:
    every: int = 200
    grad_threshold: float = 2e-4
    size_threshold: float = 0.02
    split_scale: float = 1.6
    prune_opacity: float = 0.005
    budget: int = 4096
    start: int = 256
    until: int = 100000
    top_fraction: float = 0.0  # if > 0, densify the top fraction by score


def _select_dense(grad_mag, cfg):
    """Boolean mask of Gaussians whose score marks them under-fit."""
    if cfg.top_fraction and cfg.top_fraction > 0:
        n = grad_mag.numel()
        k = max(1, min(n, int(round(n * cfg.top_fraction))))
        threshold = torch.topk(grad_mag, k).values.min()
        return grad_mag >= threshold
    return grad_mag > cfg.grad_threshold


def _major_axis_offsets_2d(scale, theta):
    """Unit offsets along each Gaussian's major axis, scaled by half its length."""
    axis = torch.argmax(scale, dim=1)
    s_major = scale.gather(1, axis[:, None]).squeeze(1)
    c, s = torch.cos(theta), torch.sin(theta)
    ex = torch.where(axis == 0, c, -s)
    ey = torch.where(axis == 0, s, c)
    off = 0.5 * s_major
    return torch.stack([ex * off, ey * off], dim=-1)


def _build_index_sets(prune, clone, split):
    keep = (~prune).nonzero(as_tuple=True)[0]
    clone_idx = clone.nonzero(as_tuple=True)[0]
    split_idx = split.nonzero(as_tuple=True)[0]
    n_keep, n_clone, n_split = keep.numel(), clone_idx.numel(), split_idx.numel()
    src = torch.cat([keep, clone_idx, split_idx, split_idx])
    is_new = torch.zeros(src.numel(), dtype=torch.bool, device=src.device)
    is_new[n_keep:] = True
    return src, is_new, keep, clone_idx, split_idx, n_keep, n_clone, n_split


def _cap(src, is_new, mu_off, log_off, grad_mag, budget, n_keep):
    if src.numel() <= budget:
        return src, is_new, mu_off, log_off
    priority = grad_mag[src].clone()
    priority[:n_keep] += 1e9  # survivors are strongly preferred
    sel = torch.topk(priority, budget).indices
    return src[sel], is_new[sel], mu_off[sel], log_off[sel]


def densify_2d(cloud, opt, grad_mag, budget, cfg):
    """One densification pass for a :class:`GaussianCloud2D`."""
    from .gaussians2d import GaussianCloud2D

    device = cloud.mu.device
    with torch.no_grad():
        scale = cloud.scale()
        max_scale = scale.max(dim=1).values
        opacity = cloud.opacity()

        prune = opacity < cfg.prune_opacity
        dense = _select_dense(grad_mag, cfg) & (~prune)
        clone = dense & (max_scale <= cfg.size_threshold)
        split = dense & (max_scale > cfg.size_threshold)

        src, is_new, keep, clone_idx, split_idx, n_keep, n_clone, n_split = (
            _build_index_sets(prune, clone, split)
        )

        mu_off = torch.zeros(src.numel(), 2, device=device)
        log_off = torch.zeros(src.numel(), 2, device=device)
        if n_split:
            off = _major_axis_offsets_2d(scale[split_idx], cloud.theta[split_idx])
            base = n_keep + n_clone
            mu_off[base + 0 :: 2] = off
            mu_off[base + 1 :: 2] = -off
            log_off[base::] = -float(torch.log(torch.tensor(cfg.split_scale)).item())

        src, is_new, mu_off, log_off = _cap(
            src, is_new, mu_off, log_off, grad_mag, budget, n_keep
        )

        mu = (cloud.mu[src] + mu_off).detach().clone().requires_grad_(True)
        log_s = (cloud.log_s[src] + log_off).detach().clone().requires_grad_(True)
        theta = cloud.theta[src].detach().clone().requires_grad_(True)
        color_raw = cloud.color_raw[src].detach().clone().requires_grad_(True)
        op_raw = cloud.op_raw[src].detach().clone().requires_grad_(True)

    new_cloud = GaussianCloud2D(mu, log_s, theta, color_raw, op_raw)
    opt.resize(src, is_new)
    return new_cloud, opt


def _major_axis_offsets_3d(scale, quat):
    """Half-length offsets along each Gaussian's major axis in world space."""
    from .gaussians3d import quaternion_to_rotation

    R = quaternion_to_rotation(quat)
    axis = torch.argmax(scale, dim=1)
    s_major = scale.gather(1, axis[:, None]).squeeze(1)
    e = torch.zeros_like(scale)
    e.scatter_(1, axis[:, None], 1.0)
    direction = (R @ e[:, :, None]).squeeze(-1)
    return 0.5 * s_major[:, None] * direction


def densify_3d(cloud, opt, grad_mag, budget, cfg):
    """One densification pass for a :class:`GaussianCloud3D`."""
    from .gaussians3d import GaussianCloud3D

    device = cloud.mu.device
    with torch.no_grad():
        scale = cloud.scale()
        max_scale = scale.max(dim=1).values
        opacity = cloud.opacity()

        prune = opacity < cfg.prune_opacity
        dense = _select_dense(grad_mag, cfg) & (~prune)
        clone = dense & (max_scale <= cfg.size_threshold)
        split = dense & (max_scale > cfg.size_threshold)

        src, is_new, keep, clone_idx, split_idx, n_keep, n_clone, n_split = (
            _build_index_sets(prune, clone, split)
        )

        mu_off = torch.zeros(src.numel(), 3, device=device)
        log_off = torch.zeros(src.numel(), 3, device=device)
        if n_split:
            off = _major_axis_offsets_3d(scale[split_idx], cloud.quat_raw[split_idx])
            base = n_keep + n_clone
            mu_off[base + 0 :: 2] = off
            mu_off[base + 1 :: 2] = -off
            log_off[base::] = -float(torch.log(torch.tensor(cfg.split_scale)).item())

        src, is_new, mu_off, log_off = _cap(
            src, is_new, mu_off, log_off, grad_mag, budget, n_keep
        )

        mu = (cloud.mu[src] + mu_off).detach().clone().requires_grad_(True)
        log_s = (cloud.log_s[src] + log_off).detach().clone().requires_grad_(True)
        quat_raw = cloud.quat_raw[src].detach().clone().requires_grad_(True)
        color_raw = cloud.color_raw[src].detach().clone().requires_grad_(True)
        op_raw = cloud.op_raw[src].detach().clone().requires_grad_(True)

    new_cloud = GaussianCloud3D(mu, log_s, quat_raw, color_raw, op_raw)
    opt.resize(src, is_new)
    return new_cloud, opt
