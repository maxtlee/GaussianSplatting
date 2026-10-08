"""Small helpers for saving comparison images."""

import torch

from .io import save_image


def side_by_side(images, gap=6, value=1.0):
    """Concatenate ``(H, W, 3)`` tensors horizontally with a light gap."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    H = images[0].shape[0]
    parts = []
    for i, im in enumerate(images):
        if i:
            parts.append(torch.full((H, gap, 3), value, device=im.device))
        parts.append(im.clamp(0, 1))
    return torch.cat(parts, dim=1)


def error_map(target, render):
    """Amplified absolute error rendered as a grayscale image."""
    diff = (target - render).abs().mean(dim=-1, keepdim=True)
    diff = (diff * 4.0).clamp(0, 1).repeat(1, 1, 3)
    return diff


def save_comparison(target, render, path, gap=6):
    """Write ``target | render | error`` as one image."""
    err = error_map(target, render)
    save_image(side_by_side([target, render, err], gap=gap), path)


def save_strip(images, path, gap=6):
    save_image(side_by_side(images, gap=gap), path)
