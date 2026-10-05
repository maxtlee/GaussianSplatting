"""Reconstruction metrics.

PSNR is reported in decibels (dB), higher is better. For images normalized to
``[0, 1]`` it reduces to ``-10 * log10(MSE)``::

    PSNR = 10 * log10(MAX**2 / MSE) = 20 * log10(MAX) - 10 * log10(MSE)
"""

import math

import torch


def mse(pred, target):
    """Mean squared error over every element."""
    return torch.mean((pred.float() - target.float()) ** 2)


def psnr(pred, target, max_value=1.0):
    """Peak signal-to-noise ratio in dB; ``inf`` for a perfect match."""
    m = mse(pred, target)
    if float(m) == 0.0:
        return float("inf")
    return float(10.0 * math.log10(max_value ** 2 / float(m)))
