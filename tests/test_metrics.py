#!/usr/bin/env python3
"""Checks for MSE and PSNR."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.metrics import mse, psnr  # noqa: E402


def test_mse_basic():
    a = torch.zeros(2, 2, 3)
    b = torch.ones(2, 2, 3)
    assert abs(float(mse(a, b)) - 1.0) < 1e-6


def test_psnr_perfect_is_inf():
    a = torch.rand(4, 4, 3)
    assert psnr(a, a) == float("inf")


def test_psnr_is_10log10_reciprocal_mse():
    # Half the pixels off by one -> MSE 0.5 -> PSNR ~= 3.0103 dB.
    a = torch.zeros(1, 1, 2)
    b = torch.tensor([[[0.0, 1.0]]])
    assert abs(float(mse(a, b)) - 0.5) < 1e-6
    assert abs(psnr(a, b) - 3.010299956639812) < 1e-6


if __name__ == "__main__":
    test_mse_basic()
    test_psnr_perfect_is_inf()
    test_psnr_is_10log10_reciprocal_mse()
    print("test_metrics: OK")
