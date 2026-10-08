#!/usr/bin/env python3
"""Checks for the resizable Adam implementation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.optim import AdamStore  # noqa: E402


def test_matches_torch_adam():
    torch.manual_seed(0)
    target = torch.tensor([[3.0, -1.0]])
    x0 = torch.zeros(5, 2)
    a = x0.clone().requires_grad_(True)
    b = x0.clone().requires_grad_(True)
    mine = AdamStore({"x": a}, lr=0.05)
    theirs = torch.optim.Adam([b], lr=0.05)
    for _ in range(60):
        mine.zero_grad({"x": a})
        (a - target).pow(2).sum().backward()
        mine.step({"x": a})
        theirs.zero_grad()
        (b - target).pow(2).sum().backward()
        theirs.step()
    assert torch.allclose(a, b, atol=1e-5)


def test_resize_copies_survivors_and_zeroes_new():
    torch.manual_seed(0)
    p = torch.zeros(4, 2, requires_grad=True)
    opt = AdamStore({"x": p}, lr=0.1)
    for _ in range(3):
        p.grad = torch.ones_like(p)
        opt.step({"x": p})
    old_m = opt.m["x"].clone()
    # New set: survivor 2, then two new entries (index 0 as placeholder).
    old_index = torch.tensor([2, 0, 0])
    is_new = torch.tensor([False, True, True])
    opt.resize(old_index, is_new)
    assert torch.allclose(opt.m["x"][0], old_m[2])
    assert torch.allclose(opt.m["x"][1], torch.zeros(2))
    assert torch.allclose(opt.m["x"][2], torch.zeros(2))


if __name__ == "__main__":
    test_matches_torch_adam()
    test_resize_copies_survivors_and_zeroes_new()
    print("test_optim: OK")
