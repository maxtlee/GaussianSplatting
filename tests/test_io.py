#!/usr/bin/env python3
"""Round-trip checks for image loading and saving."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.io import load_image, save_image  # noqa: E402


def test_load_shape_and_range():
    img = load_image(ROOT / "assets" / "images" / "coffee.png")
    assert img.shape == (256, 256, 3), img.shape
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_save_round_trip_is_close():
    img = load_image(ROOT / "assets" / "images" / "cat.png")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.png"
        save_image(img, path)
        back = load_image(path)
    # 8-bit quantization is the only loss.
    assert torch.allclose(img, back, atol=1.0 / 255.0 + 1e-6)


def test_save_clamps():
    img = torch.tensor([[[-1.0, 0.5, 2.0]]])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "clamp.png"
        save_image(img, path)
        back = load_image(path)
    assert float(back[0, 0, 0]) == 0.0
    assert abs(float(back[0, 0, 1]) - 0.5) <= 1.0 / 255.0
    assert float(back[0, 0, 2]) == 1.0


if __name__ == "__main__":
    test_load_shape_and_range()
    test_save_round_trip_is_close()
    test_save_clamps()
    print("test_io: OK")
