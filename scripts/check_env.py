#!/usr/bin/env python3
"""Check that the environment can actually run the pipeline.

Beyond importing torch, this runs a real matrix multiply on the selected device
(which catches GPU builds that report ``cuda.is_available()`` but ship no kernels
for the card), loads a sample image, and computes a PSNR.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs import get_device  # noqa: E402
from gs.io import load_image  # noqa: E402
from gs.metrics import psnr  # noqa: E402


def main():
    device = get_device()
    print(f"python     {sys.version.split()[0]}")
    print(f"torch      {torch.__version__}")
    print(f"numpy      {__import__('numpy').__version__}")
    print(f"pillow     {__import__('PIL').__version__}")
    print(f"device     {device}")

    if device == "cuda":
        print(f"gpu        {torch.cuda.get_device_name(0)}")
        print(f"arch list  {torch.cuda.get_arch_list()}")

    # A real op on the device: a build without kernels for this card fails here.
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    c = (a @ b).sum().item()
    print(f"matmul ok  sum={c:.3f}")

    img = load_image(ROOT / "assets" / "images" / "coffee.png")
    print(f"image      {tuple(img.shape)} range=[{img.min():.3f}, {img.max():.3f}]")
    print(f"psnr self  {psnr(img, img):.3f} dB")

    print("environment OK")


if __name__ == "__main__":
    main()
