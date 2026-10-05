"""Image input/output helpers.

Images are handled as float32 ``(H, W, 3)`` tensors with values in ``[0, 1]``.
Use these to move between that representation and files on disk.
"""

import numpy as np
import torch
from PIL import Image


def load_image(path):
    """Load an image as a float32 ``(H, W, 3)`` tensor with values in ``[0, 1]``."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def save_image(tensor, path):
    """Save an ``(H, W, 3)`` float tensor in ``[0, 1]`` to disk.

    Values are clamped to ``[0, 1]`` and rounded to 8-bit before writing.
    """
    x = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = np.rint(x * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)
