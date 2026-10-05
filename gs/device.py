"""Device selection."""

import torch


def get_device():
    """Return the fastest available device string: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
