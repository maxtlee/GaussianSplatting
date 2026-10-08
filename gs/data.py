"""Dataset paths and loaders."""

import json
from pathlib import Path

import torch

from .io import load_image

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "assets" / "images"
SPHERES_DIR = ROOT / "assets" / "spheres"

TWO_D_TARGETS = ["coffee", "astronaut", "cat"]


def load_target(name):
    """Load one of the 2D targets by name, or a path to any image."""
    path = Path(name)
    if not path.exists():
        path = IMAGES_DIR / f"{name}.png"
    return load_image(path)


class Camera:
    """A pinhole camera with world-to-camera pose and shared intrinsics."""

    __slots__ = ("R", "t", "K", "image", "H", "W", "name")

    def __init__(self, R, t, K, image, name=""):
        self.R = R
        self.t = t
        self.K = K
        self.image = image
        self.H, self.W = image.shape[:2]
        self.name = name


def load_spheres(device=None):
    """Load the posed sphere scene.

    Returns ``(train, val, K, W, H)`` where each camera holds its RGB target.
    """
    with open(SPHERES_DIR / "cameras.json") as f:
        meta = json.load(f)
    K = torch.tensor(meta["K"], dtype=torch.float32, device=device)
    W, H = meta["width"], meta["height"]

    def build(entries):
        cams = []
        for e in entries:
            img = load_image(SPHERES_DIR / e["file"])
            if device is not None:
                img = img.to(device)
            R = torch.tensor(e["R_wc"], dtype=torch.float32, device=device)
            t = torch.tensor(e["t"], dtype=torch.float32, device=device)
            cams.append(Camera(R, t, K, img, name=e["file"]))
        return cams

    train = build(meta["frames"])
    val = build(meta["val_frames"])
    return train, val, K, W, H
