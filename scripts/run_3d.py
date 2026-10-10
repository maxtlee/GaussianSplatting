#!/usr/bin/env python3
"""Train 3D Gaussians on the posed sphere scene and evaluate the fit.

A run writes to ``experiments/3d/<tag>/``: ``metrics.json`` (mean training and
held-out PSNR plus history), side-by-side target/render images for a few training
and held-out views, and a short orbit as a PNG sequence and mp4. Runs are
resumable: an existing ``metrics.json`` is reused unless ``--force``.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.data import Camera, load_spheres  # noqa: E402
from gs.densify import DensifyConfig  # noqa: E402
from gs.train3d import evaluate, fit_3d  # noqa: E402
from gs.viz import save_comparison  # noqa: E402


def look_at(eye, target, K, image, up=(0.0, 1.0, 0.0)):
    """OpenCV-convention camera at ``eye`` looking at ``target``."""
    dev = K.device
    eye = torch.as_tensor(eye, dtype=torch.float32, device=dev)
    target = torch.as_tensor(target, dtype=torch.float32, device=dev)
    up = torch.as_tensor(up, dtype=torch.float32, device=dev)
    z = target - eye
    z = z / z.norm()
    x = torch.linalg.cross(z, up)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z], dim=0)
    t = -(R @ eye)
    return Camera(R, t, K, image)


def orbit_cameras(K, image, n=24, radius=4.0, elevation_deg=20.0):
    cams = []
    el = math.radians(elevation_deg)
    for i in range(n):
        az = 2 * math.pi * i / n
        eye = [
            radius * math.cos(el) * math.cos(az),
            radius * math.sin(el),
            radius * math.cos(el) * math.sin(az),
        ]
        cams.append(look_at(eye, [0.0, 0.0, 0.0], K, image))
    return cams


def make_orbit_mp4(frames_dir, out_mp4, fps=12):
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                str(out_mp4),
            ],
            check=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"ffmpeg failed: {exc}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fixed")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--densify", action="store_true")
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument(
        "--size-threshold", type=float, default=0.01, help="world units"
    )
    ap.add_argument("--out", default=str(ROOT / "experiments" / "3d"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="evaluate train/val PSNR every N steps (0 = off)",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"[skip] {metrics_path}")
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    train, val, K, W, H = load_spheres(device=device)
    K = K.to(device)

    densify = None
    if args.densify:
        densify = DensifyConfig(
            every=200,
            start=args.start,
            budget=args.budget,
            top_fraction=0.10,
            size_threshold=args.size_threshold,
            prune_opacity=0.005,
            until=int(args.steps * 0.8),
        )

    t0 = time.time()
    eval_sets = {"train": train, "val": val} if args.eval_every else None
    cloud, history = fit_3d(
        train,
        n=args.n,
        steps=args.steps,
        seed=args.seed,
        device=device,
        densify=densify,
        log_every=50,
        eval_every=args.eval_every,
        eval_sets=eval_sets,
    )
    train_psnr, _ = evaluate(cloud, train)
    val_psnr, val_imgs = evaluate(cloud, val, return_images=True)

    # Side-by-side target/render for a few training and held-out views.
    for idx in (0, 12):
        _, imgs = evaluate(cloud, [train[idx]], return_images=True)
        save_comparison(
            train[idx].image.cpu(), imgs[0].cpu(), out / f"train_{idx:03d}.png"
        )
    for idx in (0, 2):
        save_comparison(
            val[idx].image.cpu(), val_imgs[idx].cpu(), out / f"val_{idx:03d}.png"
        )

    # Novel orbit of views the optimizer never saw.
    from gs.raster import render_cloud_3d

    frames_dir = out / "orbit"
    frames_dir.mkdir(exist_ok=True)
    blank = torch.zeros(H, W, 3, device=device)
    for i, cam in enumerate(orbit_cameras(K, blank, n=24)):
        frame = render_cloud_3d(cloud, cam)
        _save_png(frame, frames_dir / f"frame_{i:03d}.png")
    make_orbit_mp4(frames_dir, out / "orbit.mp4")

    meta = {
        "tag": args.tag,
        "n_target": args.n,
        "steps": args.steps,
        "seed": args.seed,
        "densify": None if densify is None else densify.__dict__,
        "final_n": int(cloud.n),
        "seconds": time.time() - t0,
        "train_psnr_mean": float(sum(train_psnr) / len(train_psnr)),
        "val_psnr_mean": float(sum(val_psnr) / len(val_psnr)),
        "train_psnr": train_psnr,
        "val_psnr": val_psnr,
        "history": history,
    }
    metrics_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[done] {args.tag}  n={cloud.n}  train={meta['train_psnr_mean']:.2f}  "
        f"val={meta['val_psnr_mean']:.2f}  ({meta['seconds']:.0f}s)"
    )


def _save_png(tensor, path):
    import numpy as np
    from PIL import Image

    x = tensor.detach().clamp(0, 1).cpu().numpy()
    Image.fromarray(np.rint(x * 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    main()
