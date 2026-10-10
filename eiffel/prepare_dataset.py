#!/usr/bin/env python3
"""Turn the SfM reconstruction into a training set.

Steps:

1. keep registered cameras with enough 3D observations (rejects poorly
   triangulated views),
2. RANSAC-select the dominant ring of camera centers in the horizontal plane
   (rejects aerial / off-axis runs of photos),
3. split into train / held-out views,
4. optionally color-normalize every image to a shared target (robust per-channel
   mean and standard deviation) so exposure and white balance roughly agree
   before training,
5. downscale to a fixed long side and write a compact dataset.

    python eiffel/prepare_dataset.py --execute
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SFM = DATA / "sfm" / "cameras.json"
RAW = DATA / "raw"
PROC = DATA / "processed"


def camera_center(R, t):
    return -np.asarray(R).T @ np.asarray(t)


def circle_from_3pts(a, b, c):
    """Circumcenter, unit normal, and radius of the circle through three 3D points."""
    u, v = b - a, c - a
    n = np.cross(u, v)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    e1 = u / np.linalg.norm(u)
    v_perp = v - (v @ e1) * e1
    if np.linalg.norm(v_perp) < 1e-9:
        return None
    e2 = v_perp / np.linalg.norm(v_perp)
    u1 = np.linalg.norm(u)
    cx, cy = v @ e1, v @ e2
    denom = 2 * (u1 * cy)
    if abs(denom) < 1e-9:
        return None
    o1 = u1 / 2
    o2 = (cx * cx + cy * cy - cx * u1) / (2 * cy)
    center = a + o1 * e1 + o2 * e2
    radius = math.hypot(o1, o2)
    return center, n / nn, radius


def ransac_circle_3d(pts, thresh, iters=4000, seed=0):
    """RANSAC a 3D circle through camera centers; returns (inlier mask, model)."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    best_mask, best_model = np.ones(n, bool), None
    if n < 3:
        return best_mask, None
    scale = np.median(np.linalg.norm(pts - pts.mean(0), axis=1)) + 1e-9
    for _ in range(iters):
        i, j, k = rng.choice(n, 3, replace=False)
        model = circle_from_3pts(pts[i], pts[j], pts[k])
        if model is None:
            continue
        center, normal, radius = model
        v = pts - center
        d_along = v @ normal
        d_perp = np.linalg.norm(v - np.outer(d_along, normal), axis=1)
        resid = np.sqrt((d_perp - radius) ** 2 + d_along ** 2)
        mask = resid < thresh * scale
        if mask.sum() > best_mask.sum():
            best_mask, best_model = mask, model
    return best_mask, best_model


def color_stats(img):
    """Robust per-channel mean and std (5th/95th percentile clipping)."""
    x = np.asarray(img, dtype=np.float32) / 255.0
    flat = x.reshape(-1, 3)
    lo = np.percentile(flat, 5, axis=0)
    hi = np.percentile(flat, 95, axis=0)
    sel = flat[(flat >= lo).all(1) & (flat <= hi).all(1)]
    if len(sel) < 16:
        sel = flat
    return sel.mean(0), sel.std(0) + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-observations", type=int, default=50)
    ap.add_argument("--circle-thresh", type=float, default=0.25)
    ap.add_argument("--dist-mad", type=float, default=3.0)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--normalize", action="store_true",
                    help="color-normalize all images to a shared target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "processed"))
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    proc = Path(args.out)

    data = json.loads(SFM.read_text())
    frames = data["frames"]
    print(f"registered cameras: {len(frames)}")

    kept = [f for f in frames if f["num_points3D"] >= args.min_observations]
    print(f"after observation filter (>={args.min_observations} points): {len(kept)}")

    centers = np.array([camera_center(f["R_wc"], f["t"]) for f in kept])
    mask, model = ransac_circle_3d(
        centers, args.circle_thresh, seed=args.seed
    )
    if model is not None and mask.sum() >= max(12, 0.3 * len(kept)):
        kept = [f for f, m in zip(kept, mask) if m]
        print(f"after RANSAC ring selection: {len(kept)}  "
              f"(radius={model[2]:.2f})")
    else:
        # Robust fallback: drop cameras whose distance from the centroid is a
        # strong outlier (extreme close-ups or far views).
        c = centers.mean(0)
        dist = np.linalg.norm(centers - c, axis=1)
        med = np.median(dist)
        mad = np.median(np.abs(dist - med)) + 1e-9
        mask = np.abs(dist - med) < args.dist_mad * mad
        filtered = [f for f, m in zip(kept, mask) if m]
        if len(filtered) >= max(12, 0.3 * len(kept)):
            kept = filtered
        print(f"RANSAC circle not confident; distance filter kept {len(kept)}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(kept))
    n_val = max(4, int(round(len(kept) * args.val_fraction)))
    val_idx = set(order[:n_val].tolist())

    records = []
    # --- first pass: statistics for normalization --------------------------
    stats = []
    for f in kept:
        img = Image.open(RAW / f["file"]).convert("RGB")
        stats.append(color_stats(img))
    means = np.array([s[0] for s in stats])
    stds = np.array([s[1] for s in stats])
    target_mean = np.median(means, axis=0)
    target_std = np.median(stds, axis=0)

    if not args.execute:
        print("dry run; pass --execute")
        print("normalize:", args.normalize, "max_side:", args.max_side)
        return

    img_dir = proc / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, (f, (mean, std)) in enumerate(zip(kept, stats)):
        img = Image.open(RAW / f["file"]).convert("RGB")
        if args.normalize:
            x = np.asarray(img, dtype=np.float32) / 255.0
            x = (x - mean) / std * target_std + target_mean
            x = np.clip(x, 0, 1)
            img = Image.fromarray((x * 255).astype(np.uint8))
        w, h = img.size
        scale = args.max_side / max(w, h)
        nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        name = f"{i:04d}.jpg"
        img.save(img_dir / name, quality=92)

        K = np.asarray(f["K"], dtype=float) * np.array(
            [[scale, 1, scale], [1, scale, scale], [1, 1, 1]], dtype=float
        )
        records.append({
            "file": f"images/{name}",
            "split": "val" if i in val_idx else "train",
            "R_wc": f["R_wc"], "t": f["t"], "K": K.tolist(),
            "width": nw, "height": nh,
        })

    points = np.array([p["xyz"] for p in data["points"]], dtype=np.float32)
    colors = np.array([p["rgb"] for p in data["points"]], dtype=np.float32) / 255.0
    np.save(proc / "points.npy", points)
    np.save(proc / "colors.npy", colors)

    meta = {
        "num_images": len(records),
        "num_train": sum(r["split"] == "train" for r in records),
        "num_val": sum(r["split"] == "val" for r in records),
        "num_points": int(len(points)),
        "max_side": args.max_side,
        "normalize": args.normalize,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "extent": np.ptp(points, axis=0).tolist(),
        "center": points.mean(axis=0).tolist(),
    }
    (proc / "cameras.json").write_text(json.dumps({"frames": records, "meta": meta},
                                                  indent=2))
    print(f"wrote {len(records)} images to {img_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
