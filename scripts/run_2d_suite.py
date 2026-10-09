#!/usr/bin/env python3
"""Run the 2D image-fitting study.

Fits each target image at several Gaussian counts and also runs an adaptive
densification schedule, writing one directory per run under ``experiments/2d/``
with the final render, a target/render/error comparison, and metrics. Runs are
resumable: an existing ``metrics.json`` is reused.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gs.data import TWO_D_TARGETS, load_target  # noqa: E402
from gs.densify import DensifyConfig  # noqa: E402
from gs.train2d import fit_2d, render_psnr  # noqa: E402
from gs.viz import save_comparison  # noqa: E402


def run_one(name, n, steps, seed, out_dir, densify=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {out_dir.name}")
        return json.loads(metrics_path.read_text())

    target = load_target(name)
    t0 = time.time()
    cloud, history = fit_2d(
        target, n=n, steps=steps, seed=seed, densify=densify, log_every=50
    )
    img, psnr = render_psnr(cloud, target)
    save_comparison(target.cpu(), img.cpu(), out_dir / "comparison.png")
    meta = {
        "image": name,
        "n_target": None if n is None else int(n),
        "steps": steps,
        "seed": seed,
        "final_n": int(cloud.n),
        "final_psnr": psnr,
        "seconds": time.time() - t0,
        "history": history,
        "densify": None if densify is None else densify.__dict__,
    }
    metrics_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[done] {out_dir.name}  n={cloud.n}  psnr={psnr:.2f}  "
        f"({meta['seconds']:.0f}s)"
    )
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--counts", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--images", nargs="+", default=TWO_D_TARGETS)
    ap.add_argument("--densify-budget", type=int, default=1024)
    ap.add_argument("--densify-start", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "2d"))
    ap.add_argument("--only-fixed", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.images:
        for n in args.counts:
            run_one(name, n, args.steps, args.seed, out / f"{name}_n{n}")
        if not args.only_fixed:
            cfg = DensifyConfig(
                every=200,
                budget=args.densify_budget,
                start=args.densify_start,
                top_fraction=0.25,
                size_threshold=0.02 * 256,
                until=int(args.steps * 0.8),
            )
            run_one(
                name,
                None,
                args.steps,
                args.seed,
                out / f"{name}_dens{args.densify_budget}",
                densify=cfg,
            )

    print("2D suite complete")


if __name__ == "__main__":
    main()
