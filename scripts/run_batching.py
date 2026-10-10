#!/usr/bin/env python3
"""Check that the memory-banding is a pure implementation detail.

A 4 GB GPU cannot materialize the full (pixels x Gaussians) weight tensor, so
rendering processes the image in horizontal bands and only evaluates Gaussians
whose support reaches each band. If this is really just a memory optimization,
the trained result should not depend on the band size. This script trains the
same model under several band sizes and compares the final PSNR, and also
checks how much the local-support cutoff changes the result.

Writes experiments/batching/batching.json and writeup/assets/batching.png.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gs.data import load_spheres, load_target  # noqa: E402
from gs.train2d import fit_2d, render_psnr  # noqa: E402
from gs.train3d import evaluate, fit_3d  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / "experiments" / "batching"
    out.mkdir(parents=True, exist_ok=True)
    res = {"2d": [], "3d": [], "cutoff": []}

    # --- 2D: coffee at N=2048, varying band height --------------------------
    target = load_target("coffee")
    H, W = target.shape[:2]
    print("=== 2D (coffee, N=2048, 600 steps) ===")
    for rows in (1, 2, 4, 8, 16, 32, 256):
        t0 = time.time()
        cloud, _ = fit_2d(
            target, n=2048, steps=600, seed=0, device=device,
            chunk=rows * W, log_every=0,
        )
        _, psnr = render_psnr(cloud, target)
        rec = {"band_rows": rows, "band_px": rows * W, "psnr": psnr,
               "seconds": time.time() - t0}
        res["2d"].append(rec)
        print(f"  rows={rows:4d}  band={rows*W:6d}px  psnr={psnr:.4f}")

    # --- 3D: sphere scene at N=2000, varying band height --------------------
    train, val, K, W3, H3 = load_spheres(device=device)
    print("=== 3D (spheres, N=2000, 800 steps) ===")
    for rows in (4, 8, 16, 32, 64, 160):
        t0 = time.time()
        cloud, _ = fit_3d(
            train, n=2000, steps=800, seed=0, device=device,
            band=rows * W3, log_every=0,
        )
        tr, _ = evaluate(cloud, train)
        va, _ = evaluate(cloud, val)
        rec = {"band_rows": rows, "band_px": rows * W3,
               "train": sum(tr) / len(tr), "val": sum(va) / len(va),
               "seconds": time.time() - t0}
        res["3d"].append(rec)
        print(f"  rows={rows:4d}  train={rec['train']:.4f}  val={rec['val']:.4f}")

    # --- cutoff: run with no local truncation -------------------------------
    print("=== cutoff check (3D, band=16 rows) ===")
    cloud, _ = fit_3d(
        train, n=2000, steps=800, seed=0, device=device, band=16 * W3,
        log_every=0, cutoff=100.0,
    )
    tr, _ = evaluate(cloud, train, cutoff=100.0)
    va, _ = evaluate(cloud, val, cutoff=100.0)
    res["cutoff"].append({"cutoff": 100.0, "band_rows": 16,
                          "train": sum(tr) / len(tr), "val": sum(va) / len(va)})
    print(f"  no cutoff  train={res['cutoff'][0]['train']:.4f} "
          f"val={res['cutoff'][0]['val']:.4f}")

    (out / "batching.json").write_text(json.dumps(res, indent=2))

    # --- figure -------------------------------------------------------------
    assets = ROOT / "writeup" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    xs = [r["band_rows"] for r in res["2d"]]
    axes[0].plot(xs, [r["psnr"] for r in res["2d"]], "-o")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("band height (rows)")
    axes[0].set_ylabel("final PSNR (dB)")
    axes[0].set_title("2D: band size vs. result (coffee, N=2048)")
    axes[0].grid(True, alpha=0.3)
    xs = [r["band_rows"] for r in res["3d"]]
    axes[1].plot(xs, [r["train"] for r in res["3d"]], "-o", label="train")
    axes[1].plot(xs, [r["val"] for r in res["3d"]], "-s", label="held-out")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("band height (rows)")
    axes[1].set_ylabel("final PSNR (dB)")
    axes[1].set_title("3D: band size vs. result (spheres, N=2000)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(assets / "batching.png", dpi=160)
    print("wrote", assets / "batching.png")


if __name__ == "__main__":
    main()
