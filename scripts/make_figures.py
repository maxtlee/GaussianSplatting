#!/usr/bin/env python3
"""Build the report figures from saved experiment metrics."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = ROOT / "experiments"
ASSETS = ROOT / "writeup" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

COLORS = {"coffee": "#8c5a3c", "astronaut": "#2f6db3", "cat": "#c45a2a"}
MARKERS = {256: "o", 1024: "s", 4096: "^"}


def load_2d():
    runs = []
    for p in sorted((EXP / "2d").glob("*/metrics.json")):
        runs.append(json.loads(p.read_text()))
    return runs


def figure_psnr_vs_count(runs):
    fixed = [r for r in runs if r["densify"] is None]
    dens = [r for r in runs if r["densify"] is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name in ("coffee", "astronaut", "cat"):
        pts = sorted(
            [(r["n_target"], r["final_psnr"]) for r in fixed if r["image"] == name]
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=COLORS[name], label=name, markersize=6)
    for r in dens:
        ax.scatter(
            [r["final_n"]],
            [r["final_psnr"]],
            color=COLORS[r["image"]],
            marker="*",
            s=160,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
    ax.scatter([], [], color="gray", marker="*", s=160, edgecolor="black",
               label="densified")
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 1024, 4096])
    ax.set_xticklabels(["256", "1024", "4096"])
    ax.set_xlabel("Gaussians (N)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("2D reconstruction quality vs. number of Gaussians")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "psnr_vs_count.png", dpi=160)
    plt.close(fig)


def figure_training_curves(runs):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for r in sorted(runs, key=lambda r: (r["image"], r["n_target"] or 0)):
        if r["densify"] is not None:
            continue
        h = r["history"]
        xs = [e["step"] for e in h]
        ys = [e["psnr"] for e in h]
        ax.plot(
            xs,
            ys,
            color=COLORS[r["image"]],
            alpha=0.75 if r["n_target"] == 4096 else 0.4,
            linewidth=1.4 if r["n_target"] == 4096 else 1.0,
        )
    ax.set_xlabel("optimization step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("2D training progress (color = image, dark = N=4096)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_curves.png", dpi=160)
    plt.close(fig)


def figure_3d():
    runs = []
    for p in sorted((EXP / "3d").glob("*/metrics.json")):
        m = json.loads(p.read_text())
        runs.append((p.parent.name, m))
    if not runs:
        print("3D metrics not ready; skipping")
        return
    labels = [f"{tag}\nN={m['final_n']}" for tag, m in runs]
    train = [m["train_psnr_mean"] for _, m in runs]
    val = [m["val_psnr_mean"] for _, m in runs]
    x = range(len(runs))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar([i - 0.18 for i in x], train, width=0.34, label="training", color="#2f6db3")
    ax.bar([i + 0.18 for i in x], val, width=0.34, label="held-out", color="#c45a2a")
    for i, (t, v) in enumerate(zip(train, val)):
        ax.text(i - 0.18, t + 0.25, f"{t:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.18, v + 0.25, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("3D reconstruction: fixed vs. densified")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "3d_fixed_vs_densified.png", dpi=160)
    plt.close(fig)


def figure_3d_curve():
    p = EXP / "3d" / "curve_long" / "metrics.json"
    if not p.exists():
        print("no 3D curve run yet")
        return
    m = json.loads(p.read_text())
    h = [e for e in m["history"] if "train" in e]
    if not h:
        return
    steps = [e["step"] for e in h]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(steps, [e["train"] for e in h], "-o", markersize=3, label="training views")
    ax.plot(steps, [e["val"] for e in h], "-s", markersize=3, label="held-out views")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"3D PSNR over training (N={m['final_n']})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "3d_training_curve.png", dpi=160)
    plt.close(fig)
    print("wrote", ASSETS / "3d_training_curve.png")


def main():
    runs = load_2d()
    if runs:
        figure_psnr_vs_count(runs)
        figure_training_curves(runs)
        print(f"wrote 2D figures from {len(runs)} runs")
    else:
        print("no 2D metrics yet")
    figure_3d()
    figure_3d_curve()
    print("figures written to", ASSETS)


if __name__ == "__main__":
    main()
