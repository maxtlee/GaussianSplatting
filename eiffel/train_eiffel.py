#!/usr/bin/env python3
"""Train 3D Gaussians on the Eiffel Tower dataset.

Adds three things the sphere scene did not need:

- **sparse initialization** from the SfM point cloud,
- **per-image appearance correction** (a learnable per-channel gain and bias
  applied to the render before the loss), so exposure and white-balance
  differences between photos do not have to be baked into the Gaussians,
- an optional **Huber loss**, so badly exposed or mismatched pixels do not
  dominate the gradient.

At the end it renders a 120-frame orbit and writes a GIF.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gs.data import Camera  # noqa: E402
from gs.densify import DensifyConfig, densify_3d  # noqa: E402
from gs.gaussians3d import GaussianCloud3D  # noqa: E402
from gs.io import load_image  # noqa: E402
from gs.optim import AdamStore  # noqa: E402
from gs.raster import pixel_grid, render_chunk_3d, render_cloud_3d  # noqa: E402
from gs.train2d import auto_band, psnr_from_mse  # noqa: E402


def load_dataset(proc, device):
    d = json.loads((proc / "cameras.json").read_text())
    cams = []
    for f in d["frames"]:
        img = load_image(proc / f["file"]).to(device)
        R = torch.tensor(f["R_wc"], dtype=torch.float32, device=device)
        t = torch.tensor(f["t"], dtype=torch.float32, device=device)
        K = torch.tensor(f["K"], dtype=torch.float32, device=device)
        cam = Camera(R, t, K, img, name=f["file"])
        cams.append((cam, f["split"]))
    pts = np.load(proc / "points.npy")
    cols = np.load(proc / "colors.npy")
    return cams, pts, cols, d["meta"]


def estimate_scale(mu):
    sub = mu[: min(len(mu), 4096)]
    d = torch.cdist(sub, sub)
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.median()) * 0.5


def build_cloud(pts, cols, n, device, seed=0):
    if len(pts) > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), n, replace=False)
        pts, cols = pts[idx], cols[idx]
    mu = torch.tensor(pts, dtype=torch.float32, device=device).requires_grad_(True)
    scale = max(estimate_scale(mu.detach()), 1e-5)
    n_pts = mu.shape[0]
    log_s = torch.full((n_pts, 3), np.log(scale), device=device, requires_grad=True)
    quat = torch.zeros(n_pts, 4, device=device)
    quat[:, 0] = 1.0
    quat.requires_grad_(True)
    c = np.clip(cols, 1e-3, 1 - 1e-3)
    color_raw = torch.tensor(
        np.log(c / (1 - c)), dtype=torch.float32, device=device
    ).requires_grad_(True)
    op_raw = torch.full((n_pts,), -2.0, device=device, requires_grad=True)
    return GaussianCloud3D(mu, log_s, quat, color_raw, op_raw)


def look_at(eye, target, K, image, up=(0.0, 1.0, 0.0)):
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
    return Camera(R, -(R @ eye), K, image)


def loss_fn(diff, mode, delta=0.1):
    if mode == "huber":
        a = diff.abs()
        return torch.where(a < delta, 0.5 * a * a, delta * (a - 0.5 * delta)).mean()
    return (diff * diff).mean()


def train(args):
    device = args.device
    proc = Path(args.processed)
    all_cams, pts, cols, meta = load_dataset(proc, device)
    train_cams = [c for c, s in all_cams if s == "train"]
    val_cams = [c for c, s in all_cams if s == "val"]
    print(f"train {len(train_cams)}  val {len(val_cams)}  points {len(pts)}")

    cloud = build_cloud(pts, cols, args.n, device, seed=args.seed)
    print(f"Gaussians {cloud.n}  init scale {float(cloud.scale().mean()):.5f}")
    opt = AdamStore(cloud.params(), lr=args.lr)

    n_cams = len(all_cams)
    gain = torch.ones(n_cams, 3, device=device, requires_grad=True)
    bias = torch.zeros(n_cams, 3, device=device, requires_grad=True)
    app_opt = (
        torch.optim.Adam([gain, bias], lr=args.app_lr)
        if args.appearance == "train"
        else None
    )
    idx_of = {id(c): i for i, (c, _) in enumerate(all_cams)}

    gen = torch.Generator().manual_seed(args.seed)
    history = []
    grad_sum = torch.zeros(cloud.n, device=device)
    since = 0

    @torch.no_grad()
    def evaluate(cams, use_appearance):
        ps = []
        for cam in cams:
            img = render_cloud_3d(cloud, cam)
            if use_appearance and args.appearance == "train":
                i = idx_of[id(cam)]
                img = img * gain[i] + bias[i]
            ps.append(psnr_from_mse(((img - cam.image) ** 2).mean()))
        return float(sum(ps) / len(ps))

    for step in range(args.steps):
        ci = int(torch.randint(len(train_cams), (1,), generator=gen).item())
        cam = train_cams[ci]
        gi = idx_of[id(cam)]
        H, W = cam.H, cam.W
        P = H * W
        xy = pixel_grid(H, W, device=device)
        tgt = cam.image.reshape(-1, 3)
        band = args.band or auto_band(H, W, cloud.n)

        opt.zero_grad(cloud.params())
        if app_opt is not None:
            app_opt.zero_grad()
        total = 0.0
        for lo in range(0, P, band):
            hi = min(lo + band, P)
            r_lo, r_hi = lo // W, (hi - 1) // W
            C = render_chunk_3d(xy[lo:hi], r_lo + 0.5, r_hi + 0.5, cloud, cam)
            if args.appearance == "train":
                C = C * gain[gi] + bias[gi]
            diff = C - tgt[lo:hi]
            term = loss_fn(diff, args.loss) * (hi - lo) / P
            if term.requires_grad:
                term.backward()
            total += float(term.detach())
        if cloud.mu.grad is not None:
            grad_sum += cloud.mu.grad.detach().norm(dim=1)
        since += 1
        opt.step(cloud.params())
        if app_opt is not None:
            app_opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            tr = evaluate(train_cams, use_appearance=False)
            va = evaluate(val_cams, use_appearance=False)
            history.append({"step": step, "n": cloud.n, "loss": total,
                            "train_psnr": tr, "val_psnr": va})
            print(f"step {step:5d} n={cloud.n:6d} loss={total:.4e} "
                  f"train={tr:.2f} val={va:.2f}", flush=True)

    return cloud, history, all_cams, train_cams, val_cams, meta, gain, bias


def make_orbit(cloud, train_cams, meta, device, width=240, height=340):
    center = torch.tensor(meta["center"], dtype=torch.float32, device=device)
    extent = np.asarray(meta["extent"], dtype=float)
    radius = float(max(extent[0], extent[2]) * 1.4 + extent[1] * 0.2)
    cam_y = float(center[1] + 0.15 * extent[1])
    target = center + torch.tensor([0.0, 0.1 * extent[1], 0.0], device=device)
    K = torch.tensor(
        [[0.9 * width, 0, width / 2], [0, 0.9 * width, height / 2], [0, 0, 1]],
        dtype=torch.float32, device=device,
    )
    blank = torch.zeros(height, width, 3, device=device)
    frames = []
    for i in range(120):
        az = 2 * np.pi * i / 120
        eye = [float(center[0]) + radius * np.cos(az),
               cam_y,
               float(center[2]) + radius * np.sin(az)]
        cam = look_at(eye, target, K, blank)
        img = render_cloud_3d(cloud, cam)
        arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=str(ROOT / "eiffel" / "data" / "processed"))
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1.6e-4)
    ap.add_argument("--app-lr", type=float, default=1e-3)
    ap.add_argument("--appearance", choices=["none", "train"], default="none")
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--band", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--densify", action="store_true")
    ap.add_argument("--densify-every", type=int, default=500)
    ap.add_argument("--out", default=str(ROOT / "eiffel" / "experiments"))
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cloud, history, all_cams, train_cams, val_cams, meta, gain, bias = train(args)
    # Attach the processed path so make_orbit can find points if needed.
    meta = {**meta, "_proc": args.processed}

    # Save metrics.
    (out / "metrics.json").write_text(json.dumps({
        "tag": args.tag, "args": vars(args), "history": history,
        "final_n": int(cloud.n),
        "train_psnr": history[-1]["train_psnr"],
        "val_psnr": history[-1]["val_psnr"],
        "seconds": time.time() - t0,
    }, indent=2, default=str))
    print(f"[done] {args.tag} n={cloud.n} train={history[-1]['train_psnr']:.2f} "
          f"val={history[-1]['val_psnr']:.2f} ({time.time()-t0:.0f}s)")

    # Comparison images for a couple of views.
    import torch as _t  # noqa

    for split, cams in (("train", train_cams), ("val", val_cams)):
        for cam in cams[:2]:
            img = render_cloud_3d(cloud, cam)
            pair = _t.cat([cam.image, img.clamp(0, 1)], dim=0)
            arr = (pair.cpu().numpy() * 255).astype("uint8")
            Image.fromarray(arr).save(out / f"{split}_{Path(cam.name).stem}.png")

    if args.gif:
        frames = make_orbit(cloud, train_cams, meta, args.device)
        frames[0].save(out / "orbit.gif", save_all=True,
                       append_images=frames[1:], duration=70, loop=0,
                       optimize=True)
        # a contact sheet for the PDF
        picks = frames[::15][:8]
        sheet = Image.new("RGB", (sum(f.width for f in picks), picks[0].height))
        x = 0
        for f in picks:
            sheet.paste(f, (x, 0))
            x += f.width
        sheet.save(out / "orbit_strip.png")
        print("wrote", out / "orbit.gif")


if __name__ == "__main__":
    main()
