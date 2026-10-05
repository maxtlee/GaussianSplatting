# GaussianSplatting

A from-scratch PyTorch implementation of Gaussian Splatting. Instead of storing
pixels, a scene is represented as a cloud of translucent Gaussians — each with a
position, a shape (scale + rotation), a color, and an opacity — and rendered by
projecting them onto the image plane and alpha-compositing them front to back.
Every step is differentiable, so the primitives are optimized directly against
reference images until the render matches.

Two stages:

- **2D.** Fit a single image with a set of 2D Gaussians, and study how render
  quality scales with the number of primitives.
- **3D.** Reconstruct a small scene from many posed views. Each 3D Gaussian is
  projected through a pinhole camera into the same 2D rasterizer, sorted by
  camera-space depth.

Adaptive densification (cloning small under-fit Gaussians, splitting large ones,
and pruning nearly transparent ones) grows the cloud where the image needs detail.

Developed for 15-474/674 (Neural Graphics).

## Setup

Needs Python 3.10+ (developed on 3.14) and pip.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_env.py
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope Process Bypass` first, or activate in `cmd` with
`.venv\Scripts\activate.bat`.

**GPU notes**

- macOS: PyTorch uses the Apple GPU (MPS) automatically, nothing to configure.
- Windows / Linux with an NVIDIA GPU: the default PyPI PyTorch wheel targets a
  recent CUDA and works on most cards.
- The development GPU is an old GTX 1050 Ti (Pascal, sm_61). The default CUDA 13
  wheel dropped Pascal and fails with "no kernel image", so on that machine use
  the CUDA 12.6 build:
  ```bash
  pip install -r requirements-cu126.txt
  ```
- CPU-only:
  ```bash
  pip install numpy pillow matplotlib
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- `gs.get_device()` selects CUDA, then Apple MPS, then CPU automatically.

## Assets

- `assets/images/` — three 256x256 RGB targets, each fit independently:
  `coffee.png` (large smooth regions), `astronaut.png` (mixed smooth + fine
  detail), `cat.png` (dense high-frequency fur).
- `assets/spheres/` — a small synthetic scene of colored spheres rendered from
  56 known camera poses: 44 training views and 12 held-out views, all 160x160,
  with shared intrinsics and per-frame poses in `cameras.json`.

Image credits: `coffee` (CC0) and `cat` (CC0) from the scikit-image sample data;
`astronaut` (public domain, NASA).

`cameras.json` uses the convention `X_cam = R_wc @ X_world + t` (OpenCV), with
`u = fx * Xc / Zc + cx`, `v = fy * Yc / Zc + cy`, and `depth = Zc`. The full 3x3
intrinsics `K` is shared by every view (`fx = K[0][0]`, `fy = K[1][1]`,
`cx = K[0][2]`, `cy = K[1][2]`).

## Usage

- Environment check (torch devices, a real matmul, image IO, PSNR):
  ```bash
  python scripts/check_env.py
  ```

Renderer, training, densification, and the 2D/3D run scripts are being built out;
the entry points will be listed here as they land.

## Layout

```
gs/           library: device selection, image IO, metrics (renderer/trainer to come)
scripts/      runnable entry points
tests/        self-contained checks
assets/       sample images and the posed sphere scene
```

## Tests

Each test file runs on its own:

```bash
python tests/test_io.py
python tests/test_metrics.py
```
