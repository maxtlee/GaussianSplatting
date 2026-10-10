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

- **2D study.** Fits each target at `N = 256, 1024, 4096` and runs an adaptive
  densification schedule, writing one directory per run under `experiments/2d/`.
  It is resumable (existing `metrics.json` files are reused), so it can be
  interrupted and restarted:
  ```bash
  python scripts/run_2d_suite.py --steps 2000
  ```

- **3D scene.** Train a fixed cloud, then a densified one:
  ```bash
  python scripts/run_3d.py --tag fixed    --n 4000 --steps 1500
  python scripts/run_3d.py --tag densified --densify --start 1000 --budget 4000 --steps 1500
  ```
  Each writes `experiments/3d/<tag>/`: metrics, target/render comparisons for
  training and held-out views, and a 24-frame novel orbit (`orbit/` PNGs plus
  `orbit.mp4`).

- **Figures.** Build the report figures from the saved metrics:
  ```bash
  python scripts/make_figures.py
  ```

- **On the development GPU.** The GTX 1050 Ti has 4 GB, so rendering runs in
  horizontal bands and evaluates only the Gaussians whose support reaches each
  band. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce
  allocator fragmentation on the longer runs.

## Layout

```
gs/gaussians2d.py   covariance from scale/rotation, gaussian weight, 2D cloud
gs/gaussians3d.py   quaternion rotation, 3D covariance, pinhole projection
gs/raster.py        banded differentiable alpha compositing (2D and 3D)
gs/densify.py       adaptive clone / split / prune, resizable optimizer state
gs/optim.py         Adam with resizable per-Gaussian moments
gs/train2d.py       fit_2d and PSNR from the exact full-image MSE
gs/train3d.py       fit_3d and multi-view evaluation
gs/data.py          targets and posed cameras
gs/io.py metrics.py device.py viz.py   image IO, MSE/PSNR, device, figures
scripts/            runnable entry points
tests/              self-contained checks
assets/             sample images and the posed sphere scene
writeup/            reports (progression + submission), figures, orbits
eiffel/             extension: scrape, SfM, and 3DGS on real photos
experiments/        generated renders and metrics (not committed)
```

## Writeups

- **Progression notes** — a self-contained walkthrough for a general audience:
  [HTML](writeup/progression.html) · [PDF](writeup/progression.pdf)
- **Submission** — organised by problem, with the required answers plus the
  extra experiments: [HTML](writeup/submission.html) ·
  [PDF](writeup/submission.pdf)
- Rebuild the PDFs: `pip install -r requirements-docs.txt && python scripts/build_writeup.py`

## Experiments beyond the basics

- **Batching validation** (`scripts/run_batching.py`): retrains the same model
  at several band heights and with the local-support cutoff disabled, showing
  the memory batching does not change the result (and that the unbanded variant
  does not fit in 4 GB).
- **Training curves** (`scripts/run_3d.py --eval-every N`): train and held-out
  PSNR over 4000 steps, showing the held-out plateau.
- **Eiffel Tower extension** (`eiffel/`): 320 scraped Wikimedia Commons photos,
  COLMAP structure-from-motion, RANSAC viewpoint selection, and lighting
  compensation before and during training. See [eiffel/README.md](eiffel/README.md).

## Tests

Each test file runs on its own:

```bash
python tests/test_io.py
python tests/test_metrics.py
python tests/test_gaussians2d.py
python tests/test_gaussians3d.py
python tests/test_raster.py
python tests/test_optim.py
python tests/test_train.py
```
