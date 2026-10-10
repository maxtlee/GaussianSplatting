# Eiffel Tower extension

A real-world test of the renderer on uncontrolled internet photographs: 320
photos of the Eiffel Tower scraped from Wikimedia Commons, reconstructed with
COLMAP and fit with 3D Gaussians.

The hard parts, relative to the synthetic sphere scene, are that the photos

- have **unknown poses** (so structure-from-motion has to recover them),
- come from **many cameras with different exposures and white balance**,
- include **bad or inconsistent images** (night, fog, off-topic, crops),
- and the tower is a **repetitive, symmetric structure**, which is hard to match.

## Pipeline

```bash
# 1. scrape (dry-run by default)
python eiffel/fetch_images.py --limit 320 \
    --category "Category:Eiffel Tower from the northeast" ... --execute

# 2. structure from motion (COLMAP via pycolmap)
python eiffel/run_sfm.py --execute

# 3. select + normalize + downscale
python eiffel/prepare_dataset.py --normalize --max-side 640 --execute

# 4. fit
python eiffel/train_eiffel.py --tag pre+app --n 30000 --steps 4000 \
    --appearance train --loss huber --gif
```

## Choices

- **RANSAC image selection** (`prepare_dataset.py`): keep registered cameras
  with enough triangulated points, then RANSAC-fit a circle to the camera
  centers in the horizontal plane (COLMAP uses EXIF gravity, so the model is
  gravity-aligned) and keep the inliers. This rejects aerial views, close-ups,
  and off-topic frames and leaves a consistent ring of viewpoints.
- **Appearance / lighting compensation**, in two places:
  1. *Before training* (`--normalize`): every image is affine-corrected
     per channel so its robust mean and standard deviation match the median
     across the dataset — a shared target exposure and white balance.
  2. *During training* (`--appearance train`): a per-image per-channel gain and
     bias is optimized jointly with the Gaussians; it is applied to the render
     before the loss, so the Gaussians can learn one canonical color while each
     image absorbs its own exposure.
- **Robust loss** (`--loss huber`): downweights mismatched or badly exposed
  pixels so they do not dominate the gradient.
- **Sparse initialization**: Gaussians start at the SfM points with the point
  colors instead of a random cloud.

Raw images are not committed; see `CREDITS.md` for per-file license and author.
