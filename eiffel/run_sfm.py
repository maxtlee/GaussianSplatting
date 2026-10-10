#!/usr/bin/env python3
"""Structure-from-motion for the scraped Eiffel Tower photographs.

Runs COLMAP (through pycolmap) feature extraction, exhaustive matching, and
incremental mapping, then exports the registered cameras in the same
world-to-camera convention the renderer uses
(``x_cam = R_wc @ X_world + t``) plus the sparse 3D points for initialization.

    python eiffel/run_sfm.py --execute
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pycolmap

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
OUT = DATA / "sfm"


def export_reconstruction(rec, path):
    frames = []
    for image_id in rec.reg_image_ids():
        im = rec.image(image_id)
        pose = im.cam_from_world()
        R = np.asarray(pose.rotation.matrix(), dtype=float)
        t = np.asarray(pose.translation, dtype=float).reshape(3)
        cam = im.camera
        K = np.asarray(cam.calibration_matrix(), dtype=float)
        frames.append(
            {
                "file": im.name,
                "R_wc": R.tolist(),
                "t": t.tolist(),
                "K": K.tolist(),
                "width": int(cam.width),
                "height": int(cam.height),
                "num_points3D": int(im.num_points3D),
            }
        )
    points = [
        {
            "xyz": np.asarray(p.xyz, dtype=float).tolist(),
            "rgb": np.asarray(p.color, dtype=int).tolist(),
            "error": float(p.error),
        }
        for p in rec.points3D.values()
    ]
    path.write_text(
        json.dumps(
            {
                "num_reg_images": rec.num_reg_images(),
                "num_points3D": rec.num_points3D(),
                "frames": frames,
                "points": points,
            },
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "database.db"
    n_images = len(list(RAW.glob("*.jpg"))) + len(list(RAW.glob("*.png")))
    print(f"images: {n_images}  db: {db}")
    if not args.execute:
        print("dry run; pass --execute")
        return

    if args.force:
        db.unlink(missing_ok=True)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = args.camera_model
    reader.default_focal_length_factor = 1.2
    fe = pycolmap.FeatureExtractionOptions()
    fe.max_image_size = args.max_image_size
    fe.num_threads = args.threads

    t0 = time.time()
    if not db.exists() or db.stat().st_size == 0:
        print("extracting SIFT features ...", flush=True)
        pycolmap.extract_features(
            db, RAW, reader_options=reader, extraction_options=fe
        )
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    print("matching (exhaustive) ...", flush=True)
    t1 = time.time()
    pycolmap.match_exhaustive(db)
    print(f"  done in {time.time()-t1:.0f}s", flush=True)

    print("incremental mapping ...", flush=True)
    t2 = time.time()
    reconstructions = pycolmap.incremental_mapping(db, RAW, OUT)
    if not reconstructions:
        print("no reconstruction produced")
        return
    rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    print(
        f"  {len(reconstructions)} models; best registered "
        f"{rec.num_reg_images()} images / {rec.num_points3D()} points "
        f"in {time.time()-t2:.0f}s",
        flush=True,
    )
    rec.write(OUT / "reconstruction")
    export_reconstruction(rec, OUT / "cameras.json")
    print("wrote", OUT / "cameras.json")


if __name__ == "__main__":
    main()
