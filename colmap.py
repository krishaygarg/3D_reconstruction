#!/usr/bin/env python3
"""
ACM AI - 3D Reconstruction Pipeline: COLMAP Feature Extraction & Mapping
=========================================================================
Uses pycolmap to:
1. Extract SIFT features with OPENCV camera model and SINGLE camera intrinsics.
2. Run exhaustive or sequential feature matching.
3. Compute incremental 3D reconstruction / bundle adjustment.
4. Undistort images into standard COLMAP dataset format for 3D Gaussian Splatting (gsplat / Nerfstudio).
5. Package undistorted dataset into a zip archive for Colab training.
"""

import sys
import logging
import zipfile
from pathlib import Path
from typing import Optional, Dict

import pycolmap

logger = logging.getLogger("ColmapPipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def colmap_reconstruction(
    db,
    IMAGES,
    SPARSE,
    UNDIST,
    saved: Optional[int] = None,
    matching_mode: str = "exhaustive",
    archive_zip: Optional[Path] = None
) -> Dict:
    """
    Runs the full pycolmap Structure-from-Motion pipeline:
    - Feature extraction
    - Feature matching
    - Incremental mapping
    - Image undistortion
    - Moves sparse models to UNDIST/sparse/0/
    """
    db = Path(db).resolve()
    IMAGES = Path(IMAGES).resolve()
    SPARSE = Path(SPARSE).resolve()
    UNDIST = Path(UNDIST).resolve()

    SPARSE.mkdir(parents=True, exist_ok=True)
    UNDIST.mkdir(parents=True, exist_ok=True)

    if db.exists():
        db.unlink()

    # Filter only valid image files to prevent bitmap errors on metadata JSON files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_names = sorted([f.name for f in IMAGES.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    if not image_names:
        raise FileNotFoundError(f"No valid image files found in {IMAGES}")

    logger.info(f"Extracting SIFT features for {len(image_names)} images using OPENCV camera model...")
    reader_options = pycolmap.ImageReaderOptions(camera_model="OPENCV")
    pycolmap.extract_features(
        str(db),
        str(IMAGES),
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options
    )

    if matching_mode == "sequential":
        logger.info("Running sequential feature matching with loop detection...")
        pycolmap.match_sequential(str(db))
    else:
        logger.info("Running exhaustive feature matching (comparing all image pairs)...")
        pycolmap.match_exhaustive(str(db))

    logger.info("Running incremental 3D mapping and bundle adjustment...")
    recs = pycolmap.incremental_mapping(str(db), str(IMAGES), str(SPARSE))

    if not recs:
        raise RuntimeError(
            "COLMAP incremental mapping did not reconstruct any cameras. "
            "Ensure the input frames have sufficient visual overlap, steady illumination, and texture."
        )

    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    num_reg = best.num_reg_images()
    num_points = best.num_points3D()
    logger.info(
        f"Reconstruction successful! Best model #{best_id}: "
        f"{num_reg}/{len(image_names)} registered images ({num_reg/len(image_names)*100:.1f}%), "
        f"{num_points} 3D points."
    )

    logger.info(f"Undistorting images into {UNDIST}...")
    pycolmap.undistort_images(str(UNDIST), str(SPARSE / str(best_id)), str(IMAGES))

    target_sparse_dir = UNDIST / "sparse" / "0"
    target_sparse_dir.mkdir(parents=True, exist_ok=True)
    for f in (UNDIST / "sparse").glob("*.bin"):
        f.rename(target_sparse_dir / f.name)

    logger.info(f"Undistorted COLMAP dataset prepared at: {UNDIST}")

    # Optionally package into zip for Colab / Google Drive
    zip_path = None
    if archive_zip is not None:
        zip_path = Path(archive_zip).resolve()
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            zip_path.unlink()
        logger.info(f"Packaging undistorted COLMAP dataset into handoff archive: {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in UNDIST.rglob("*"):
                if item.is_file() and not item.name.startswith("."):
                    arcname = item.relative_to(UNDIST)
                    zf.write(item, arcname=str(arcname))
        logger.info(f"COLMAP archive created: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

    return {
        "num_registered_images": num_reg,
        "total_images": len(image_names),
        "num_points3d": num_points,
        "best_model_id": best_id,
        "undistorted_dir": str(UNDIST),
        "archive_zip": str(zip_path) if zip_path else None
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pycolmap 3D Reconstruction")
    parser.add_argument("-i", "--images", required=True, help="Path to input images directory")
    parser.add_argument("-w", "--workspace", default="ws", help="Path to workspace directory (default: ws)")
    parser.add_argument("-z", "--zip", default="colmap_undistorted.zip", help="Path for output zipped dataset (default: colmap_undistorted.zip)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential matching instead of exhaustive")
    args = parser.parse_args()

    ws = Path(args.workspace)
    images_dir = Path(args.images)
    db_path = ws / "database.db"
    sparse_dir = ws / "sparse"
    undist_dir = ws / "undistorted"

    colmap_reconstruction(
        db=db_path,
        IMAGES=images_dir,
        SPARSE=sparse_dir,
        UNDIST=undist_dir,
        matching_mode="sequential" if args.sequential else "exhaustive",
        archive_zip=Path(args.zip)
    )

