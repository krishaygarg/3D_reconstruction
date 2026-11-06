#!/usr/bin/env python3
"""
ACM AI - End-to-End 3D Reconstruction Pipeline
================================================
Unified master workflow connecting:
  Phase 1: Ingestion / Capture & Preprocessing (Shake trimming, sampling, blur filtering)
  Phase 2: Camera Pose Estimation & Structure-from-Motion (COLMAP via pycolmap)
  Phase 3: Image Undistortion & Packaging for 3D Gaussian Splatting (gsplat / Nerfstudio)

Modes:
- Default: Streams uCO3D shoe video from Hugging Face via RemoteZip, extracts frames,
  runs COLMAP, and packages the undistorted dataset into colmap_undistorted.zip.
- Local Video: python3 main.py --video path/to/orbit.mp4
- Handoff ZIP: python3 main.py --input group2_handoff.zip
- Synthetic Orbit Test: python3 main.py --synthetic
"""

import sys
import shutil
import zipfile
import argparse
import logging
from pathlib import Path

from video import extract, extract_frames
from colmap import colmap_reconstruction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MasterPipeline")

# Default uCO3D dataset parameters (compatible with Krishay & Santiago setup)
BASE_UCO3D = "https://huggingface.co/datasets/facebook/uco3d/resolve/main/"
DEFAULT_RGB_ZIP = BASE_UCO3D + "part_rgb_videos_0232.zip"
DEFAULT_UCO3D_DIR = "uco3d_shoe"


def run_pipeline(
    input_source=None,
    workspace: str = "ws",
    n_frames: int = 100,
    max_side: int = 1280,
    quality_filter: bool = True,
    matching_mode: str = "exhaustive",
    archive_name: str = "colmap_undistorted.zip",
    synthetic_mode: bool = False
):
    ws = Path(workspace).resolve()
    images_dir = ws / "images"
    sparse_dir = ws / "sparse"
    undist_dir = ws / "undistorted"
    db_path = ws / "database.db"

    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    undist_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("      ACM AI - END-TO-END 3D RECONSTRUCTION PIPELINE")
    print("=" * 65)

    # 1. Ingestion / Frame Extraction
    if synthetic_mode:
        from generate_sample_video import create_synthetic_orbit_video
        synth_video = ws / "synthetic_orbit.mp4"
        logger.info(f"Generating synthetic 360-degree orbit video at {synth_video}...")
        create_synthetic_orbit_video(str(synth_video), duration_sec=16.0, fps=30, fast_mode=True)
        video_path = synth_video
        logger.info(f"Extracting & quality-filtering {n_frames} frames from synthetic orbit...")
        saved = extract_frames(n_frames, max_side, video_path, images_dir, quality_filter=quality_filter)

    elif input_source and Path(input_source).is_file() and str(input_source).endswith(".zip"):
        # Group 1 Handoff ZIP archive
        zip_path = Path(input_source).resolve()
        logger.info(f"Ingesting pre-packaged handoff archive: {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".jpg") or member.endswith(".png"):
                    zf.extract(member, images_dir)
                elif member.endswith(".json"):
                    zf.extract(member, ws)
        saved = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
        logger.info(f"Extracted {saved} frames from archive into {images_dir}")

    elif input_source and Path(input_source).is_dir():
        # Pre-extracted image directory
        src_dir = Path(input_source).resolve()
        logger.info(f"Ingesting images from directory: {src_dir}...")
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        for f in src_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                shutil.copy(f, images_dir / f.name)
        saved = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
        logger.info(f"Ingested {saved} frames into {images_dir}")

    elif input_source and Path(input_source).is_file():
        # Local video file
        video_path = Path(input_source).resolve()
        logger.info(f"Processing local video file: {video_path}...")
        saved = extract_frames(n_frames, max_side, video_path, images_dir, quality_filter=quality_filter)

    else:
        # Default: Remote uCO3D streaming download via RemoteZip
        url = input_source if input_source else DEFAULT_RGB_ZIP
        logger.info("No local video provided. Streaming uCO3D shoe dataset from Hugging Face...")
        video_path = extract(url, DEFAULT_UCO3D_DIR)
        saved = extract_frames(n_frames, max_side, video_path, images_dir, quality_filter=quality_filter)

    # 2. Structure-from-Motion (COLMAP via pycolmap)
    archive_path = ws / archive_name
    recon_summary = colmap_reconstruction(
        db=db_path,
        IMAGES=images_dir,
        SPARSE=sparse_dir,
        UNDIST=undist_dir,
        saved=saved,
        matching_mode=matching_mode,
        archive_zip=archive_path
    )

    # Also create shoe_colmap.zip alias for zero-config notebook compatibility
    shoe_zip = ws / "shoe_colmap.zip"
    if shoe_zip != archive_path and archive_path.exists():
        shutil.copy(archive_path, shoe_zip)

    print("\n" + "=" * 65)
    print("      RECONSTRUCTION COMPLETE - READY FOR 3D GAUSSIAN SPLATTING")
    print("=" * 65)
    print(f" Registered Cameras:  {recon_summary['num_registered_images']} / {recon_summary['total_images']}")
    print(f" 3D Sparse Points:    {recon_summary['num_points3d']}")
    print(f" Undistorted Dataset: {recon_summary['undistorted_dir']}")
    print(f" Handoff Archive:     {recon_summary['archive_zip']}")
    print(f" Colab Archive Alias: {shoe_zip}")
    print("\n NEXT STEPS FOR 3D GAUSSIAN SPLATTING:")
    print(" 1. Upload 'shoe_colmap.zip' (or colmap_undistorted.zip) to Google Drive.")
    print(" 2. Open 'train_gsplat_shoe.ipynb' in Google Colab (with T4 GPU runtime).")
    print(" 3. Run all cells to train gsplat for 7K-30K steps.")
    print(" 4. Download the generated .ply model and view in SuperSplat: https://playcanvas.com/supersplat/editor")
    print("=" * 65 + "\n")
    return recon_summary


def main():
    parser = argparse.ArgumentParser(
        description="ACM AI - End-to-End 3D Reconstruction Pipeline (Video → Preprocessing → COLMAP → 3DGS)"
    )
    parser.add_argument("-v", "--video", default=None, help="Path to input video file (e.g. raw_orbit.mp4)")
    parser.add_argument("-i", "--input", default=None, help="Input source: video file, images directory, or handoff zip archive")
    parser.add_argument("-w", "--workspace", default="ws", help="Workspace directory (default: ws)")
    parser.add_argument("-n", "--count", type=int, default=100, help="Target frame count within [80, 130] (default: 100)")
    parser.add_argument("--max-side", type=int, default=1280, help="Max edge in pixels (default: 1280)")
    parser.add_argument("--no-filter", action="store_true", help="Disable blur filtering and handshake trimming")
    parser.add_argument("--sequential", action="store_true", help="Use sequential feature matching (faster for orbits)")
    parser.add_argument("--archive", default="colmap_undistorted.zip", help="Output zip archive name (default: colmap_undistorted.zip)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test orbit video and run full reconstruction offline")
    args = parser.parse_args()

    input_source = args.video if args.video else args.input

    run_pipeline(
        input_source=input_source,
        workspace=args.workspace,
        n_frames=args.count,
        max_side=args.max_side,
        quality_filter=not args.no_filter,
        matching_mode="sequential" if args.sequential else "exhaustive",
        archive_name=args.archive,
        synthetic_mode=args.synthetic
    )


if __name__ == "__main__":
    main()

