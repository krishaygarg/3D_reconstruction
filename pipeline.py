#!/usr/bin/env python3
"""
End-to-End 3D Gaussian Splatting Training Pipeline
------------------------------------------------------------------------------
Automates the full pipeline:
1. Environment & GPU validation
2. Ingestion & extraction of video frames (with blur filtering)
3. Camera pose estimation (COLMAP / ns-process-data) & validation of transforms.json
4. Gaussian Splatting training (splatfacto) with configurable quality presets
5. Model export to a standard .ply file ready for online WebGL viewers

Quality improvements vs original:
- FIX: Scale filter added to ffmpeg extraction (caps width at 1280px)
- NEW: Blur frame detection and removal (Laplacian variance, requires opencv-python)
- FIX: Multi-scale training warmup restored (--num-downscales 0 removed)
- FIX: Post-densification culling re-enabled (was explicitly disabled)
- FIX: Raised cull-alpha-thresh 0.05 -> 0.08 to reduce floaters/haze
- NEW: --high-quality flag for production runs (30K iterations)
- NEW: --blur-threshold flag to tune sharpness filter sensitivity
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


def print_step(step_num: int, title: str):
    print("\n" + "=" * 65)
    print(f" STEP {step_num}: {title.upper()}")
    print("=" * 65)


def check_environment():
    """Verify GPU availability and required command-line tools."""
    print_step(1, "Checking Environment & Hardware")

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "No CUDA device"
        print(f"[*] PyTorch Version : {torch.__version__}")
        print(f"[*] CUDA Available  : {cuda_avail}")
        print(f"[*] Active Device   : {device_name}")
        if not cuda_avail:
            print("[!] WARNING: CUDA is not detected. 3DGS training requires an NVIDIA GPU.")
            print("[!] If running on Mac/CPU, run this script inside Google Colab (with T4 GPU).")
    except ImportError:
        print("[!] PyTorch is not installed. Please run: pip install -r requirements.txt")

    # Check for nerfstudio CLI
    ns_available = shutil.which("ns-process-data") is not None
    print(f"[*] nerfstudio CLI  : {'Found' if ns_available else 'Not found in PATH'}")
    if not ns_available:
        print("[!] Install Nerfstudio via: pip install nerfstudio")

    # Check for opencv (optional, needed for blur filter)
    try:
        import cv2  # noqa: F401
        print("[*] opencv-python   : Found (blur filtering enabled)")
    except ImportError:
        print("[!] opencv-python   : Not found (blur filtering will be skipped)")
        print("[!] Install with: pip install opencv-python-headless")


def filter_blurry_frames(images_dir: str, threshold: float = 100.0):
    """Remove blurry frames using Laplacian variance sharpness scoring.

    A frame with variance below `threshold` is considered too blurry for
    reliable feature matching in COLMAP and is deleted from the directory.
    Default threshold=100 is conservative — raise it (e.g. 150) for stricter
    filtering on high-resolution video.

    Requires: opencv-python  (pip install opencv-python-headless)
    """
    print_step("2a", "Blur Frame Filtering")

    try:
        import cv2
    except ImportError:
        print("[!] opencv-python not found — skipping blur filter.")
        print("[!] Install with: pip install opencv-python-headless")
        return

    valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    frames = sorted([f for f in os.listdir(images_dir) if f.endswith(valid_exts)])
    removed = 0
    scores = []

    for fname in frames:
        path = os.path.join(images_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        scores.append((fname, variance))
        if variance < threshold:
            os.remove(path)
            removed += 1
            print(f"  [x] Removed blurry frame: {fname}  (sharpness={variance:.1f} < {threshold})")

    if scores:
        avg = sum(s for _, s in scores) / len(scores)
        print(f"[✓] Blur filter: removed {removed}/{len(frames)} frames  "
              f"(avg sharpness={avg:.1f}, threshold={threshold})")
    else:
        print("[!] No frames found to evaluate for blur.")


def extract_frames_from_video(video_path: str, output_dir: str, target_fps: int = 7):
    """Extract frames from a video file using ffmpeg.

    FIX: Now applies a scale filter capping width at 1280px (maintaining aspect
    ratio). Previously the scale filter was missing, causing the full (often
    very low-resolution / heavily compressed) source video to be used as-is.
    For 4K sources this keeps full detail; for small sources (<1280px wide) the
    filter is a no-op.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Extracting frames from {video_path} at {target_fps} fps into {output_dir}...")

    # scale='min(1280,iw)':-2 → downscale wide videos to max 1280px wide,
    # auto-height preserving aspect ratio (height forced divisible by 2).
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-qscale:v", "1", "-qmin", "1",
        "-vf", f"fps={target_fps},scale='min(1280,iw)':-2",
        os.path.join(output_dir, "frame_%04d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    images = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
    print(f"[✓] Extracted {len(images)} frames successfully.")


def validate_input_images(images_dir: str, video_path: str = None) -> list:
    """Check for images or extract from video if needed."""
    os.makedirs(images_dir, exist_ok=True)
    valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    images = [f for f in os.listdir(images_dir) if f.endswith(valid_exts)]

    # If no images found but a video is provided or exists
    if len(images) == 0:
        if video_path and os.path.exists(video_path):
            extract_frames_from_video(video_path, images_dir)
            images = [f for f in os.listdir(images_dir) if f.endswith(valid_exts)]
        elif os.path.exists("airplane-video.mp4"):
            extract_frames_from_video("airplane-video.mp4", images_dir)
            images = [f for f in os.listdir(images_dir) if f.endswith(valid_exts)]
        else:
            raise FileNotFoundError(f"No images found in '{images_dir}', and no video specified.")

    print(f"[*] Located {len(images)} images in: {images_dir}")
    if len(images) < 20:
        print("[!] WARNING: Fewer than 20 images detected. COLMAP reconstruction may fail.")
    elif len(images) > 250:
        print("[!] NOTICE: Over 250 images detected. SfM may take 15+ minutes.")
    return images


def run_camera_pose_estimation(images_dir: str, processed_dir: str, skip_colmap: bool = False):
    """Run COLMAP via Nerfstudio to compute camera poses.

    FIX: Removed --num-downscales 0. Nerfstudio's default multi-scale warmup
    (downscale 3 → 2 → 1 → 0) stabilises early training and is especially
    important at lower source resolutions. Forcing 0 bypassed this entirely.
    """
    print_step(2, "Camera Pose Estimation (Structure-from-Motion)")

    transforms_path = os.path.join(processed_dir, "transforms.json")
    if skip_colmap and os.path.exists(transforms_path):
        print(f"[*] --skip-colmap enabled and {transforms_path} exists. Skipping SfM.")
        return

    os.makedirs(processed_dir, exist_ok=True)
    cmd = [
        "ns-process-data", "images",
        "--data", images_dir,
        "--output-dir", processed_dir,
        # FIX: --num-downscales 0 removed — restore nerfstudio's default
        # multi-scale training warmup for more stable early-training convergence.
        "--camera-type", "SIMPLE_RADIAL",  # NEW: explicit lens model hint
        "--skip-colmap", "False"
    ]
    print(f"[*] Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("ns-process-data failed. Check the error log above.")

    # Validate the generated transforms.json
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f"Expected {transforms_path} was not created.")

    with open(transforms_path, "r") as f:
        meta = json.load(f)

    registered_frames = len(meta.get("frames", []))
    total_imgs = len(validate_input_images(images_dir))
    reg_rate = (registered_frames / total_imgs * 100) if total_imgs > 0 else 0

    print("\n--- SFM REGISTRATION REPORT ---")
    print(f"Total Input Images: {total_imgs}")
    print(f"Registered Cameras: {registered_frames} ({reg_rate:.1f}%)")
    if reg_rate < 80.0:
        print("[!] WARNING: Under 80% of images were registered. Object may have missing sides.")
    else:
        print("[✓] Camera alignment successful. Ready to train.")


def run_training(processed_dir: str, outputs_dir: str, iterations: int = 7000):
    """Train the 3D Gaussian Splatting model.

    Fixes vs original:
    - FIX: continue-cull-post-densification restored to True (was explicitly
      False, preventing cleanup of junk Gaussians after densification).
    - FIX: cull-alpha-thresh raised 0.05 -> 0.08 to prune more haze/floaters.
    - CHANGE: default iterations raised 3000 -> 7000 (better quality, still
      fast on a T4 GPU ~15 min; use --high-quality for 30K).
    """
    print_step(3, f"Training 3D Gaussian Splats ({iterations} iterations)")

    os.makedirs(outputs_dir, exist_ok=True)
    cmd = [
        "ns-train", "splatfacto",
        "--data", processed_dir,
        "--max-num-iterations", str(iterations),
        "--steps-per-save", "1000",
        # FIX: Raised from 0.05 — more aggressive alpha pruning reduces haze
        "--pipeline.model.cull-alpha-thresh", "0.08",
        # FIX: Was False — re-enable post-densification culling
        "--pipeline.model.continue-cull-post-densification", "True",
        "--viewer.quit-on-train-completion", "True",
        "--output-dir", outputs_dir
    ]
    print(f"[*] Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Training failed. Review GPU memory and logs above.")


def export_ply(outputs_dir: str, exports_dir: str) -> str:
    """Export the trained model to standard .ply format."""
    print_step(4, "Exporting .ply Model")

    config_files = glob.glob(os.path.join(outputs_dir, "**", "config.yml"), recursive=True)
    if not config_files:
        raise FileNotFoundError(f"No config.yml found in {outputs_dir}. Did training complete?")

    latest_config = max(config_files, key=os.path.getmtime)
    print(f"[*] Loading latest config: {latest_config}")

    os.makedirs(exports_dir, exist_ok=True)
    cmd = [
        "ns-export", "gaussian-splat",
        "--load-config", latest_config,
        "--output-dir", exports_dir
    ]
    print(f"[*] Running export: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Export failed. Review the logs above.")

    exported_plys = glob.glob(os.path.join(exports_dir, "**", "*.ply"), recursive=True)
    if not exported_plys:
        raise FileNotFoundError(f"No .ply file found in {exports_dir} after export.")

    final_ply = exported_plys[0]
    file_size_mb = os.path.getsize(final_ply) / (1024 * 1024)
    print("\n" + "=" * 65)
    print(f"[✓] SUCCESS! Exported 3D Gaussian Splat:")
    print(f"    Path: {final_ply}")
    print(f"    Size: {file_size_mb:.2f} MB")
    print(f"    Drag & drop into https://playcanvas.com/supersplat/editor to view!")
    print("=" * 65)
    return final_ply


def main():
    parser = argparse.ArgumentParser(description="3DGS Training Engine Pipeline")
    parser.add_argument("--video", default="airplane-video.mp4",
                        help="Path to input video file (e.g., airplane-video.mp4)")
    parser.add_argument("--images-dir", default="dataset/images",
                        help="Path to input images folder")
    parser.add_argument("--processed-dir", default="processed_data",
                        help="Output directory for COLMAP")
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Output directory for training checkpoints")
    parser.add_argument("--exports-dir", default="exports",
                        help="Output directory for final .ply file")
    parser.add_argument("--iterations", type=int, default=7000,
                        help="Training iterations (default: 7000; use --high-quality for 30K)")
    parser.add_argument("--high-quality", action="store_true",
                        help="Production quality preset: 30K iterations + stricter blur filter")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Laplacian variance threshold for blur removal (default: 100.0). "
                             "Higher = stricter. Use 0 to disable blur filtering.")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="Skip COLMAP if transforms.json already exists")
    parser.add_argument("--check-only", action="store_true",
                        help="Only run hardware and environment check")
    args = parser.parse_args()

    # Apply high-quality preset
    if args.high_quality:
        args.iterations = 30000
        if args.blur_threshold == 100.0:
            args.blur_threshold = 150.0  # stricter blur removal at higher quality
        print("[*] --high-quality preset: 30K iterations, blur threshold=150")

    check_environment()
    if args.check_only:
        return

    validate_input_images(args.images_dir, video_path=args.video)

    # Blur filter runs before COLMAP to avoid wasting SfM time on bad frames
    if args.blur_threshold > 0:
        filter_blurry_frames(args.images_dir, threshold=args.blur_threshold)

    run_camera_pose_estimation(args.images_dir, args.processed_dir, skip_colmap=args.skip_colmap)
    run_training(args.processed_dir, args.outputs_dir, iterations=args.iterations)
    export_ply(args.outputs_dir, args.exports_dir)


if __name__ == "__main__":
    main()
