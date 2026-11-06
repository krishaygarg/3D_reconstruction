#!/usr/bin/env python3
"""
ACM AI - 3D Reconstruction Data Pipeline (Group 1)
=================================================
Automated video frame extraction, blur filtering, resolution normalization,
and handoff archive packaging.

Key Requirements addressed:
1. Trims first and last 1-2 seconds (hand shake elimination).
2. Frame sampling targeting 80-130 frames (or consistent 2-3 fps).
3. Blur detection via Laplacian variance with smart neighbor replacement.
4. Resolution normalization: longest edge downscaled to exactly 1280px (aspect ratio preserved).
5. Packaging into a compressed zip archive for Group 2 handoff.
"""

import os
import sys
import math
import json
import shutil
import zipfile
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CapturePipeline")


def calculate_sharpness(image_bgr: np.ndarray) -> float:
    """
    Computes the sharpness score of an image using the variance of the Laplacian.
    Higher values indicate sharp edges, while low values indicate motion blur
    or autofocus hunting.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def resize_longest_edge(image_bgr: np.ndarray, target_longest_edge: int = 1280) -> np.ndarray:
    """
    Downscales image so that its longest edge is target_longest_edge pixels,
    strictly preserving aspect ratio. Uses INTER_AREA for high-quality downsampling.
    Guarantees even dimensions (divisible by 2) for downstream CV pipelines.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    
    if longest <= target_longest_edge:
        # If image is already smaller or equal, ensure even dimensions without rescaling
        if height % 2 == 0 and width % 2 == 0:
            return image_bgr
        even_h = height - (height % 2)
        even_w = width - (width % 2)
        return image_bgr[:even_h, :even_w]

    scale = target_longest_edge / float(longest)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    # Ensure even dimensions (required for video codecs, PyTorch tensors, and COLMAP)
    if new_width % 2 != 0:
        new_width -= 1
    if new_height % 2 != 0:
        new_height -= 1

    new_width = max(2, new_width)
    new_height = max(2, new_height)

    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def enhance_frame(
    image_bgr: np.ndarray,
    denoise_strength: float = 3.0,
    sharpen_amount: float = 0.8,
    sharpen_sigma: float = 1.2,
    clahe_clip: float = 1.5,
    use_nlmeans: bool = False
) -> np.ndarray:
    """
    Cleans up a frame to make it clearer for reconstruction/training:
    1. Edge-preserving denoising (Bilateral filter by default for high speed and
       pristine feature preservation, or Non-Local Means when use_nlmeans=True).
    2. Local contrast boost (CLAHE on luminance only, so colors stay stable).
    3. Unsharp-mask sharpening to restore edge definition.
    Set any strength to 0 to disable that step.
    """
    out = image_bgr

    if denoise_strength > 0:
        if use_nlmeans:
            out = cv2.fastNlMeansDenoisingColored(
                out, None, denoise_strength, denoise_strength, 7, 21
            )
        else:
            d = 5
            sigma_color = max(10, min(100, int(denoise_strength * 12)))
            sigma_space = max(10, min(100, int(denoise_strength * 12)))
            out = cv2.bilateralFilter(out, d, sigma_color, sigma_space)

    if clahe_clip > 0:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        out = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), sharpen_sigma)
        out = cv2.addWeighted(out, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)

    return out


def get_video_info(cap: cv2.VideoCapture) -> Dict:
    """Extracts fundamental video properties with robust fallbacks for arbitrary video containers."""
    def _safe_float(val, default=0.0):
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else f
        except (ValueError, TypeError):
            return default

    def _safe_int(val, default=0):
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else int(f)
        except (ValueError, TypeError):
            return default

    total_frames = _safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS))
    width = _safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = _safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0:
        fps = 30.0  # Safe default if metadata is missing or variable
        
    if total_frames <= 0:
        logger.info("Video metadata did not report frame count. Probing stream frames...")
        count = 0
        while cap.grab():
            count += 1
        total_frames = max(1, count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info(f"Stream probe completed: found {total_frames} total frames.")

    if width <= 0 or height <= 0:
        ret, frame = cap.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            width = max(2, width)
            height = max(2, height)
        
    duration = total_frames / fps if fps > 0 else 0.0
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration
    }


def compute_sample_indices(
    total_frames: int,
    fps: float,
    trim_start_sec: float = 1.5,
    trim_end_sec: float = 1.5,
    target_count: Optional[int] = 100,
    sample_fps: Optional[float] = None,
    all_frames: bool = False
) -> Tuple[List[int], Dict]:
    """
    Computes frame indices to extract based on shake trimming and sampling parameters.
    Handles any video duration gracefully, including very short or long clips.
    """
    duration = total_frames / fps if fps > 0 else 0.0
    
    # Adaptively adjust trimming if video is shorter than the requested trim duration
    eff_trim_start = trim_start_sec
    eff_trim_end = trim_end_sec
    
    if (trim_start_sec + trim_end_sec) >= duration:
        if duration > 1.0:
            eff_trim_start = min(0.5, duration * 0.1)
            eff_trim_end = min(0.5, duration * 0.1)
            logger.info(
                f"Video duration ({duration:.2f}s) is short. "
                f"Auto-adjusting trimming to {eff_trim_start:.2f}s start / {eff_trim_end:.2f}s end."
            )
        else:
            eff_trim_start = 0.0
            eff_trim_end = 0.0
            logger.info(f"Video duration ({duration:.2f}s) is under 1 second. Shake trimming disabled.")

    start_frame = int(math.ceil(eff_trim_start * fps))
    end_frame = int(math.floor(total_frames - (eff_trim_end * fps)))
    
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))

    usable_frames = max(1, (end_frame - start_frame) + 1)
    usable_duration = usable_frames / fps

    logger.info(
        f"Raw video: {total_frames} frames ({duration:.2f}s). "
        f"Usable range: frames [{start_frame}..{end_frame}] ({usable_duration:.2f}s, {usable_frames} frames)."
    )

    if all_frames:
        indices = list(range(start_frame, end_frame + 1))
        logger.info(f"All-frames mode: selected all {len(indices)} frames.")
    elif sample_fps is not None and sample_fps > 0:
        # Sample at fixed FPS
        step = fps / sample_fps
        indices = [int(round(start_frame + i * step)) for i in range(max(1, int(round(usable_duration * sample_fps))))]
        indices = [idx for idx in indices if start_frame <= idx <= end_frame]
        logger.info(f"Fixed sampling rate of {sample_fps} FPS selected {len(indices)} candidate frames.")
    else:
        # Target count mode: target 80-130 frames by default, or user-specified count
        count = target_count if target_count is not None else 100
        
        # If user explicitly provided a count, respect it; if default, clamp only when sufficient frames exist
        if count > usable_frames:
            logger.info(f"Target count {count} exceeds usable frames ({usable_frames}). Extracting all {usable_frames} frames.")
            count = usable_frames

        if count >= usable_frames:
            indices = list(range(start_frame, end_frame + 1))
        else:
            step = usable_frames / float(count)
            indices = [int(round(start_frame + (i + 0.5) * step)) for i in range(count)]
            indices = [min(idx, end_frame) for idx in indices]
        
        logger.info(f"Selected {len(indices)} evenly-spaced candidate frames (effective rate: {len(indices)/max(usable_duration, 0.001):.2f} fps).")

    # Ensure uniqueness and sorted order
    indices = sorted(list(dict.fromkeys(indices)))
    if not indices:
        indices = [start_frame]

    meta = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "usable_frames": usable_frames,
        "usable_duration_sec": usable_duration,
        "initial_candidates": len(indices)
    }
    return indices, meta


def extract_and_filter_frames(
    video_path: str,
    output_dir: Path,
    trim_start: float = 1.5,
    trim_end: float = 1.5,
    target_count: int = 100,
    sample_fps: Optional[float] = None,
    max_dimension: int = 1280,
    jpeg_quality: int = 95,
    blur_threshold: Optional[float] = None,
    smart_replace_window: int = 5,
    enhance: bool = True,
    denoise_strength: float = 3.0,
    sharpen_amount: float = 0.8,
    clahe_clip: float = 1.5,
    all_frames: bool = False,
    use_nlmeans: bool = False
) -> Dict:
    """
    Main extraction pipeline:
    - Reads video.
    - Samples frames.
    - Detects blur and replaces with sharpest nearby candidate if available.
    - Resizes longest edge to 1280px.
    - Saves high quality JPEGs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    # Automatically respect orientation tags on phone videos if supported
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass

    video_info = get_video_info(cap)
    logger.info(
        f"Loaded video '{Path(video_path).name}': "
        f"{video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f} FPS, "
        f"Duration: {video_info['duration_sec']:.2f}s ({video_info['total_frames']} frames)"
    )

    sample_indices, sample_meta = compute_sample_indices(
        total_frames=video_info["total_frames"],
        fps=video_info["fps"],
        trim_start_sec=trim_start,
        trim_end_sec=trim_end,
        target_count=target_count,
        sample_fps=sample_fps,
        all_frames=all_frames
    )

    # First pass: read candidate frames and calculate initial sharpness distribution
    logger.info("Analyzing candidate frames and calculating sharpness metrics...")
    candidates = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        score = calculate_sharpness(frame)
        # Store normalized frame directly to minimize memory usage for high-resolution 4K video
        normalized = resize_longest_edge(frame, target_longest_edge=max_dimension)
        candidates.append({
            "target_idx": idx,
            "chosen_idx": idx,
            "score": score,
            "frame": normalized,
            "replaced": False
        })

    if not candidates:
        cap.release()
        raise RuntimeError("No frames could be successfully read from the video.")

    scores = [c["score"] for c in candidates]
    median_score = float(np.median(scores))
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    logger.info(f"Sharpness distribution - Median: {median_score:.1f}, Mean: {mean_score:.1f}, Std: {std_score:.1f}")

    # Determine adaptive blur threshold if not explicitly set
    # A frame is considered abnormally blurry if its Laplacian variance is below
    # 50% of the median sharpness or below (mean - 1.5 * std).
    if blur_threshold is None or blur_threshold <= 0:
        adaptive_threshold = max(20.0, min(median_score * 0.55, mean_score - 1.2 * std_score))
    else:
        adaptive_threshold = blur_threshold

    logger.info(f"Blur rejection threshold set to: {adaptive_threshold:.2f}")

    # Second pass: blur filtering and smart replacement
    saved_frames = []
    blurry_count = 0
    replaced_count = 0
    dropped_count = 0
    fps_safe = video_info["fps"] if video_info["fps"] > 0 else 30.0

    for i, item in enumerate(candidates):
        frame = item["frame"]
        score = item["score"]
        chosen_idx = item["target_idx"]
        replaced = False

        if score < adaptive_threshold:
            blurry_count += 1
            logger.warning(
                f"Frame candidate {i+1}/{len(candidates)} (frame #{chosen_idx}) has low sharpness ({score:.1f} < {adaptive_threshold:.1f}). "
                f"Searching nearby window (+/- {smart_replace_window} frames) for sharp alternative..."
            )

            # Smart neighbor search within +/- smart_replace_window frames
            best_sub_frame = frame
            best_sub_score = score
            best_sub_idx = chosen_idx

            window_start = max(sample_meta["start_frame"], chosen_idx - smart_replace_window)
            window_end = min(sample_meta["end_frame"], chosen_idx + smart_replace_window)

            for offset_idx in range(window_start, window_end + 1):
                if offset_idx == chosen_idx:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, offset_idx)
                ret_sub, cand_frame = cap.read()
                if ret_sub and cand_frame is not None:
                    cand_score = calculate_sharpness(cand_frame)
                    if cand_score > best_sub_score:
                        best_sub_score = cand_score
                        best_sub_frame = resize_longest_edge(cand_frame, target_longest_edge=max_dimension)
                        best_sub_idx = offset_idx

            if best_sub_score >= adaptive_threshold:
                logger.info(
                    f"-> Successfully replaced with sharper neighbor frame #{best_sub_idx} (score: {best_sub_score:.1f})."
                )
                frame = best_sub_frame
                score = best_sub_score
                chosen_idx = best_sub_idx
                replaced = True
                replaced_count += 1
            else:
                # If still under threshold, decide whether to drop or keep
                min_threshold = min(80, int(len(candidates) * 0.8))
                projected_count = len(candidates) - (dropped_count + 1)
                if projected_count >= min_threshold:
                    logger.warning(f"-> Dropping blurry frame #{chosen_idx} (best score was {best_sub_score:.1f}).")
                    dropped_count += 1
                    continue
                else:
                    if best_sub_idx != chosen_idx:
                        logger.warning(
                            f"-> Replaced with best available neighbor frame #{best_sub_idx} (score {best_sub_score:.1f}) to preserve 360-degree orbit coverage."
                        )
                        frame = best_sub_frame
                        score = best_sub_score
                        chosen_idx = best_sub_idx
                        replaced = True
                        replaced_count += 1
                    else:
                        logger.warning(
                            f"-> Preserving frame #{chosen_idx} (score {score:.1f}) to preserve 360-degree orbit coverage."
                        )

        # Resolution Normalization: frame is already scaled to max_dimension
        processed_frame = frame

        # Clean-up: denoise + contrast + sharpen (after downscale so detail isn't wasted)
        if enhance:
            processed_frame = enhance_frame(
                processed_frame,
                denoise_strength=denoise_strength,
                sharpen_amount=sharpen_amount,
                clahe_clip=clahe_clip,
                use_nlmeans=use_nlmeans
            )
        
        # Save frame as JPEG with 4-digit zero-padded index
        frame_filename = f"frame_{len(saved_frames)+1:04d}.jpg"
        frame_path = output_dir / frame_filename

        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality),
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
        ]
        cv2.imwrite(str(frame_path), processed_frame, encode_params)

        saved_frames.append({
            "filename": frame_filename,
            "original_frame_idx": chosen_idx,
            "timestamp_sec": round(chosen_idx / fps_safe, 3),
            "sharpness": round(score, 2),
            "replaced": replaced,
            "width": processed_frame.shape[1],
            "height": processed_frame.shape[0]
        })

    cap.release()

    final_count = len(saved_frames)
    final_res = f"{saved_frames[0]['width']}x{saved_frames[0]['height']}" if saved_frames else "N/A"
    longest_actual = max(saved_frames[0]['width'], saved_frames[0]['height']) if saved_frames else 0
    
    logger.info(
        f"Processing complete! Extracted {final_count} frames into '{output_dir}'. "
        f"Dimensions: {final_res} (Longest edge = {longest_actual}px, target: {max_dimension}px). "
        f"Quality checks: {blurry_count} blurry candidates detected, {replaced_count} replaced, {dropped_count} dropped."
    )

    # Validate deliverable target
    target_met = (80 <= final_count <= 130)
    if target_met:
        logger.info(f"Target count requirement met: {final_count} frames is within [80, 130].")
    else:
        logger.warning(f"Final count ({final_count}) is outside target [80, 130]! Check input video duration.")

    summary = {
        "video_info": video_info,
        "sample_meta": sample_meta,
        "config": {
            "trim_start_sec": trim_start,
            "trim_end_sec": trim_end,
            "target_count": target_count,
            "sample_fps": sample_fps,
            "max_dimension": max_dimension,
            "jpeg_quality": jpeg_quality,
            "enhance": enhance,
            "denoise_strength": denoise_strength,
            "sharpen_amount": sharpen_amount,
            "clahe_clip": clahe_clip,
            "adaptive_threshold": round(adaptive_threshold, 2)
        },
        "stats": {
            "initial_candidates": len(candidates),
            "blurry_detected": blurry_count,
            "smart_replaced": replaced_count,
            "dropped": dropped_count,
            "final_extracted_count": final_count,
            "target_range": [80, 130],
            "target_met": target_met,
            "final_resolution": final_res,
            "mean_sharpness": round(float(np.mean([f["sharpness"] for f in saved_frames])), 2) if saved_frames else 0.0,
            "min_sharpness": round(float(np.min([f["sharpness"] for f in saved_frames])), 2) if saved_frames else 0.0,
            "max_sharpness": round(float(np.max([f["sharpness"] for f in saved_frames])), 2) if saved_frames else 0.0
        },
        "frames": saved_frames
    }

    # Save metadata JSON into the output folder
    meta_path = output_dir / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def package_archive(output_dir: Path, archive_path: Path) -> Path:
    """
    Compresses the final images and metadata into a zip file for Group 2 handoff.
    """
    output_dir = Path(output_dir).resolve()
    archive_path = Path(archive_path).resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    logger.info(f"Packaging {output_dir} into handoff archive '{archive_path}'...")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in sorted(output_dir.iterdir()):
            if file.is_file() and not file.name.startswith(".") and file.resolve() != archive_path:
                zipf.write(file, arcname=file.name)

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    logger.info(f"Handoff archive successfully created: {archive_path} ({size_mb:.2f} MB)")
    return archive_path


def main():
    parser = argparse.ArgumentParser(
        description="ACM AI Group 1 - Video Capture, Frame Sampling, Blur Filtering & Normalization Pipeline"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input raw video file (e.g. 4K 60fps MP4/MOV)")
    parser.add_argument("-o", "--output", default="extracted_frames", help="Output directory for processed JPEG frames (default: extracted_frames)")
    parser.add_argument("-a", "--archive", default="group2_handoff.zip", help="Path for compressed handoff zip file (default: group2_handoff.zip)")
    parser.add_argument("-n", "--count", type=int, default=100, help="Target frame count within [80, 130] (default: 100)")
    parser.add_argument("--fps", type=float, default=None, help="Fixed sampling rate in frames per second (e.g. 2.5). If not set, auto-targets --count.")
    parser.add_argument("--all", "--all-frames", dest="all_frames", action="store_true", help="Extract every single frame from usable duration.")
    parser.add_argument("--trim-start", type=float, default=1.5, help="Seconds to trim from start to eliminate camera shake (default: 1.5s)")
    parser.add_argument("--trim-end", type=float, default=1.5, help="Seconds to trim from end to eliminate stop shake (default: 1.5s)")
    parser.add_argument("--max-dim", type=int, default=1280, help="Longest edge resolution in pixels (default: 1280)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--blur-threshold", type=float, default=None, help="Explicit Laplacian blur threshold. If not set, computed adaptively.")
    parser.add_argument("--no-enhance", action="store_true", help="Disable denoise/contrast/sharpen clean-up of frames.")
    parser.add_argument("--denoise", type=float, default=3.0, help="Denoise strength, 0 disables (default: 3.0)")
    parser.add_argument("--nlmeans", action="store_true", help="Use Non-Local Means denoising instead of fast edge-preserving bilateral filter.")
    parser.add_argument("--sharpen", type=float, default=0.8, help="Unsharp-mask amount, 0 disables (default: 0.8)")
    parser.add_argument("--clahe", type=float, default=1.5, help="CLAHE contrast clip limit, 0 disables (default: 1.5)")
    parser.add_argument("--no-archive", action="store_true", help="Skip creating the handoff zip archive.")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error(f"Input video file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    archive_path = Path(args.archive).resolve()

    try:
        summary = extract_and_filter_frames(
            video_path=str(input_path),
            output_dir=output_dir,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            target_count=args.count,
            sample_fps=args.fps,
            max_dimension=args.max_dim,
            jpeg_quality=args.quality,
            blur_threshold=args.blur_threshold,
            enhance=not args.no_enhance,
            denoise_strength=args.denoise,
            sharpen_amount=args.sharpen,
            clahe_clip=args.clahe,
            all_frames=args.all_frames,
            use_nlmeans=args.nlmeans
        )

        if not args.no_archive:
            package_archive(output_dir, archive_path)

        print("\n" + "="*60)
        print("          GROUP 1 DELIVERABLE READY FOR HANDOFF")
        print("="*60)
        print(f" Source Video:      {input_path.name}")
        print(f" Final Frame Count: {summary['stats']['final_extracted_count']} frames (Target: 80-130)")
        print(f" Resolution:        {summary['stats']['final_resolution']} (max edge 1280px)")
        print(f" Mean Sharpness:    {summary['stats']['mean_sharpness']}")
        print(f" Output Folder:     {output_dir}")
        if not args.no_archive:
            print(f" Handoff Archive:   {archive_path}")
        print("="*60 + "\n")

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
