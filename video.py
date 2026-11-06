#!/usr/bin/env python3
"""
ACM AI - Video Extraction & Dataset Ingestion
=============================================
Provides:
1. extract(IN, OUT): Streams and extracts a shoe video from the uCO3D Hugging Face dataset via RemoteZip.
2. extract_frames(n_frames, maxs, video, IMAGES, quality_filter=False):
   Extracts frames with resolution downscaling (max side <= maxs) and optional Laplacian blur filtering.
"""

from pathlib import Path
import logging
import cv2

try:
    from remotezip import RemoteZip
except ImportError:
    RemoteZip = None

logger = logging.getLogger("VideoIngestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def extract(IN: str, OUT: str) -> Path:
    """
    Downloads and extracts a shoe video from the uCO3D Hugging Face dataset zip file.
    Uses RemoteZip to only download the selected video file without downloading the full multi-GB archive.
    """
    out_dir = Path(OUT)
    existing_videos = list(out_dir.rglob("rgb_video.mp4"))
    if existing_videos:
        logger.info(f"Using existing video found in {out_dir}: {existing_videos[0]}")
        return existing_videos[0]

    if RemoteZip is None:
        raise ImportError("remotezip is required to stream from Hugging Face. Run: pip install remotezip")

    logger.info(f"Connecting to remote dataset archive: {IN} ...")
    with RemoteZip(IN) as z:
        vids = [i for i in z.infolist() if i.filename.endswith("rgb_video.mp4")]
        shoes = [i for i in vids if any(k in i.filename.lower() for k in ("sneaker", "shoe", "boot"))] or vids
        shoes.sort(key=lambda i: i.file_size)
        pick = shoes[len(shoes) // 2]  # Median size sample
        logger.info(f"Selected remote video: {pick.filename} ({pick.file_size / 1e6:.1f} MB)")
        z.extract(pick, OUT)

    found = list(out_dir.rglob("rgb_video.mp4"))
    if not found:
        raise FileNotFoundError(f"Could not find extracted rgb_video.mp4 in {OUT}")
    return found[0]


def extract_frames(n_frames: int, maxs: int, video, IMAGES, quality_filter: bool = False) -> int:
    """
    Extracts frames from video into IMAGES folder with longest-edge normalization.
    If quality_filter=True, invokes the Group 1 pipeline with Laplacian blur detection
    and smart neighbor replacement.
    """
    IMAGES = Path(IMAGES)
    IMAGES.mkdir(parents=True, exist_ok=True)
    video_path = Path(video).resolve()

    if quality_filter:
        from process_video import extract_and_filter_frames
        summary = extract_and_filter_frames(
            video_path=str(video_path),
            output_dir=IMAGES,
            target_count=n_frames,
            max_dimension=maxs,
            trim_start=1.5,
            trim_end=1.5
        )
        return summary["stats"]["final_extracted_count"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1
        while cap.grab():
            total += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    n_frames = min(max(1, n_frames), total)
    if total > 1 and n_frames > 1:
        keep = {round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)}
    else:
        keep = {0}

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in keep:
            h, w = frame.shape[:2]
            longest = max(h, w)
            if longest > maxs:
                s = maxs / float(longest)
                new_w = round(w * s)
                new_h = round(h * s)
                if new_w % 2 != 0: new_w -= 1
                if new_h % 2 != 0: new_h -= 1
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            out_file = IMAGES / f"frame_{saved+1:04d}.jpg"
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        idx += 1

    cap.release()
    logger.info(f"Extracted and saved {saved} frames into {IMAGES}")
    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract video frames or stream from uCO3D")
    parser.add_argument("-i", "--input", default=None, help="Input video file or remote zip url")
    parser.add_argument("-o", "--output", default="ws/images", help="Output directory for frames (default: ws/images)")
    parser.add_argument("-n", "--count", type=int, default=100, help="Number of frames (default: 100)")
    parser.add_argument("--max-side", type=int, default=1280, help="Max edge in pixels (default: 1280)")
    parser.add_argument("--quality-filter", action="store_true", help="Enable blur filtering and smart replacement")
    args = parser.parse_args()

    if args.input and Path(args.input).exists():
        extract_frames(args.count, args.max_side, args.input, args.output, quality_filter=args.quality_filter)
    else:
        url = args.input or "https://huggingface.co/datasets/facebook/uco3d/resolve/main/part_rgb_videos_0232.zip"
        vid = extract(url, "uco3d_shoe")
        extract_frames(args.count, args.max_side, vid, args.output, quality_filter=args.quality_filter)

