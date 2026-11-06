#!/usr/bin/env python3
"""
Generate a synthetic 360-degree orbit sample video for testing the Group 1 pipeline.
Simulates:
- 360-degree rotation around a textured object with surface details.
- Hand shake during the first 1.5s and last 1.5s.
- High resolution (e.g. 4K or 1080p for fast testing) at 60 FPS.
- A simulated autofocus hunting / motion blur spike in the middle to verify blur filtering.
"""

import sys
import math
import argparse
import numpy as np
import cv2
from pathlib import Path

def create_synthetic_orbit_video(
    output_path: str = "sample_orbit_4k.mp4",
    duration_sec: float = 18.0,
    fps: int = 60,
    width: int = 3840,
    height: int = 2160,
    fast_mode: bool = False
):
    if fast_mode:
        width = 1920
        height = 1080
        fps = 30
        print(f"Fast test mode enabled: generating {width}x{height} @ {fps}fps...")

    total_frames = int(duration_sec * fps)
    print(f"Generating synthetic orbit video '{output_path}': {width}x{height}, {fps} FPS, {duration_sec}s ({total_frames} frames)...")

    # FourCC codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError("Failed to initialize video writer.")

    # Pre-generate a high-detail procedural texture
    tex_size = 512
    texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    
    # Checkerboard and fractal-like geometric textures
    for y in range(tex_size):
        for x in range(tex_size):
            c1 = ((x // 32) + (y // 32)) % 2 == 0
            c2 = ((x // 8) ^ (y // 8)) % 3 == 0
            r = 180 if c1 else 40
            g = 120 if c2 else 70
            b = 220 if (x * y) % 17 == 0 else 100
            texture[y, x] = [b, g, r]

    # Draw high-frequency text and circles for photogrammetry feature points
    cv2.putText(texture, "ACM AI 3D", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    cv2.putText(texture, "TEXTURE", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
    for i in range(15):
        cx = int((i * 37) % tex_size)
        cy = int((i * 61) % tex_size)
        cv2.circle(texture, (cx, cy), 12, (50, 255, 100), -1)

    cx_screen, cy_screen = width // 2, height // 2
    base_radius = min(width, height) * 0.28

    for f in range(total_frames):
        t = f / fps
        progress = f / float(total_frames)
        angle = progress * 2.0 * math.pi  # 360-degree rotation

        # Simulate hand shake in first 1.5s and last 1.5s
        shake_x = 0.0
        shake_y = 0.0
        if t < 1.5:
            shake_intensity = (1.5 - t) / 1.5 * 25.0
            shake_x = math.sin(f * 0.9) * shake_intensity
            shake_y = math.cos(f * 1.3) * shake_intensity
        elif t > (duration_sec - 1.5):
            shake_intensity = (t - (duration_sec - 1.5)) / 1.5 * 25.0
            shake_x = math.sin(f * 0.8) * shake_intensity
            shake_y = math.cos(f * 1.1) * shake_intensity

        # Create blank canvas with subtle studio gradient
        frame = np.full((height, width, 3), 28, dtype=np.uint8)
        
        # Subtle studio floor grid
        floor_y = int(cy_screen + base_radius * 1.1)
        cv2.line(frame, (0, floor_y), (width, floor_y), (60, 60, 60), 2)

        # Draw 3D-shaded orbiting cylinder/object
        obj_x = int(cx_screen + shake_x)
        obj_y = int(cy_screen + shake_y)

        # Rotate texture slice
        tex_shift = int((angle / (2.0 * math.pi)) * tex_size) % tex_size
        rolled_tex = np.roll(texture, tex_shift, axis=1)

        # Warp into an elliptical perspective
        obj_w = int(base_radius * 1.3)
        obj_h = int(base_radius * 1.6)
        scaled_tex = cv2.resize(rolled_tex, (obj_w, obj_h), interpolation=cv2.INTER_LINEAR)

        # Shading gradient across object to simulate lighting
        shading = np.linspace(0.4, 1.2, obj_w).reshape(1, -1, 1)
        shaded_tex = np.clip(scaled_tex.astype(np.float32) * shading, 0, 255).astype(np.uint8)

        # Place onto frame with boundary clipping
        x1 = obj_x - obj_w // 2
        y1 = obj_y - obj_h // 2
        x2 = x1 + obj_w
        y2 = y1 + obj_h

        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(width, x2)
        y2_clip = min(height, y2)

        if x1_clip < x2_clip and y1_clip < y2_clip:
            tex_x1 = x1_clip - x1
            tex_y1 = y1_clip - y1
            tex_x2 = tex_x1 + (x2_clip - x1_clip)
            tex_y2 = tex_y1 + (y2_clip - y1_clip)
            frame[y1_clip:y2_clip, x1_clip:x2_clip] = shaded_tex[tex_y1:tex_y2, tex_x1:tex_x2]

        # Overlay reference metadata
        deg = int(math.degrees(angle)) % 360
        status_text = f"Simulated Orbit: {deg:03d} deg | Time: {t:.2f}s / {duration_sec:.1f}s"
        cv2.putText(frame, status_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2 if width > 2000 else 0.7, (200, 200, 200), 2)

        # Simulate a momentary blur event (e.g. 3 frames around t=8.5s)
        if 8.45 <= t <= 8.55:
            # Artificial motion blur kernel
            ksize = 25 if width > 2000 else 15
            kernel = np.zeros((ksize, ksize))
            kernel[int((ksize-1)/2), :] = np.ones(ksize)
            kernel /= ksize
            frame = cv2.filter2D(frame, -1, kernel)

        out.write(frame)

        if f % (fps * 3) == 0 or f == total_frames - 1:
            print(f"  Rendered {f+1}/{total_frames} frames ({progress*100:.1f}%)...")

    out.release()
    print(f"Generated sample video successfully: {output_path} ({Path(output_path).stat().st_size / (1024*1024):.1f} MB)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic 360 orbit sample video.")
    parser.add_argument("-o", "--output", default="sample_orbit.mp4", help="Output video path")
    parser.add_argument("-d", "--duration", type=float, default=18.0, help="Duration in seconds (default: 18.0)")
    parser.add_argument("--fps", type=int, default=60, help="Framerate (default: 60)")
    parser.add_argument("--fast", action="store_true", help="Generate 1080p @ 30fps for ultra-fast testing")
    args = parser.parse_args()

    create_synthetic_orbit_video(
        output_path=args.output,
        duration_sec=args.duration,
        fps=args.fps,
        width=3840 if not args.fast else 1920,
        height=2160 if not args.fast else 1080,
        fast_mode=args.fast
    )
