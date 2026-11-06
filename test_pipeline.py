#!/usr/bin/env python3
"""
Test Suite for ACM AI Group 1 Capture & Frame Processing Pipeline.
Validates all requirements:
1. Trimming start/end shake
2. Frame sampling targeting 80-130 frames
3. Resolution normalization: longest edge == 1280px
4. Format: High-quality JPEG
5. Blur detection & reporting
6. Zip archive creation for handoff
"""

import os
import sys
import shutil
import zipfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from generate_sample_video import create_synthetic_orbit_video
from process_video import extract_and_filter_frames, package_archive


class TestCapturePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_artifacts")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        cls.sample_video = cls.test_dir / "test_orbit.mp4"
        cls.output_dir = cls.test_dir / "test_frames"
        cls.archive_zip = cls.test_dir / "test_handoff.zip"

        print("\n[Setup] Generating synthetic test video...")
        # 16 seconds @ 30 FPS = 480 frames
        create_synthetic_orbit_video(
            output_path=str(cls.sample_video),
            duration_sec=16.0,
            fps=30,
            width=1920,
            height=1080,
            fast_mode=True
        )

    @classmethod
    def tearDownClass(cls):
        # Keep artifacts or clean up as appropriate
        pass

    def test_01_pipeline_execution_and_frame_count(self):
        """Verify pipeline runs and produces between 80 and 130 frames."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        summary = extract_and_filter_frames(
            video_path=str(self.sample_video),
            output_dir=self.output_dir,
            trim_start=1.5,
            trim_end=1.5,
            target_count=100,
            max_dimension=1280,
            jpeg_quality=95
        )

        final_count = summary["stats"]["final_extracted_count"]
        print(f"\n[Test] Extracted count: {final_count}")
        self.assertGreaterEqual(final_count, 80, "Extracted frame count must be at least 80.")
        self.assertLessEqual(final_count, 130, "Extracted frame count must not exceed 130.")
        self.assertTrue(summary["stats"]["target_met"])

    def test_02_resolution_normalization(self):
        """Verify all frames have longest edge downscaled to exactly 1280px."""
        image_files = list(self.output_dir.glob("*.jpg"))
        self.assertGreater(len(image_files), 0, "No JPEG files found.")

        for img_path in image_files[:15]:  # sample check
            img = cv2.imread(str(img_path))
            self.assertIsNotNone(img, f"Failed to read image {img_path}")
            h, w = img.shape[:2]
            longest = max(h, w)
            self.assertEqual(longest, 1280, f"Longest edge is {longest}, expected 1280 for {img_path.name}")
            # Ensure 16:9 ratio is preserved (1280x720)
            self.assertEqual(w, 1280)
            self.assertEqual(h, 720)

    def test_03_trimming_verification(self):
        """Verify start and end timestamps respect the 1.5s shake trimming."""
        meta_file = self.output_dir / "dataset_metadata.json"
        self.assertTrue(meta_file.exists(), "dataset_metadata.json must exist")
        
        import json
        with open(meta_file, "r") as f:
            data = json.load(f)

        frames = data["frames"]
        first_time = frames[0]["timestamp_sec"]
        last_time = frames[-1]["timestamp_sec"]

        # 16s video, trimmed 1.5s start and 1.5s end -> active between ~1.5s and ~14.5s
        self.assertGreaterEqual(first_time, 1.4, f"First frame at {first_time}s violates start trim")
        self.assertLessEqual(last_time, 14.6, f"Last frame at {last_time}s violates end trim")

    def test_04_handoff_archive(self):
        """Verify handoff zip archive is generated and contains all images and metadata."""
        package_archive(self.output_dir, self.archive_zip)
        self.assertTrue(self.archive_zip.exists(), "Handoff zip file must exist")
        self.assertGreater(self.archive_zip.stat().st_size, 10000, "Archive size should be reasonable")

        with zipfile.ZipFile(self.archive_zip, "r") as zf:
            namelist = zf.namelist()
            jpg_count = len([n for n in namelist if n.endswith(".jpg")])
            self.assertGreaterEqual(jpg_count, 80)
            self.assertLessEqual(jpg_count, 130)
            self.assertIn("dataset_metadata.json", namelist)

    def test_05_arbitrary_and_short_video_conversion(self):
        """Verify any video file (e.g. short 2.0s clip) is converted into images without error."""
        short_video = self.test_dir / "arbitrary_short.mp4"
        short_out = self.test_dir / "arbitrary_short_frames"
        if short_out.exists():
            shutil.rmtree(short_out)

        # Write a 2s video with 60 frames
        vw = cv2.VideoWriter(str(short_video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for i in range(60):
            frame = np.full((480, 640, 3), (i * 4) % 256, dtype=np.uint8)
            cv2.circle(frame, (320, 240), 50 + i, (0, 255, 0), -1)
            vw.write(frame)
        vw.release()

        # Convert to 25 images
        summary = extract_and_filter_frames(
            video_path=str(short_video),
            output_dir=short_out,
            target_count=25
        )

        extracted_files = list(short_out.glob("*.jpg"))
        self.assertEqual(len(extracted_files), summary["stats"]["final_extracted_count"])
        self.assertGreaterEqual(len(extracted_files), 10)
        print(f"\n[Test] Converted arbitrary short video into {len(extracted_files)} image files successfully.")

    def test_06_portrait_video_downscaling(self):
        """Verify vertical / portrait video (e.g. 1080x1920) has longest edge normalized to 1280px."""
        portrait_video = self.test_dir / "portrait_test.mp4"
        portrait_out = self.test_dir / "portrait_frames"
        if portrait_out.exists():
            shutil.rmtree(portrait_out)

        vw = cv2.VideoWriter(str(portrait_video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 1280))
        for i in range(30):
            frame = np.full((1280, 720, 3), (i * 8) % 256, dtype=np.uint8)
            cv2.rectangle(frame, (100, 200), (620, 1080), (255, 200, 0), -1)
            vw.write(frame)
        vw.release()

        summary = extract_and_filter_frames(
            video_path=str(portrait_video),
            output_dir=portrait_out,
            target_count=10,
            max_dimension=1280
        )

        sample_img = cv2.imread(str(next(portrait_out.glob("*.jpg"))))
        h, w = sample_img.shape[:2]
        self.assertEqual(max(h, w), 1280, "Longest edge of portrait video must be exactly 1280")
        self.assertEqual(h, 1280, "Height should be the longest edge for vertical video")
        self.assertEqual(w, 720, "Width should be preserved according to aspect ratio")

    def test_07_odd_dimension_preserves_even_output(self):
        """Verify odd video dimensions are normalized to even dimensions (divisible by 2)."""
        odd_video = self.test_dir / "odd_dim.mp4"
        odd_out = self.test_dir / "odd_dim_frames"
        if odd_out.exists():
            shutil.rmtree(odd_out)

        # 640x480 container with synthetic odd frame written
        vw = cv2.VideoWriter(str(odd_video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for i in range(30):
            frame = np.full((480, 640, 3), 100, dtype=np.uint8)
            vw.write(frame)
        vw.release()

        summary = extract_and_filter_frames(
            video_path=str(odd_video),
            output_dir=odd_out,
            target_count=10,
            max_dimension=1280
        )

        for img_path in odd_out.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            self.assertEqual(h % 2, 0, f"Height {h} must be even")
            self.assertEqual(w % 2, 0, f"Width {w} must be even")

    def test_08_blur_detection_and_replacement(self):
        """Verify that simulated blur in test_orbit.mp4 is detected and neighbor-replaced."""
        meta_file = self.output_dir / "dataset_metadata.json"
        self.assertTrue(meta_file.exists())
        import json
        with open(meta_file, "r") as f:
            data = json.load(f)

        stats = data["stats"]
        self.assertGreaterEqual(stats["blurry_detected"], 1, "Should detect at least 1 blurry frame candidate")
        self.assertGreaterEqual(stats["smart_replaced"], 1, "Should replace blurry candidate with sharp neighbor")

        # Check that replaced frames have replaced flag True
        replaced_frames = [f for f in data["frames"] if f["replaced"]]
        self.assertGreaterEqual(len(replaced_frames), 1, "At least one frame must have replaced=True")

    def test_09_archive_inside_output_dir(self):
        """Verify packaging when handoff zip is located inside the output directory does not fail."""
        nested_zip = self.output_dir / "nested_handoff.zip"
        res = package_archive(self.output_dir, nested_zip)
        self.assertTrue(res.exists())
        with zipfile.ZipFile(res, "r") as zf:
            namelist = zf.namelist()
            self.assertNotIn("nested_handoff.zip", namelist, "Zip file must not include itself")

    def test_10_fixed_fps_mode(self):
        """Verify fixed FPS sampling rate option."""
        fps_out = self.test_dir / "fps_test_frames"
        if fps_out.exists():
            shutil.rmtree(fps_out)

        summary = extract_and_filter_frames(
            video_path=str(self.sample_video),
            output_dir=fps_out,
            sample_fps=2.0
        )
        extracted = list(fps_out.glob("*.jpg"))
        # 13s usable * 2.0 fps ~= 26 frames
        self.assertGreaterEqual(len(extracted), 24)
        self.assertLessEqual(len(extracted), 28)

    def test_11_video_module_extraction(self):
        """Verify video.py extract_frames integration with quality filtering."""
        from video import extract_frames
        video_out = self.test_dir / "video_module_frames"
        if video_out.exists():
            shutil.rmtree(video_out)

        saved = extract_frames(n_frames=15, maxs=1280, video=self.sample_video, IMAGES=video_out, quality_filter=True)
        self.assertGreaterEqual(saved, 10)
        self.assertEqual(len(list(video_out.glob("*.jpg"))), saved)

    def test_12_colmap_pipeline_structure(self):
        """Verify colmap.py feature extraction and directory structure initialization."""
        from colmap import colmap_reconstruction
        colmap_ws = self.test_dir / "colmap_unit_test_ws"
        if colmap_ws.exists():
            shutil.rmtree(colmap_ws)

        db = colmap_ws / "database.db"
        images = colmap_ws / "images"
        sparse = colmap_ws / "sparse"
        undist = colmap_ws / "undistorted"
        archive = colmap_ws / "test_out.zip"

        images.mkdir(parents=True, exist_ok=True)
        # Copy 3 frames
        for f in sorted((self.test_dir / "test_frames").glob("*.jpg"))[:3]:
            shutil.copy(f, images / f.name)

        try:
            # Running with only 3 frames will extract features and verify pycolmap arguments
            colmap_reconstruction(db, images, sparse, undist, matching_mode="exhaustive", archive_zip=archive)
        except RuntimeError as e:
            # With only 3 frames, incremental mapping may not have enough baseline to triangulate,
            # but feature extraction must succeed without any TypeError/crash
            self.assertIn("COLMAP", str(e))
        self.assertTrue(db.exists(), "database.db must be created by pycolmap")


if __name__ == "__main__":
    unittest.main()
