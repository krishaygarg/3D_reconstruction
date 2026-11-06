#!/usr/bin/env bash
# ==============================================================================
# ACM AI - 3D Reconstruction Data Pipeline (Group 1)
# FFmpeg Direct Frame Extraction & Normalization Script
# ==============================================================================
# This script uses FFmpeg to:
# 1. Trim the start and end of the raw video to drop handshake.
# 2. Sample frames at 2.5 fps (or configured rate).
# 3. Downscale longest edge to 1280px maintaining aspect ratio.
# 4. Save high quality JPEGs.
# 5. Compress into a handoff zip archive.
# ==============================================================================

set -euo pipefail

INPUT_VIDEO="${1:-}"
OUTPUT_DIR="${2:-extracted_frames_ffmpeg}"
ARCHIVE_NAME="${3:-group2_handoff_ffmpeg.zip}"
TARGET_SPEC="${4:-100}"

if [ -z "$INPUT_VIDEO" ]; then
    echo "Usage: $0 <path_to_raw_video> [output_dir] [archive_name] [target_frames_or_fps]"
    echo "Example: $0 raw_orbit_4k.mp4 extracted_frames group2_handoff.zip 100"
    echo "Example (fixed FPS): $0 raw_orbit_4k.mp4 extracted_frames group2_handoff.zip 2.5"
    exit 1
fi

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video '$INPUT_VIDEO' does not exist."
    exit 1
fi

for cmd in ffmpeg ffprobe zip awk; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: Required command '$cmd' is not installed or not in PATH."
        exit 1
    fi
done

# Configuration defaults
TRIM_START=1.5       # Seconds to discard at start (handshake)
TRIM_END=1.5         # Seconds to discard at end
MAX_EDGE=1280        # Longest edge resolution (1280px)

echo "==> Inspecting video duration..."
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_VIDEO")
if [ -z "$DURATION" ] || [ "$DURATION" = "N/A" ]; then
    echo "Error: Could not determine video duration with ffprobe."
    exit 1
fi
echo "Total duration: ${DURATION}s"

# Calculate usable duration
USABLE_DURATION=$(awk -v dur="$DURATION" -v ts="$TRIM_START" -v te="$TRIM_END" 'BEGIN {print dur - ts - te}')
if (( $(awk -v ud="$USABLE_DURATION" 'BEGIN {print (ud <= 0)}') )); then
    echo "Notice: Video duration (${DURATION}s) is short. Adjusting start/end trim to 0s to avoid frame loss."
    TRIM_START=0
    TRIM_END=0
    USABLE_DURATION="$DURATION"
fi

# Determine whether TARGET_SPEC is a target frame count (e.g. 100) or an explicit FPS rate (e.g. 2.5)
if [[ "$TARGET_SPEC" =~ \. ]] || (( $(awk -v ts="$TARGET_SPEC" 'BEGIN {print (ts <= 15.0)}') )); then
    SAMPLE_FPS="$TARGET_SPEC"
    echo "Using fixed extraction rate: ${SAMPLE_FPS} fps"
else
    SAMPLE_FPS=$(awk -v ud="$USABLE_DURATION" -v count="$TARGET_SPEC" 'BEGIN {
        fps = count / ud;
        if (fps <= 0) fps = 2.5;
        printf "%.3f", fps;
    }')
    echo "Auto-calculated extraction rate for target ${TARGET_SPEC} frames: ${SAMPLE_FPS} fps"
fi

echo "Trimming ${TRIM_START}s from start and ${TRIM_END}s from end (Usable: ${USABLE_DURATION}s)..."

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.jpg "$OUTPUT_DIR"/dataset_metadata.json

echo "==> Extracting, sampling at ${SAMPLE_FPS} fps, and scaling longest edge to ${MAX_EDGE}px..."

# FFmpeg filter:
# - accurate_seek and ss/t handles precise trimming
# - fps sets sampling rate
# - scale handles aspect-ratio preserving longest edge = 1280:
#   if input width >= input height (landscape): scale=1280:-2
#   if input height > input width (portrait): scale=-2:1280
ffmpeg -y -accurate_seek -ss "$TRIM_START" -i "$INPUT_VIDEO" -t "$USABLE_DURATION" \
  -vf "fps=fps=${SAMPLE_FPS},scale='if(gte(iw,ih),${MAX_EDGE},-2)':'if(gte(iw,ih),-2,${MAX_EDGE})'" \
  -q:v 2 \
  "$OUTPUT_DIR/frame_%04d.jpg"

FRAME_COUNT=$(ls -1 "$OUTPUT_DIR"/frame_*.jpg 2>/dev/null | wc -l | tr -d ' ')

echo "==> Extracted $FRAME_COUNT frames."

# Probe resolution of first frame
SAMPLE_FRAME=$(ls -1 "$OUTPUT_DIR"/frame_*.jpg 2>/dev/null | head -n 1)
RES="N/A"
if [ -n "$SAMPLE_FRAME" ] && [ -f "$SAMPLE_FRAME" ]; then
    RES=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$SAMPLE_FRAME" 2>/dev/null || echo "1280x720")
fi

TARGET_MET="false"
if [ "$FRAME_COUNT" -ge 80 ] && [ "$FRAME_COUNT" -le 130 ]; then
    echo "SUCCESS: Frame count ($FRAME_COUNT) is within target range [80, 130]."
    TARGET_MET="true"
else
    echo "NOTE: Frame count is $FRAME_COUNT. (Target range: [80, 130])."
fi

# Generate dataset_metadata.json for handoff consistency
cat <<EOF > "$OUTPUT_DIR/dataset_metadata.json"
{
  "source_video": "$(basename "$INPUT_VIDEO")",
  "pipeline": "ffmpeg_direct",
  "config": {
    "trim_start_sec": $TRIM_START,
    "trim_end_sec": $TRIM_END,
    "max_dimension": $MAX_EDGE,
    "sample_fps": $SAMPLE_FPS
  },
  "stats": {
    "total_duration_sec": $DURATION,
    "usable_duration_sec": $USABLE_DURATION,
    "final_extracted_count": $FRAME_COUNT,
    "target_range": [80, 130],
    "target_met": $TARGET_MET,
    "final_resolution": "$RES"
  }
}
EOF

# Create archive
mkdir -p "$(dirname "$ARCHIVE_NAME")"
ARCHIVE_FULLPATH="$(cd "$(dirname "$ARCHIVE_NAME")" 2>/dev/null && pwd)/$(basename "$ARCHIVE_NAME")"
echo "==> Compressing frames and metadata into $ARCHIVE_FULLPATH..."
(cd "$OUTPUT_DIR" && zip -q -r "$ARCHIVE_FULLPATH" frame_*.jpg dataset_metadata.json)

echo "==> Archive ready for Group 2: $ARCHIVE_FULLPATH"
