#!/usr/bin/env python3
"""
Extract random 2-second clips from long videos
This simulates CCTV sampling and creates training data from long videos
"""

import cv2
import random
from pathlib import Path
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
LONG_VIDEOS_DIR = Path("/workspace/youtube_long_videos")
OUTPUT_DIR = Path("/workspace/nonviolence_clips_from_long")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIPS_PER_VIDEO = 20  # Extract 20 random clips from each long video
CLIP_DURATION_SECONDS = 2  # 2-second clips
TARGET_FPS = 30

print("=" * 80)
print("EXTRACT CLIPS FROM LONG VIDEOS")
print("=" * 80)
print()

# Find all long videos
video_files = list(LONG_VIDEOS_DIR.glob("*.mp4")) + list(LONG_VIDEOS_DIR.glob("*.webm"))

if not video_files:
    print(f"❌ No videos found in {LONG_VIDEOS_DIR}")
    exit(1)

print(f"Found {len(video_files)} long videos")
print(f"Extracting {CLIPS_PER_VIDEO} clips per video")
print(f"Clip duration: {CLIP_DURATION_SECONDS} seconds")
print()

total_clips_extracted = 0
failed_videos = 0

for idx, video_file in enumerate(video_files, 1):
    print(f"[{idx}/{len(video_files)}] Processing: {video_file.name}")

    try:
        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            print(f"  ❌ Failed to open video")
            failed_videos += 1
            continue

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0

        if duration_seconds < 60:  # Skip videos shorter than 1 minute
            print(f"  ⏭️  Too short ({duration_seconds:.1f}s), skipping")
            cap.release()
            continue

        print(f"  Duration: {duration_seconds/60:.1f} min, FPS: {fps:.1f}, Frames: {total_frames}")

        # Calculate frames per clip
        frames_per_clip = int(fps * CLIP_DURATION_SECONDS)

        # Sample random start positions
        max_start_frame = total_frames - frames_per_clip
        if max_start_frame <= 0:
            print(f"  ⏭️  Video too short for clip extraction")
            cap.release()
            continue

        start_frames = random.sample(range(0, max_start_frame), min(CLIPS_PER_VIDEO, max_start_frame))

        clips_extracted = 0

        for clip_idx, start_frame in enumerate(start_frames):
            # Set video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Extract frames
            frames = []
            for _ in range(frames_per_clip):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if len(frames) < frames_per_clip * 0.8:  # Need at least 80% of frames
                continue

            # Save clip as video
            clip_name = f"{video_file.stem}_clip_{clip_idx:03d}.mp4"
            clip_path = OUTPUT_DIR / clip_name

            # Write video
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

            for frame in frames:
                out.write(frame)

            out.release()
            clips_extracted += 1

        cap.release()

        print(f"  ✓ Extracted {clips_extracted} clips")
        total_clips_extracted += clips_extracted

    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")
        failed_videos += 1

print()
print("=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print()
print(f"✓ Processed: {len(video_files)} videos")
print(f"✓ Extracted: {total_clips_extracted} clips")
print(f"❌ Failed: {failed_videos} videos")
print(f"📁 Output: {OUTPUT_DIR}")
print()

# Calculate total size
clips = list(OUTPUT_DIR.glob("*.mp4"))
total_size = sum(f.stat().st_size for f in clips)
print(f"Total clips: {len(clips)}")
print(f"Total size: {total_size/(1024**3):.2f} GB")
print()
print("These clips simulate CCTV sampling from long videos")
print("=" * 80)
