#!/usr/bin/env python3
"""
Balance dataset by adding longest YouTube violence videos
Current: 6,124 violence vs 7,432 non-violence (training)
Target: Balance by adding ~1,308 videos from YouTube folder
"""

import cv2
from pathlib import Path
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

YOUTUBE_VIOLENCE_DIR = Path("/workspace/violence_detection_mvp/downloaded_youtube_videos")
TRAINING_DIR = Path("/workspace/Training")

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'}

# Current counts
CURRENT_VIOLENCE = 6124
CURRENT_NONVIOLENCE = 7432
NEEDED_TO_BALANCE = CURRENT_NONVIOLENCE - CURRENT_VIOLENCE  # 1,308

print("=" * 80)
print("BALANCE DATASET WITH YOUTUBE VIOLENCE VIDEOS")
print("=" * 80)
print()

print("Current dataset:")
print(f"  Violence:     {CURRENT_VIOLENCE:,} videos")
print(f"  Non-Violence: {CURRENT_NONVIOLENCE:,} videos")
print(f"  Imbalance:    {NEEDED_TO_BALANCE:,} videos short")
print()

print(f"Plan: Add {NEEDED_TO_BALANCE:,} LONGEST videos from YouTube folder")
print()

# ============================================================================
# FIND ALL YOUTUBE VIDEOS
# ============================================================================

print("=" * 80)
print("SCANNING YOUTUBE FOLDER")
print("=" * 80)
print()

if not YOUTUBE_VIOLENCE_DIR.exists():
    print(f"❌ Not found: {YOUTUBE_VIOLENCE_DIR}")
    exit(1)

# Collect all videos
all_videos = []
for ext in VIDEO_EXTENSIONS:
    all_videos.extend(YOUTUBE_VIOLENCE_DIR.glob(f"*{ext}"))
    all_videos.extend(YOUTUBE_VIOLENCE_DIR.rglob(f"*{ext}"))

all_videos = list(set(all_videos))  # Remove duplicates

print(f"Found {len(all_videos):,} videos in YouTube folder")
print()

if len(all_videos) < NEEDED_TO_BALANCE:
    print(f"⚠️  Warning: Only {len(all_videos)} videos available, need {NEEDED_TO_BALANCE}")
    print(f"   Will use all available videos")
    NEEDED_TO_BALANCE = len(all_videos)
    print()

# ============================================================================
# GET VIDEO DURATIONS
# ============================================================================

print("=" * 80)
print("ANALYZING VIDEO DURATIONS")
print("=" * 80)
print()

video_durations = []

print("Scanning videos (this may take a few minutes)...")
for idx, video_path in enumerate(all_videos, 1):
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{len(all_videos)}")

    try:
        cap = cv2.VideoCapture(str(video_path))

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
                video_durations.append({
                    'path': video_path,
                    'duration': duration,
                    'size_mb': video_path.stat().st_size / (1024 * 1024)
                })

        cap.release()
    except:
        continue

print(f"✓ Successfully analyzed {len(video_durations):,} videos")
print()

# ============================================================================
# SELECT LONGEST VIDEOS
# ============================================================================

print("=" * 80)
print("SELECTING LONGEST VIDEOS")
print("=" * 80)
print()

# Sort by duration (longest first)
video_durations.sort(key=lambda x: x['duration'], reverse=True)

# Select top N longest videos
selected_videos = video_durations[:NEEDED_TO_BALANCE]

print(f"Selected {len(selected_videos):,} longest videos:")
print(f"  Shortest selected: {selected_videos[-1]['duration']:.1f} seconds")
print(f"  Longest selected:  {selected_videos[0]['duration']:.1f} seconds")
print(f"  Average duration:  {sum(v['duration'] for v in selected_videos) / len(selected_videos):.1f} seconds")
print()

total_size_gb = sum(v['size_mb'] for v in selected_videos) / 1024
print(f"Total size to move: {total_size_gb:.2f} GB")
print()

# ============================================================================
# SPLIT INTO TRAIN/VAL/TEST
# ============================================================================

print("=" * 80)
print("SPLITTING INTO TRAIN/VAL/TEST")
print("=" * 80)
print()

# Use same 80/10/10 split
train_count = int(len(selected_videos) * 0.8)
val_count = int(len(selected_videos) * 0.1)
test_count = len(selected_videos) - train_count - val_count

train_videos = selected_videos[:train_count]
val_videos = selected_videos[train_count:train_count+val_count]
test_videos = selected_videos[train_count+val_count:]

print(f"Split plan:")
print(f"  Train: {len(train_videos):,} videos")
print(f"  Val:   {len(val_videos):,} videos")
print(f"  Test:  {len(test_videos):,} videos")
print()

# ============================================================================
# CONFIRM
# ============================================================================

print("=" * 80)
print("⚠️  CONFIRMATION REQUIRED")
print("=" * 80)
print()
print(f"About to MOVE {len(selected_videos):,} videos from:")
print(f"  {YOUTUBE_VIOLENCE_DIR}")
print()
print("To:")
print(f"  {TRAINING_DIR}/train/Violent/   (+{len(train_videos):,} videos)")
print(f"  {TRAINING_DIR}/val/Violent/     (+{len(val_videos):,} videos)")
print(f"  {TRAINING_DIR}/test/Violent/    (+{len(test_videos):,} videos)")
print()
print("After balancing:")
print(f"  Train Violence:     {CURRENT_VIOLENCE} → {CURRENT_VIOLENCE + train_count:,}")
print(f"  Train Non-Violence: {CURRENT_NONVIOLENCE:,}")
print(f"  Difference:         {NEEDED_TO_BALANCE} → {CURRENT_NONVIOLENCE - (CURRENT_VIOLENCE + train_count)}")
print()

response = input("Type 'YES' to proceed: ")
if response.strip().upper() != "YES":
    print("Cancelled.")
    exit(0)

# ============================================================================
# MOVE VIDEOS
# ============================================================================

print()
print("=" * 80)
print("MOVING VIDEOS")
print("=" * 80)
print()

# Move train videos
print(f"Moving {len(train_videos)} videos to train/Violent/...")
train_dir = TRAINING_DIR / "train" / "Violent"
for idx, video_info in enumerate(train_videos, 1):
    target = train_dir / video_info['path'].name
    shutil.move(str(video_info['path']), str(target))
    if idx % 100 == 0:
        print(f"  {idx}/{len(train_videos)}")

# Move val videos
print(f"Moving {len(val_videos)} videos to val/Violent/...")
val_dir = TRAINING_DIR / "val" / "Violent"
for idx, video_info in enumerate(val_videos, 1):
    target = val_dir / video_info['path'].name
    shutil.move(str(video_info['path']), str(target))
    if idx % 100 == 0:
        print(f"  {idx}/{len(val_videos)}")

# Move test videos
print(f"Moving {len(test_videos)} videos to test/Violent/...")
test_dir = TRAINING_DIR / "test" / "Violent"
for idx, video_info in enumerate(test_videos, 1):
    target = test_dir / video_info['path'].name
    shutil.move(str(video_info['path']), str(target))
    if idx % 100 == 0:
        print(f"  {idx}/{len(test_videos)}")

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print()

# Final counts
final_train_violent = len(list((TRAINING_DIR / "train" / "Violent").glob("*.*")))
final_val_violent = len(list((TRAINING_DIR / "val" / "Violent").glob("*.*")))
final_test_violent = len(list((TRAINING_DIR / "test" / "Violent").glob("*.*")))

final_train_nonviolent = len(list((TRAINING_DIR / "train" / "NonViolent").glob("*.*")))
final_val_nonviolent = len(list((TRAINING_DIR / "val" / "NonViolent").glob("*.*")))
final_test_nonviolent = len(list((TRAINING_DIR / "test" / "NonViolent").glob("*.*")))

print("Final dataset structure:")
print(f"{TRAINING_DIR}/")
print(f"├── train/")
print(f"│   ├── Violent/     {final_train_violent:,} videos (was {CURRENT_VIOLENCE:,})")
print(f"│   └── NonViolent/  {final_train_nonviolent:,} videos")
print(f"├── val/")
print(f"│   ├── Violent/     {final_val_violent:,} videos")
print(f"│   └── NonViolent/  {final_val_nonviolent:,} videos")
print(f"└── test/")
print(f"    ├── Violent/     {final_test_violent:,} videos")
print(f"    └── NonViolent/  {final_test_nonviolent:,} videos")
print()

total_violent = final_train_violent + final_val_violent + final_test_violent
total_nonviolent = final_train_nonviolent + final_val_nonviolent + final_test_nonviolent
balance_diff = abs(total_violent - total_nonviolent)
balance_pct = (min(total_violent, total_nonviolent) / max(total_violent, total_nonviolent)) * 100

print(f"Total Violent:     {total_violent:,}")
print(f"Total Non-Violent: {total_nonviolent:,}")
print(f"Balance:           {balance_pct:.1f}% ({balance_diff} difference)")
print()

if balance_pct >= 95:
    print("✅ Dataset is well balanced!")
elif balance_pct >= 85:
    print("✅ Dataset balance is acceptable")
else:
    print("⚠️  Dataset still has some imbalance")

print()
print("=" * 80)
