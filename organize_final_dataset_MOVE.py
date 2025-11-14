#!/usr/bin/env python3
"""
MOVE videos from scattered directories into organized structure
- NO COPYING (saves space)
- NO DELETING (just moves)
- Split: 80% train, 10% val, 10% test
"""

import shutil
from pathlib import Path
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output structure
OUTPUT_BASE = Path("/workspace/Training")

# Non-Violence source directories
NONVIOLENCE_DIRS = [
    "/workspace/datasets/violence_detection/airtlab/violence-detection-dataset/non-violent/cam1",
    "/workspace/datasets/violence_detection/airtlab/violence-detection-dataset/non-violent/cam2",
    "/workspace/datasets/violence_detection/fight_detection_surv/noFight",
    "/workspace/datasets/violence_detection/vioperu/train/NonFight",
    "/workspace/datasets/violence_detection/vioperu/val/NonFight",
    "/workspace/exact_datasets/RLVS_Hockey/Violence Detection Dataset/NonViolence/RLVS",
    "/workspace/exact_datasets/RWF2000/RWF-2000/train/NonFight",
    "/workspace/exact_datasets/RWF2000/RWF-2000/val/NonFight",
    "/workspace/organized_dataset/test/nonviolent",
    "/workspace/ucf_crime_dropbox/normal_videos/Testing_Normal_Videos_Anomaly",
    "/workspace/organized_dataset/train/nonviolent",
    "/workspace/ucf_crime_dropbox/normal_videos/Normal_Videos_for_Event_Recognition",
    "/workspace/ucf_crime_dropbox/normal_videos/Training-Normal-Videos-Part-1",
    "/workspace/ucf_crime_dropbox/normal_videos/Training-Normal-Videos-Part-2",
]

# Violence source directories
VIOLENCE_DIRS = [
    "/workspace/datasets/violence_detection/airtlab/violence-detection-dataset/violent/cam1",
    "/workspace/datasets/violence_detection/airtlab/violence-detection-dataset/violent/cam2",
    "/workspace/datasets/violence_detection/fight_detection_surv/fight",
    "/workspace/datasets/violence_detection/vioperu/train/Fight",
    "/workspace/datasets/violence_detection/vioperu/val/Fight",
    "/workspace/exact_datasets/RLVS_Hockey/Violence Detection Dataset/Violence/RLVS",
    "/workspace/exact_datasets/RWF2000/RWF-2000/train/Fight",
    "/workspace/exact_datasets/RWF2000/RWF-2000/val/Fight",
    "/workspace/organized_dataset/test/violent",
    "/workspace/violence_detection_mvp/downloaded_reddit_videos",
    "/workspace/violence_detection_mvp/downloaded_worldstar",
]

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'}

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================================
# FUNCTIONS
# ============================================================================

def collect_videos(directories):
    """Collect all video files from given directories"""
    all_videos = []

    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            print(f"⚠️  Not found: {dir_path}")
            continue

        for ext in VIDEO_EXTENSIONS:
            all_videos.extend(path.glob(f"*{ext}"))
            all_videos.extend(path.rglob(f"*{ext}"))  # Recursive for nested

    # Remove duplicates
    all_videos = list(set(all_videos))
    return all_videos

def create_output_structure():
    """Create output directory structure"""
    dirs = [
        OUTPUT_BASE / "train" / "Violent",
        OUTPUT_BASE / "train" / "NonViolent",
        OUTPUT_BASE / "val" / "Violent",
        OUTPUT_BASE / "val" / "NonViolent",
        OUTPUT_BASE / "test" / "Violent",
        OUTPUT_BASE / "test" / "NonViolent",
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("ORGANIZE FINAL DATASET - MOVE OPERATION")
print("=" * 80)
print()

# Create output structure
print("Creating output structure...")
create_output_structure()
print(f"✓ Created: {OUTPUT_BASE}/")
print()

# Collect all videos
print("=" * 80)
print("COLLECTING NON-VIOLENCE VIDEOS")
print("=" * 80)
nonviolence_videos = collect_videos(NONVIOLENCE_DIRS)
print(f"✓ Found {len(nonviolence_videos):,} non-violence videos")
print()

print("=" * 80)
print("COLLECTING VIOLENCE VIDEOS")
print("=" * 80)
violence_videos = collect_videos(VIOLENCE_DIRS)
print(f"✓ Found {len(violence_videos):,} violence videos")
print()

# Shuffle for random split
random.seed(42)
random.shuffle(nonviolence_videos)
random.shuffle(violence_videos)

# Calculate split indices
def calculate_splits(total):
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count

nv_train, nv_val, nv_test = calculate_splits(len(nonviolence_videos))
v_train, v_val, v_test = calculate_splits(len(violence_videos))

print("=" * 80)
print("SPLIT PLAN")
print("=" * 80)
print()
print(f"NON-VIOLENCE:")
print(f"  Train: {nv_train:,} videos ({TRAIN_RATIO*100:.0f}%)")
print(f"  Val:   {nv_val:,} videos ({VAL_RATIO*100:.0f}%)")
print(f"  Test:  {nv_test:,} videos ({TEST_RATIO*100:.0f}%)")
print()
print(f"VIOLENCE:")
print(f"  Train: {v_train:,} videos ({TRAIN_RATIO*100:.0f}%)")
print(f"  Val:   {v_val:,} videos ({VAL_RATIO*100:.0f}%)")
print(f"  Test:  {v_test:,} videos ({TEST_RATIO*100:.0f}%)")
print()

# Confirm before moving
print("=" * 80)
print("⚠️  WARNING: ABOUT TO MOVE FILES")
print("=" * 80)
print()
print("This will MOVE (not copy) files from:")
print(f"  - {len(NONVIOLENCE_DIRS)} non-violence directories")
print(f"  - {len(VIOLENCE_DIRS)} violence directories")
print()
print(f"To: {OUTPUT_BASE}/")
print()
print("Files will be MOVED (not copied, not deleted)")
print()

# Ask for confirmation
response = input("Type 'YES' to continue: ")
if response.strip().upper() != "YES":
    print("Cancelled.")
    exit(0)

print()
print("=" * 80)
print("MOVING FILES")
print("=" * 80)
print()

# Move non-violence videos
print("Moving non-violence videos...")

# Non-violence train
print(f"  Train: Moving {nv_train} videos...")
for i, video in enumerate(nonviolence_videos[:nv_train]):
    target = OUTPUT_BASE / "train" / "NonViolent" / video.name
    shutil.move(str(video), str(target))
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{nv_train}")

# Non-violence val
print(f"  Val: Moving {nv_val} videos...")
for i, video in enumerate(nonviolence_videos[nv_train:nv_train+nv_val]):
    target = OUTPUT_BASE / "val" / "NonViolent" / video.name
    shutil.move(str(video), str(target))
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{nv_val}")

# Non-violence test
print(f"  Test: Moving {nv_test} videos...")
for i, video in enumerate(nonviolence_videos[nv_train+nv_val:]):
    target = OUTPUT_BASE / "test" / "NonViolent" / video.name
    shutil.move(str(video), str(target))
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{nv_test}")

print()
print("Moving violence videos...")

# Violence train
print(f"  Train: Moving {v_train} videos...")
for i, video in enumerate(violence_videos[:v_train]):
    target = OUTPUT_BASE / "train" / "Violent" / video.name
    shutil.move(str(video), str(target))
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{v_train}")

# Violence val
print(f"  Val: Moving {v_val} videos...")
for i, video in enumerate(violence_videos[v_train:v_train+v_val]):
    target = OUTPUT_BASE / "val" / "Violent" / video.name
    shutil.move(str(video), str(target))
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{v_val}")

# Violence test
print(f"  Test: Moving {v_test} videos...")
for i, video in enumerate(violence_videos[v_train+v_val:]):
    target = OUTPUT_BASE / "test" / "Violent" / video.name
    shutil.move(str(video), str(target))
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{v_test}")

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print()

# Final count
final_counts = {
    'train_violent': len(list((OUTPUT_BASE / "train" / "Violent").glob("*.*"))),
    'train_nonviolent': len(list((OUTPUT_BASE / "train" / "NonViolent").glob("*.*"))),
    'val_violent': len(list((OUTPUT_BASE / "val" / "Violent").glob("*.*"))),
    'val_nonviolent': len(list((OUTPUT_BASE / "val" / "NonViolent").glob("*.*"))),
    'test_violent': len(list((OUTPUT_BASE / "test" / "Violent").glob("*.*"))),
    'test_nonviolent': len(list((OUTPUT_BASE / "test" / "NonViolent").glob("*.*"))),
}

print(f"Final dataset structure:")
print(f"{OUTPUT_BASE}/")
print(f"├── train/")
print(f"│   ├── Violent/     {final_counts['train_violent']:,} videos")
print(f"│   └── NonViolent/  {final_counts['train_nonviolent']:,} videos")
print(f"├── val/")
print(f"│   ├── Violent/     {final_counts['val_violent']:,} videos")
print(f"│   └── NonViolent/  {final_counts['val_nonviolent']:,} videos")
print(f"└── test/")
print(f"    ├── Violent/     {final_counts['test_violent']:,} videos")
print(f"    └── NonViolent/  {final_counts['test_nonviolent']:,} videos")
print()

total_videos = sum(final_counts.values())
print(f"Total videos organized: {total_videos:,}")
print()
print("✅ All files MOVED successfully")
print("=" * 80)
