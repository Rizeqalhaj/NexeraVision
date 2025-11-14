#!/usr/bin/env python3
"""
Fix dataset balance - currently 983 videos short on non-violence side
Options:
1. Move 983 violence videos back to youtube folder (remove excess)
2. Add 983 non-violence videos if available

Will check what's available and give you options
"""

from pathlib import Path
import shutil
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAINING_DIR = Path("/workspace/Training")

# Current imbalance
CURRENT_VIOLENT = 10287
CURRENT_NONVIOLENT = 9304
IMBALANCE = CURRENT_VIOLENT - CURRENT_NONVIOLENT  # 983

# Potential sources for non-violence videos
POTENTIAL_NONVIOLENCE_SOURCES = [
    "/workspace/violence_detection_mvp/downloaded_youtube_videos",  # Check for any non-violence
    "/workspace/youtube_nonviolence_videos",  # Shorts
    "/workspace/youtube_long_videos",  # Long videos
    "/workspace/nonviolence_clips_from_long",  # Clips
]

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'}

print("=" * 80)
print("FIX DATASET BALANCE")
print("=" * 80)
print()

print("Current imbalance:")
print(f"  Violent:     {CURRENT_VIOLENT:,} videos")
print(f"  Non-Violent: {CURRENT_NONVIOLENT:,} videos")
print(f"  Difference:  {IMBALANCE:,} videos (too many violence)")
print()

# ============================================================================
# CHECK FOR AVAILABLE NON-VIOLENCE VIDEOS
# ============================================================================

print("=" * 80)
print("CHECKING FOR AVAILABLE NON-VIOLENCE VIDEOS")
print("=" * 80)
print()

available_nonviolence = []

for source_dir in POTENTIAL_NONVIOLENCE_SOURCES:
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"⚠️  Not found: {source_dir}")
        continue

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(source_path.glob(f"*{ext}"))
        videos.extend(source_path.rglob(f"*{ext}"))

    videos = list(set(videos))

    if videos:
        print(f"✓ {source_dir}")
        print(f"  Found: {len(videos):,} videos")
        available_nonviolence.extend(videos)

print()
print(f"Total available non-violence videos: {len(available_nonviolence):,}")
print()

# ============================================================================
# DETERMINE BEST APPROACH
# ============================================================================

print("=" * 80)
print("BALANCING OPTIONS")
print("=" * 80)
print()

if len(available_nonviolence) >= IMBALANCE:
    print(f"✅ OPTION 1: Add {IMBALANCE:,} non-violence videos from available sources")
    print(f"   Available: {len(available_nonviolence):,} videos")
    print(f"   This will achieve perfect 1:1 balance")
    print()
    chosen_option = 1
else:
    print(f"⚠️  Only {len(available_nonviolence):,} non-violence videos available")
    print(f"   Need {IMBALANCE:,} to balance")
    print()
    print(f"✅ OPTION 2: Remove {IMBALANCE:,} violence videos (move back to source)")
    print(f"   This will balance at {CURRENT_NONVIOLENT:,} videos each")
    print()
    chosen_option = 2

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if chosen_option == 1:
    print(f"RECOMMENDED: Add {IMBALANCE:,} non-violence videos")
    print()
    print("This will give you:")
    print(f"  Violent:     {CURRENT_VIOLENT:,} videos")
    print(f"  Non-Violent: {CURRENT_NONVIOLENT + IMBALANCE:,} videos")
    print(f"  Balance:     100% (perfect 1:1)")
    print()
else:
    print(f"RECOMMENDED: Remove {IMBALANCE:,} excess violence videos")
    print()
    print("This will give you:")
    print(f"  Violent:     {CURRENT_NONVIOLENT:,} videos")
    print(f"  Non-Violent: {CURRENT_NONVIOLENT:,} videos")
    print(f"  Balance:     100% (perfect 1:1)")
    print()

response = input(f"Proceed with Option {chosen_option}? Type 'YES' to continue: ")
if response.strip().upper() != "YES":
    print("Cancelled.")
    exit(0)

print()

# ============================================================================
# EXECUTE CHOSEN OPTION
# ============================================================================

if chosen_option == 1:
    # Add non-violence videos
    print("=" * 80)
    print("ADDING NON-VIOLENCE VIDEOS")
    print("=" * 80)
    print()

    # Randomly select videos to add
    random.seed(42)
    random.shuffle(available_nonviolence)
    videos_to_add = available_nonviolence[:IMBALANCE]

    # Split into train/val/test (80/10/10)
    train_count = int(len(videos_to_add) * 0.8)
    val_count = int(len(videos_to_add) * 0.1)
    test_count = len(videos_to_add) - train_count - val_count

    train_videos = videos_to_add[:train_count]
    val_videos = videos_to_add[train_count:train_count+val_count]
    test_videos = videos_to_add[train_count+val_count:]

    print(f"Adding:")
    print(f"  Train: {len(train_videos):,} videos")
    print(f"  Val:   {len(val_videos):,} videos")
    print(f"  Test:  {len(test_videos):,} videos")
    print()

    # Move train
    print("Moving to train/NonViolent/...")
    train_dir = TRAINING_DIR / "train" / "NonViolent"
    for idx, video in enumerate(train_videos, 1):
        target = train_dir / video.name
        shutil.move(str(video), str(target))
        if idx % 100 == 0:
            print(f"  {idx}/{len(train_videos)}")

    # Move val
    print("Moving to val/NonViolent/...")
    val_dir = TRAINING_DIR / "val" / "NonViolent"
    for idx, video in enumerate(val_videos, 1):
        target = val_dir / video.name
        shutil.move(str(video), str(target))
        if idx % 50 == 0:
            print(f"  {idx}/{len(val_videos)}")

    # Move test
    print("Moving to test/NonViolent/...")
    test_dir = TRAINING_DIR / "test" / "NonViolent"
    for idx, video in enumerate(test_videos, 1):
        target = test_dir / video.name
        shutil.move(str(video), str(target))
        if idx % 50 == 0:
            print(f"  {idx}/{len(test_videos)}")

else:
    # Remove excess violence videos
    print("=" * 80)
    print("REMOVING EXCESS VIOLENCE VIDEOS")
    print("=" * 80)
    print()

    # Collect all violence videos
    all_violent_train = list((TRAINING_DIR / "train" / "Violent").glob("*.*"))
    all_violent_val = list((TRAINING_DIR / "val" / "Violent").glob("*.*"))
    all_violent_test = list((TRAINING_DIR / "test" / "Violent").glob("*.*"))

    # Calculate how many to remove from each split (proportionally)
    train_remove = int(IMBALANCE * 0.8)
    val_remove = int(IMBALANCE * 0.1)
    test_remove = IMBALANCE - train_remove - val_remove

    print(f"Removing:")
    print(f"  Train: {train_remove} videos")
    print(f"  Val:   {val_remove} videos")
    print(f"  Test:  {test_remove} videos")
    print()

    # Create backup directory
    backup_dir = Path("/workspace/excess_violence_videos")
    backup_dir.mkdir(exist_ok=True)

    # Remove from train
    random.shuffle(all_violent_train)
    print("Moving from train/Violent/ to backup...")
    for idx, video in enumerate(all_violent_train[:train_remove], 1):
        target = backup_dir / video.name
        shutil.move(str(video), str(target))
        if idx % 100 == 0:
            print(f"  {idx}/{train_remove}")

    # Remove from val
    random.shuffle(all_violent_val)
    print("Moving from val/Violent/ to backup...")
    for idx, video in enumerate(all_violent_val[:val_remove], 1):
        target = backup_dir / video.name
        shutil.move(str(video), str(target))
        if idx % 50 == 0:
            print(f"  {idx}/{val_remove}")

    # Remove from test
    random.shuffle(all_violent_test)
    print("Moving from test/Violent/ to backup...")
    for idx, video in enumerate(all_violent_test[:test_remove], 1):
        target = backup_dir / video.name
        shutil.move(str(video), str(target))
        if idx % 50 == 0:
            print(f"  {idx}/{test_remove}")

    print()
    print(f"✓ Moved {IMBALANCE} excess videos to: {backup_dir}")

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

print("Final balanced dataset:")
print(f"{TRAINING_DIR}/")
print(f"├── train/")
print(f"│   ├── Violent/     {final_train_violent:,} videos")
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
print(f"Difference:        {balance_diff}")
print(f"Balance:           {balance_pct:.1f}%")
print()

if balance_pct >= 99:
    print("✅ PERFECT BALANCE ACHIEVED!")
else:
    print(f"⚠️  Balance: {balance_pct:.1f}% (close enough)")

print()
print("=" * 80)
