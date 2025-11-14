#!/usr/bin/env python3
"""
Create physical train/val/test splits with actual video files
Organizes videos into folder structure for easy training
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: Combined dataset
COMBINED_DATASET = "/workspace/datasets/combined_dataset"

# Output: Physical folder structure
OUTPUT_DIR = "/workspace/organized_dataset"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Video extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']

# Random seed for reproducibility
RANDOM_SEED = 42

# Mode: 'copy' or 'move'
# copy = safe, preserves originals
# move = saves space, empties combined_dataset
MODE = 'copy'


# ============================================================================
# FUNCTIONS
# ============================================================================

def find_videos(directory: Path):
    """Find all videos in directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(list(directory.glob(f'*{ext}')))
    return videos


def create_physical_splits():
    """Create physical folder structure with actual video files."""
    print("="*80)
    print("CREATE PHYSICAL TRAIN/VAL/TEST SPLITS")
    print("="*80)
    print()

    combined_path = Path(COMBINED_DATASET)

    if not combined_path.exists():
        print(f"❌ ERROR: {COMBINED_DATASET} not found")
        print(f"   Run combine_all_datasets.py first!")
        return False

    violent_dir = combined_path / "violent"
    nonviolent_dir = combined_path / "nonviolent"

    if not violent_dir.exists() or not nonviolent_dir.exists():
        print(f"❌ ERROR: violent/ or nonviolent/ folders not found")
        return False

    # Find all videos
    print("📁 Finding videos...")
    violent_videos = find_videos(violent_dir)
    nonviolent_videos = find_videos(nonviolent_dir)

    print(f"   ⚠️  Violent: {len(violent_videos):,} videos")
    print(f"   ✅ Non-Violent: {len(nonviolent_videos):,} videos")
    print(f"   📊 Total: {len(violent_videos) + len(nonviolent_videos):,} videos")

    if len(violent_videos) == 0 or len(nonviolent_videos) == 0:
        print(f"\n❌ ERROR: Need both violent and non-violent videos!")
        return False

    # Check balance
    ratio = min(len(violent_videos), len(nonviolent_videos)) / max(len(violent_videos), len(nonviolent_videos))
    print(f"\n⚖️  Class Balance: {ratio:.2%}")

    # Shuffle with seed
    random.seed(RANDOM_SEED)
    random.shuffle(violent_videos)
    random.shuffle(nonviolent_videos)

    # Calculate split sizes
    violent_train_size = int(len(violent_videos) * TRAIN_RATIO)
    violent_val_size = int(len(violent_videos) * VAL_RATIO)

    nonviolent_train_size = int(len(nonviolent_videos) * TRAIN_RATIO)
    nonviolent_val_size = int(len(nonviolent_videos) * VAL_RATIO)

    # Split violent videos
    violent_splits = {
        'train': violent_videos[:violent_train_size],
        'val': violent_videos[violent_train_size:violent_train_size + violent_val_size],
        'test': violent_videos[violent_train_size + violent_val_size:]
    }

    # Split non-violent videos
    nonviolent_splits = {
        'train': nonviolent_videos[:nonviolent_train_size],
        'val': nonviolent_videos[nonviolent_train_size:nonviolent_train_size + nonviolent_val_size],
        'test': nonviolent_videos[nonviolent_train_size + nonviolent_val_size:]
    }

    # Print split statistics
    print("\n" + "="*80)
    print("📊 SPLIT STATISTICS")
    print("="*80)

    for split_name in ['train', 'val', 'test']:
        v_count = len(violent_splits[split_name])
        nv_count = len(nonviolent_splits[split_name])
        total = v_count + nv_count

        split_ratio = {'train': TRAIN_RATIO, 'val': VAL_RATIO, 'test': TEST_RATIO}[split_name]

        print(f"\n{'🔵 TRAIN' if split_name == 'train' else '🟢 VAL' if split_name == 'val' else '🟡 TEST'} ({split_ratio*100:.0f}%): {total:,} videos")
        print(f"   ⚠️  Violent: {v_count:,} ({v_count/total*100:.1f}%)")
        print(f"   ✅ Non-Violent: {nv_count:,} ({nv_count/total*100:.1f}%)")

    # Ask for confirmation
    print("\n" + "="*80)
    print("OPERATION MODE")
    print("="*80)
    print(f"\nCurrent mode: {MODE.upper()}")

    if MODE == 'move':
        print("   ⚠️  WARNING: This will EMPTY the combined_dataset folder!")
        print("   Original files will be moved to train/val/test folders.")
    else:
        print("   ✅ SAFE: Files will be copied, originals preserved.")

    print("\nChange mode? (current: {})".format(MODE))
    print("  1. copy - Copy files (safe, needs 2x space)")
    print("  2. move - Move files (saves space)")
    print()

    mode_choice = input("Select mode (1/2) [press Enter to keep current]: ").strip()

    if mode_choice == '1':
        mode = 'copy'
    elif mode_choice == '2':
        mode = 'move'
        confirm = input("⚠️  Confirm MOVE mode? Type 'yes': ").strip().lower()
        if confirm != 'yes':
            print("Cancelled. Using copy mode.")
            mode = 'copy'
    else:
        mode = MODE

    # Create output directory structure
    print("\n" + "="*80)
    print("📁 CREATING FOLDER STRUCTURE")
    print("="*80)
    print()

    output_path = Path(OUTPUT_DIR)

    folders = []
    for split in ['train', 'val', 'test']:
        for category in ['violent', 'nonviolent']:
            folder = output_path / split / category
            folder.mkdir(parents=True, exist_ok=True)
            folders.append(folder)
            print(f"   ✅ {folder}")

    # Copy/Move files
    print("\n" + "="*80)
    print(f"{'📦 COPYING' if mode == 'copy' else '🚚 MOVING'} FILES")
    print("="*80)

    stats = {'train': {'violent': 0, 'nonviolent': 0},
             'val': {'violent': 0, 'nonviolent': 0},
             'test': {'violent': 0, 'nonviolent': 0},
             'errors': 0}

    # Process each split
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'🔵' if split_name == 'train' else '🟢' if split_name == 'val' else '🟡'} Processing {split_name.upper()}...")

        # Process violent videos
        dest_dir = output_path / split_name / 'violent'
        videos = violent_splits[split_name]

        for i, video in enumerate(tqdm(videos, desc=f"   Violent")):
            try:
                # Keep original filename to preserve any useful info
                dst = dest_dir / video.name

                # Handle duplicate names
                counter = 1
                while dst.exists():
                    stem = video.stem
                    dst = dest_dir / f"{stem}_{counter}{video.suffix}"
                    counter += 1

                if mode == 'move':
                    shutil.move(str(video), str(dst))
                else:
                    shutil.copy2(str(video), str(dst))

                stats[split_name]['violent'] += 1

            except Exception as e:
                print(f"\n      ⚠️  Error with {video.name}: {e}")
                stats['errors'] += 1

        # Process non-violent videos
        dest_dir = output_path / split_name / 'nonviolent'
        videos = nonviolent_splits[split_name]

        for i, video in enumerate(tqdm(videos, desc=f"   Non-Violent")):
            try:
                dst = dest_dir / video.name

                # Handle duplicate names
                counter = 1
                while dst.exists():
                    stem = video.stem
                    dst = dest_dir / f"{stem}_{counter}{video.suffix}"
                    counter += 1

                if mode == 'move':
                    shutil.move(str(video), str(dst))
                else:
                    shutil.copy2(str(video), str(dst))

                stats[split_name]['nonviolent'] += 1

            except Exception as e:
                print(f"\n      ⚠️  Error with {video.name}: {e}")
                stats['errors'] += 1

    # Save metadata
    metadata = {
        'source': str(COMBINED_DATASET),
        'output': str(OUTPUT_DIR),
        'mode': mode,
        'random_seed': RANDOM_SEED,
        'splits': stats,
        'ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        }
    }

    metadata_file = output_path / "split_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("✅ PHYSICAL SPLITS CREATED")
    print("="*80)

    print(f"\n📁 Output Structure: {OUTPUT_DIR}")
    print(f"   ├── train/")
    print(f"   │   ├── violent/ ({stats['train']['violent']:,} videos)")
    print(f"   │   └── nonviolent/ ({stats['train']['nonviolent']:,} videos)")
    print(f"   ├── val/")
    print(f"   │   ├── violent/ ({stats['val']['violent']:,} videos)")
    print(f"   │   └── nonviolent/ ({stats['val']['nonviolent']:,} videos)")
    print(f"   └── test/")
    print(f"       ├── violent/ ({stats['test']['violent']:,} videos)")
    print(f"       └── nonviolent/ ({stats['test']['nonviolent']:,} videos)")

    print(f"\n📊 Summary:")
    train_total = stats['train']['violent'] + stats['train']['nonviolent']
    val_total = stats['val']['violent'] + stats['val']['nonviolent']
    test_total = stats['test']['violent'] + stats['test']['nonviolent']

    print(f"   🔵 Train: {train_total:,} videos (70%)")
    print(f"   🟢 Val:   {val_total:,} videos (15%)")
    print(f"   🟡 Test:  {test_total:,} videos (15%)")
    print(f"   ❌ Errors: {stats['errors']}")

    if mode == 'move':
        print(f"\n   ⚠️  Original {COMBINED_DATASET} is now empty (files moved)")

    print("\n🎯 NEXT STEP:")
    print(f"   Train the model:")
    print(f"   python3 train_dual_rtx5000.py")
    print()

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     CREATE PHYSICAL TRAIN/VAL/TEST FOLDER STRUCTURE         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    success = create_physical_splits()

    if not success:
        print("❌ Failed to create splits. Check errors above.")
        return

    print("="*80)
    print("✅ ALL DONE! Dataset ready for training!")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
