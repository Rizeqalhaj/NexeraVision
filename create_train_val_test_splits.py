#!/usr/bin/env python3
"""
Create train/val/test splits (70/15/15) from combined dataset
Generates text files with video paths and labels for training
"""

import os
import random
from pathlib import Path
from collections import defaultdict
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: Combined dataset (after running combine_all_datasets.py)
COMBINED_DATASET = "/workspace/datasets/combined_dataset"

# Output: Organized dataset with splits
OUTPUT_DIR = "/workspace/organized_dataset"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Video extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================================
# FUNCTIONS
# ============================================================================

def find_videos(directory: Path):
    """Find all videos in directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(list(directory.glob(f'*{ext}')))
    return videos


def create_splits():
    """Create train/val/test splits."""
    print("="*80)
    print("CREATE TRAIN/VAL/TEST SPLITS")
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
        print(f"❌ ERROR: violent/ or nonviolent/ folders not found in {COMBINED_DATASET}")
        print(f"   Expected structure:")
        print(f"   {COMBINED_DATASET}/violent/")
        print(f"   {COMBINED_DATASET}/nonviolent/")
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

    if ratio < 0.3:
        print(f"   ⚠️  SEVERE IMBALANCE!")
        smaller_class = "non-violent" if len(nonviolent_videos) < len(violent_videos) else "violent"
        print(f"   ⚠️  Consider collecting more {smaller_class} data")
    elif ratio < 0.7:
        print(f"   ⚠️  Moderate imbalance")
    else:
        print(f"   ✅ Good balance!")

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
    violent_train = violent_videos[:violent_train_size]
    violent_val = violent_videos[violent_train_size:violent_train_size + violent_val_size]
    violent_test = violent_videos[violent_train_size + violent_val_size:]

    # Split non-violent videos
    nonviolent_train = nonviolent_videos[:nonviolent_train_size]
    nonviolent_val = nonviolent_videos[nonviolent_train_size:nonviolent_train_size + nonviolent_val_size]
    nonviolent_test = nonviolent_videos[nonviolent_train_size + nonviolent_val_size:]

    # Combine and shuffle each split
    train_videos = [(v, 1) for v in violent_train] + [(v, 0) for v in nonviolent_train]
    val_videos = [(v, 1) for v in violent_val] + [(v, 0) for v in nonviolent_val]
    test_videos = [(v, 1) for v in violent_test] + [(v, 0) for v in nonviolent_test]

    random.shuffle(train_videos)
    random.shuffle(val_videos)
    random.shuffle(test_videos)

    # Print split statistics
    print("\n" + "="*80)
    print("📊 SPLIT STATISTICS")
    print("="*80)

    print(f"\n🔵 TRAIN ({TRAIN_RATIO*100:.0f}%): {len(train_videos):,} videos")
    train_violent = sum(1 for _, label in train_videos if label == 1)
    train_nonviolent = len(train_videos) - train_violent
    print(f"   ⚠️  Violent: {train_violent:,} ({train_violent/len(train_videos)*100:.1f}%)")
    print(f"   ✅ Non-Violent: {train_nonviolent:,} ({train_nonviolent/len(train_videos)*100:.1f}%)")

    print(f"\n🟢 VALIDATION ({VAL_RATIO*100:.0f}%): {len(val_videos):,} videos")
    val_violent = sum(1 for _, label in val_videos if label == 1)
    val_nonviolent = len(val_videos) - val_violent
    print(f"   ⚠️  Violent: {val_violent:,} ({val_violent/len(val_videos)*100:.1f}%)")
    print(f"   ✅ Non-Violent: {val_nonviolent:,} ({val_nonviolent/len(val_videos)*100:.1f}%)")

    print(f"\n🟡 TEST ({TEST_RATIO*100:.0f}%): {len(test_videos):,} videos")
    test_violent = sum(1 for _, label in test_videos if label == 1)
    test_nonviolent = len(test_videos) - test_violent
    print(f"   ⚠️  Violent: {test_violent:,} ({test_violent/len(test_videos)*100:.1f}%)")
    print(f"   ✅ Non-Violent: {test_nonviolent:,} ({test_nonviolent/len(test_videos)*100:.1f}%)")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save splits as text files (path,label format)
    print("\n" + "="*80)
    print("💾 SAVING SPLITS")
    print("="*80)

    train_file = output_path / "train_videos.txt"
    val_file = output_path / "val_videos.txt"
    test_file = output_path / "test_videos.txt"

    print(f"\n   Writing {train_file}")
    with open(train_file, 'w') as f:
        for video, label in train_videos:
            f.write(f"{video},{label}\n")

    print(f"   Writing {val_file}")
    with open(val_file, 'w') as f:
        for video, label in val_videos:
            f.write(f"{video},{label}\n")

    print(f"   Writing {test_file}")
    with open(test_file, 'w') as f:
        for video, label in test_videos:
            f.write(f"{video},{label}\n")

    # Save metadata
    metadata = {
        'source': COMBINED_DATASET,
        'output': OUTPUT_DIR,
        'total_videos': len(violent_videos) + len(nonviolent_videos),
        'splits': {
            'train': {
                'total': len(train_videos),
                'violent': train_violent,
                'nonviolent': train_nonviolent
            },
            'val': {
                'total': len(val_videos),
                'violent': val_violent,
                'nonviolent': val_nonviolent
            },
            'test': {
                'total': len(test_videos),
                'violent': test_violent,
                'nonviolent': test_nonviolent
            }
        },
        'ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'random_seed': RANDOM_SEED
    }

    metadata_file = output_path / "split_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✅ Metadata saved: {metadata_file}")

    print("\n" + "="*80)
    print("✅ SPLITS CREATED SUCCESSFULLY")
    print("="*80)

    print(f"\n📁 Output location: {OUTPUT_DIR}")
    print(f"   - train_videos.txt ({len(train_videos):,} videos)")
    print(f"   - val_videos.txt ({len(val_videos):,} videos)")
    print(f"   - test_videos.txt ({len(test_videos):,} videos)")

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
    print("║     CREATE TRAIN/VAL/TEST SPLITS (70/15/15)                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    success = create_splits()

    if not success:
        print("❌ Failed to create splits. Check errors above.")
        return

    print("="*80)
    print("✅ ALL DONE! Ready for training!")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
