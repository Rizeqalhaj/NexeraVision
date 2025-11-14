#!/usr/bin/env python3
"""
Final Dataset Organizer for Violence Detection Training

Combines all non-violence sources into final training dataset:
- Previously downloaded: 10,454 non-violence videos
- UCF101 extracted: 8,000-10,000 non-violence videos
- Violence videos: 14,000 videos

Creates train/val/test splits ready for training.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def count_videos_in_dir(directory: Path, extensions=('.mp4', '.avi', '.webm', '.mov')) -> int:
    """Count video files in directory recursively"""
    count = 0
    for ext in extensions:
        count += len(list(directory.rglob(f'*{ext}')))
    return count


def collect_video_paths(directory: Path, extensions=('.mp4', '.avi', '.webm', '.mov')) -> List[Path]:
    """Collect all video paths from directory"""
    videos = []
    for ext in extensions:
        videos.extend(directory.rglob(f'*{ext}'))
    return videos


def organize_dataset(
    violence_dir: Path,
    nonviolence_dirs: List[Path],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_videos_per_class: int = None
):
    """
    Organize videos into train/val/test splits

    Args:
        violence_dir: Directory containing violence videos
        nonviolence_dirs: List of directories containing non-violence videos
        output_dir: Output directory for organized dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        max_videos_per_class: Maximum videos per class (for balancing)
    """

    print(f"\n{'='*70}")
    print("Organizing Final Dataset")
    print(f"{'='*70}\n")

    # Validate split ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Split ratios must sum to 1.0"

    # Collect violence videos
    print("📊 Collecting violence videos...")
    violence_videos = collect_video_paths(violence_dir)
    print(f"   Found: {len(violence_videos):,} violence videos")

    # Collect non-violence videos from all sources
    print("\n📊 Collecting non-violence videos...")
    nonviolence_videos = []
    for i, nv_dir in enumerate(nonviolence_dirs, 1):
        if not nv_dir.exists():
            print(f"   ⚠️  Directory not found: {nv_dir}")
            continue

        videos = collect_video_paths(nv_dir)
        nonviolence_videos.extend(videos)
        print(f"   [{i}/{len(nonviolence_dirs)}] {nv_dir.name}: {len(videos):,} videos")

    print(f"\n   Total non-violence: {len(nonviolence_videos):,} videos")

    # Balance classes if needed
    if max_videos_per_class:
        print(f"\n⚖️  Balancing classes (max {max_videos_per_class:,} per class)...")
        if len(violence_videos) > max_videos_per_class:
            violence_videos = random.sample(violence_videos, max_videos_per_class)
            print(f"   Violence: {len(violence_videos):,} videos (sampled)")
        if len(nonviolence_videos) > max_videos_per_class:
            nonviolence_videos = random.sample(nonviolence_videos, max_videos_per_class)
            print(f"   Non-violence: {len(nonviolence_videos):,} videos (sampled)")

    print(f"\n{'='*70}")
    print("Dataset Summary Before Split")
    print(f"{'='*70}")
    print(f"Violence:     {len(violence_videos):,} videos")
    print(f"Non-violence: {len(nonviolence_videos):,} videos")
    print(f"Total:        {len(violence_videos) + len(nonviolence_videos):,} videos")
    print(f"Balance:      {len(nonviolence_videos) / len(violence_videos) * 100:.1f}%")
    print(f"{'='*70}\n")

    # Shuffle videos
    random.shuffle(violence_videos)
    random.shuffle(nonviolence_videos)

    # Calculate split sizes
    n_violence = len(violence_videos)
    n_nonviolence = len(nonviolence_videos)

    violence_train_end = int(n_violence * train_ratio)
    violence_val_end = violence_train_end + int(n_violence * val_ratio)

    nonviolence_train_end = int(n_nonviolence * train_ratio)
    nonviolence_val_end = nonviolence_train_end + int(n_nonviolence * val_ratio)

    # Split datasets
    splits = {
        'train': {
            'violence': violence_videos[:violence_train_end],
            'nonviolence': nonviolence_videos[:nonviolence_train_end]
        },
        'val': {
            'violence': violence_videos[violence_train_end:violence_val_end],
            'nonviolence': nonviolence_videos[nonviolence_train_end:nonviolence_val_end]
        },
        'test': {
            'violence': violence_videos[violence_val_end:],
            'nonviolence': nonviolence_videos[nonviolence_val_end:]
        }
    }

    # Create output directories and copy files
    print("📁 Creating dataset structure...")
    stats = defaultdict(lambda: defaultdict(int))

    for split_name, split_data in splits.items():
        for class_name, videos in split_data.items():
            # Create directory
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Copy videos
            print(f"\n   {split_name}/{class_name}: Copying {len(videos):,} videos...")

            for i, video_path in enumerate(videos, 1):
                # Create unique filename to avoid conflicts
                dest_name = f"{video_path.stem}_{i:06d}{video_path.suffix}"
                dest_path = split_dir / dest_name

                try:
                    shutil.copy2(video_path, dest_path)
                    stats[split_name][class_name] += 1

                    if i % 500 == 0:
                        print(f"      Progress: {i:,}/{len(videos):,} ({i/len(videos)*100:.1f}%)")

                except Exception as e:
                    print(f"      ✗ Failed to copy {video_path.name}: {e}")

            print(f"      ✓ Completed: {stats[split_name][class_name]:,} videos")

    # Save statistics
    print(f"\n{'='*70}")
    print("Final Dataset Statistics")
    print(f"{'='*70}\n")

    total_videos = 0
    for split_name in ['train', 'val', 'test']:
        violence_count = stats[split_name]['violence']
        nonviolence_count = stats[split_name]['nonviolence']
        split_total = violence_count + nonviolence_count
        total_videos += split_total

        print(f"{split_name.upper():6s}:")
        print(f"  Violence:     {violence_count:6,} ({violence_count/split_total*100:5.1f}%)")
        print(f"  Non-violence: {nonviolence_count:6,} ({nonviolence_count/split_total*100:5.1f}%)")
        print(f"  Total:        {split_total:6,}")
        print()

    print(f"{'='*70}")
    print(f"TOTAL DATASET: {total_videos:,} videos")
    print(f"{'='*70}\n")

    # Save detailed statistics
    stats_file = output_dir / 'dataset_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'splits': dict(stats),
            'total_videos': total_videos,
            'split_ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            },
            'balance': {
                'violence': len(violence_videos),
                'nonviolence': len(nonviolence_videos),
                'ratio': len(nonviolence_videos) / len(violence_videos)
            }
        }, f, indent=2)

    print(f"📊 Statistics saved to: {stats_file}\n")

    return stats, total_videos


def verify_dataset(dataset_dir: Path):
    """Verify organized dataset structure"""
    print(f"\n{'='*70}")
    print("Dataset Verification")
    print(f"{'='*70}\n")

    required_dirs = [
        'train/violence', 'train/nonviolence',
        'val/violence', 'val/nonviolence',
        'test/violence', 'test/nonviolence'
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = dataset_dir / dir_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        video_count = count_videos_in_dir(full_path) if exists else 0

        print(f"{status} {dir_path:25s}: {video_count:6,} videos")

        if not exists:
            all_exist = False

    print(f"\n{'='*70}\n")

    if all_exist:
        print("✅ Dataset structure is valid!")
        print("\n🚀 Ready for training!")
        print("\nNext step:")
        print("cd violence_detection_mvp")
        print(f"python train_rtx5000_dual_IMPROVED.py --dataset-path {dataset_dir.absolute()}")
    else:
        print("❌ Dataset structure is incomplete!")

    print(f"\n{'='*70}\n")


def main():
    print("="*70)
    print("Final Dataset Organizer for Violence Detection")
    print("="*70)
    print("Combines all non-violence sources into training dataset\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python organize_final_dataset.py [options]")
            print()
            print("Interactive mode (recommended):")
            print("  python organize_final_dataset.py")
            print()
            print("Command-line mode:")
            print("  python organize_final_dataset.py \\")
            print("    --violence /path/to/violence \\")
            print("    --nonviolence /path/to/nv1 /path/to/nv2 \\")
            print("    --output /path/to/organized_dataset \\")
            print("    --balance 14000")
            print()
            print("Options:")
            print("  --violence PATH       : Directory with violence videos")
            print("  --nonviolence PATH... : One or more directories with non-violence videos")
            print("  --output PATH         : Output directory for organized dataset")
            print("  --balance N           : Balance classes to N videos each")
            print("  --train-ratio 0.7     : Training set ratio (default: 0.7)")
            print("  --val-ratio 0.15      : Validation set ratio (default: 0.15)")
            print("  --test-ratio 0.15     : Test set ratio (default: 0.15)")
            sys.exit(0)

    # Interactive mode
    print("Interactive Dataset Organization")
    print("="*70)

    # Get violence directory
    violence_default = "./violence_videos"
    violence_input = input(f"\nViolence videos directory (default: {violence_default}): ").strip()
    violence_dir = Path(violence_input) if violence_input else Path(violence_default)

    if not violence_dir.exists():
        print(f"❌ Violence directory not found: {violence_dir}")
        sys.exit(1)

    # Get non-violence directories
    print("\nNon-violence video directories:")
    print("(Enter paths one per line, empty line to finish)")

    nonviolence_dirs = []
    default_dirs = [
        "./nonviolence_videos",
        "./ucf101_nonviolent"
    ]

    # Show defaults
    for i, default_dir in enumerate(default_dirs, 1):
        print(f"  {i}. {default_dir}")

    print("\nUse defaults? (y/n): ", end="")
    use_defaults = input().strip().lower()

    if use_defaults == 'y':
        nonviolence_dirs = [Path(d) for d in default_dirs if Path(d).exists()]
    else:
        print("Enter directory paths (empty to finish):")
        while True:
            nv_input = input(f"  [{len(nonviolence_dirs)+1}]: ").strip()
            if not nv_input:
                break
            nv_dir = Path(nv_input)
            if nv_dir.exists():
                nonviolence_dirs.append(nv_dir)
                print(f"     ✓ Added: {nv_dir}")
            else:
                print(f"     ⚠️  Directory not found: {nv_dir}")

    if not nonviolence_dirs:
        print("❌ No non-violence directories provided!")
        sys.exit(1)

    # Get output directory
    output_default = "./organized_dataset"
    output_input = input(f"\nOutput directory (default: {output_default}): ").strip()
    output_dir = Path(output_input) if output_input else Path(output_default)

    # Get balance option
    balance_input = input("\nBalance classes? (y/n, default: y): ").strip().lower()
    balance = balance_input != 'n'

    if balance:
        max_videos_input = input("Max videos per class (default: auto-balance, press Enter): ").strip()
        max_videos_per_class = int(max_videos_input) if max_videos_input else None
    else:
        max_videos_per_class = None

    # Confirm
    print(f"\n{'='*70}")
    print("Configuration Summary")
    print(f"{'='*70}")
    print(f"Violence directory:     {violence_dir}")
    print(f"Non-violence directories:")
    for nv_dir in nonviolence_dirs:
        print(f"  - {nv_dir}")
    print(f"Output directory:       {output_dir}")
    print(f"Balance classes:        {balance}")
    if max_videos_per_class:
        print(f"Max per class:          {max_videos_per_class:,}")
    print(f"{'='*70}\n")

    confirm = input("Proceed with organization? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Organize dataset
    random.seed(42)  # Reproducible splits

    stats, total_videos = organize_dataset(
        violence_dir=violence_dir,
        nonviolence_dirs=nonviolence_dirs,
        output_dir=output_dir,
        max_videos_per_class=max_videos_per_class
    )

    # Verify dataset
    verify_dataset(output_dir)

    print("\n✅ Dataset organization complete!")


if __name__ == "__main__":
    main()
