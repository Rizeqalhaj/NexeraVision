#!/usr/bin/env python3
"""
Add New Videos to Existing Organized Dataset

Adds new videos from:
- /workspace/Non-violence (new non-violence videos)
- /workspace/downloaded_videos (new violence videos)

To existing dataset:
- /workspace/organized_dataset/train/
- /workspace/organized_dataset/val/
- /workspace/organized_dataset/test/

With 70/15/15 split ratio
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import List, Dict


def collect_videos(directory: Path, extensions=('.mp4', '.avi', '.webm', '.mov')) -> List[Path]:
    """Collect all video files from directory"""
    videos = []
    if not directory.exists():
        print(f"⚠️  Directory not found: {directory}")
        return videos

    for ext in extensions:
        videos.extend(directory.rglob(f'*{ext}'))
    return videos


def split_videos(videos: List[Path], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> Dict[str, List[Path]]:
    """Split videos into train/val/test sets"""
    random.shuffle(videos)

    n_total = len(videos)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return {
        'train': videos[:n_train],
        'val': videos[n_train:n_train + n_val],
        'test': videos[n_train + n_val:]
    }


def move_videos_to_split(videos: List[Path], dest_dir: Path, class_name: str) -> int:
    """Move videos to destination directory"""
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for video in videos:
        # Get existing files to create unique name
        existing_files = list(dest_dir.glob(f'*{video.suffix}'))
        new_name = f"{class_name}_{len(existing_files) + moved:06d}{video.suffix}"
        dest_path = dest_dir / new_name

        try:
            shutil.move(str(video), str(dest_path))
            moved += 1

            if moved % 100 == 0:
                print(f"      Progress: {moved}/{len(videos)}")

        except Exception as e:
            print(f"   ✗ Failed to move {video.name}: {e}")

    return moved


def main():
    print("="*70)
    print("Add New Videos to Existing Dataset (MOVE)")
    print("="*70)
    print("Split ratio: 70% train / 15% val / 15% test")
    print("⚠️  Videos will be MOVED (not copied) from source directories\n")

    # Define paths
    nonviolence_source = Path('/workspace/Non-violence')
    violence_source = Path('/workspace/downloaded_videos')
    dataset_root = Path('/workspace/organized_dataset')

    print("📂 Source directories:")
    print(f"   Non-violence: {nonviolence_source}")
    print(f"   Violence:     {violence_source}")
    print(f"\n📂 Destination dataset:")
    print(f"   {dataset_root}\n")

    # Collect new videos
    print("="*70)
    print("Collecting new videos...")
    print("="*70)

    print("\n📊 Non-violence videos:")
    nonviolence_videos = collect_videos(nonviolence_source)
    print(f"   Found: {len(nonviolence_videos):,} videos")

    print("\n📊 Violence videos:")
    violence_videos = collect_videos(violence_source)
    print(f"   Found: {len(violence_videos):,} videos")

    if len(nonviolence_videos) == 0 and len(violence_videos) == 0:
        print("\n❌ No videos found in source directories!")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Total new videos to add:")
    print(f"{'='*70}")
    print(f"Non-violence: {len(nonviolence_videos):,}")
    print(f"Violence:     {len(violence_videos):,}")
    print(f"Total:        {len(nonviolence_videos) + len(violence_videos):,}")
    print(f"{'='*70}\n")

    # Confirm
    confirm = input("Proceed with adding videos to dataset? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Split videos into train/val/test
    random.seed(42)  # Reproducible splits

    print("\n" + "="*70)
    print("Splitting videos (70/15/15)...")
    print("="*70)

    nv_splits = split_videos(nonviolence_videos)
    v_splits = split_videos(violence_videos)

    print(f"\nNon-violence split:")
    print(f"   Train: {len(nv_splits['train']):,} videos")
    print(f"   Val:   {len(nv_splits['val']):,} videos")
    print(f"   Test:  {len(nv_splits['test']):,} videos")

    print(f"\nViolence split:")
    print(f"   Train: {len(v_splits['train']):,} videos")
    print(f"   Val:   {len(v_splits['val']):,} videos")
    print(f"   Test:  {len(v_splits['test']):,} videos")

    # Copy videos to dataset
    print("\n" + "="*70)
    print("Adding videos to dataset...")
    print("="*70)

    stats = {
        'train': {'violent': 0, 'nonviolent': 0},
        'val': {'violent': 0, 'nonviolent': 0},
        'test': {'violent': 0, 'nonviolent': 0}
    }

    for split_name in ['train', 'val', 'test']:
        print(f"\n📁 Processing {split_name}...")

        # Add non-violence videos
        if len(nv_splits[split_name]) > 0:
            dest_dir = dataset_root / split_name / 'nonviolent'
            print(f"   Moving {len(nv_splits[split_name]):,} non-violence videos...")
            moved = move_videos_to_split(nv_splits[split_name], dest_dir, 'nonviolent')
            stats[split_name]['nonviolent'] = moved
            print(f"   ✓ Moved {moved:,} non-violence videos")

        # Add violence videos
        if len(v_splits[split_name]) > 0:
            dest_dir = dataset_root / split_name / 'violent'
            print(f"   Moving {len(v_splits[split_name]):,} violence videos...")
            moved = move_videos_to_split(v_splits[split_name], dest_dir, 'violent')
            stats[split_name]['violent'] = moved
            print(f"   ✓ Moved {moved:,} violence videos")

    # Show final statistics
    print("\n" + "="*70)
    print("FINAL DATASET STATISTICS")
    print("="*70)

    total_added = 0
    for split_name in ['train', 'val', 'test']:
        split_dir = dataset_root / split_name

        # Count total videos in each class
        nv_dir = split_dir / 'nonviolent'
        v_dir = split_dir / 'violent'

        nv_total = len(list(nv_dir.glob('*.*'))) if nv_dir.exists() else 0
        v_total = len(list(v_dir.glob('*.*'))) if v_dir.exists() else 0
        split_total = nv_total + v_total

        added_nv = stats[split_name]['nonviolent']
        added_v = stats[split_name]['violent']
        added_split = added_nv + added_v
        total_added += added_split

        print(f"\n{split_name.upper()}:")
        print(f"   Violent:     {v_total:6,} total ({added_v:,} newly added)")
        print(f"   Non-violent: {nv_total:6,} total ({added_nv:,} newly added)")
        print(f"   Split total: {split_total:6,} ({added_split:,} newly added)")
        print(f"   Balance:     {nv_total/v_total*100:.1f}% non-violent" if v_total > 0 else "")

    # Grand total
    all_videos = len(list(dataset_root.rglob('*.*')))

    print(f"\n{'='*70}")
    print(f"TOTAL DATASET: {all_videos:,} videos")
    print(f"Videos added:  {total_added:,}")
    print(f"{'='*70}\n")

    print("✅ Dataset update complete!")
    print("\n🚀 Ready for training:")
    print("cd /workspace/violence_detection_mvp")
    print("python train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset --epochs 100 --batch-size 64")
    print()


if __name__ == "__main__":
    main()
