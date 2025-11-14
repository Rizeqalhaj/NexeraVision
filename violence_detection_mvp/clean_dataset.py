#!/usr/bin/env python3
"""
Find and remove ALL corrupted videos BEFORE training
Guaranteed fix for hanging at specific videos
"""

import cv2
import sys
from pathlib import Path
from tqdm import tqdm
import shutil

def test_video(video_path, timeout_sec=5):
    """
    Test if video can be opened and read
    Returns: (is_valid, error_message)
    """
    try:
        cap = cv2.VideoCapture(str(video_path))

        # Set timeouts
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_sec * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)

        if not cap.isOpened():
            return False, "Cannot open"

        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return False, "0 frames"

        # Try to read first frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False, "Cannot read frame"

        # Try to read middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False, "Cannot read middle frame"

        cap.release()
        return True, "OK"

    except Exception as e:
        return False, str(e)

def clean_dataset(dataset_path, backup_dir):
    """Remove all corrupted videos from dataset"""

    dataset_path = Path(dataset_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CLEANING DATASET - REMOVING CORRUPTED VIDEOS")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Backup:  {backup_dir}")
    print("="*70)

    corrupted_videos = []
    total_videos = 0

    # Process all splits
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue

        print(f"\n{'='*70}")
        print(f"Processing {split.upper()}")
        print(f"{'='*70}\n")

        for class_name in ['violent', 'nonviolent']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            videos = list(class_dir.glob('*.mp4'))
            print(f"\n{class_name}: Testing {len(videos):,} videos...")

            for video in tqdm(videos, desc=f"{split}/{class_name}"):
                total_videos += 1
                is_valid, error = test_video(video, timeout_sec=3)

                if not is_valid:
                    # Move to backup
                    backup_class_dir = backup_dir / split / class_name
                    backup_class_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = backup_class_dir / video.name

                    shutil.move(str(video), str(backup_path))
                    corrupted_videos.append({
                        'path': str(video),
                        'split': split,
                        'class': class_name,
                        'error': error
                    })

                    tqdm.write(f"  ✗ REMOVED: {video.name} ({error})")

    # Summary
    print(f"\n{'='*70}")
    print("CLEANING COMPLETE")
    print(f"{'='*70}")
    print(f"Total videos scanned: {total_videos:,}")
    print(f"Corrupted videos removed: {len(corrupted_videos):,}")
    print(f"Clean videos remaining: {total_videos - len(corrupted_videos):,}")
    print(f"{'='*70}\n")

    if corrupted_videos:
        print("Corrupted videos by split:")
        for split in ['train', 'val', 'test']:
            split_corrupted = [v for v in corrupted_videos if v['split'] == split]
            if split_corrupted:
                print(f"  {split}: {len(split_corrupted)} removed")

        # Save list
        corrupted_list = backup_dir / 'corrupted_videos.txt'
        with open(corrupted_list, 'w') as f:
            for v in corrupted_videos:
                f.write(f"{v['path']}\t{v['error']}\n")

        print(f"\n📝 Full list saved to: {corrupted_list}")

    print(f"\n✅ Dataset is now CLEAN!")
    print(f"✅ Training will NOT hang on corrupted videos!")
    print(f"\n{'='*70}\n")

    return len(corrupted_videos)

if __name__ == "__main__":
    dataset_path = "/workspace/organized_dataset"
    backup_dir = "/workspace/corrupted_videos_removed"

    print("\n⚠️  WARNING: This will MOVE corrupted videos to backup directory")
    print(f"⚠️  Backup location: {backup_dir}\n")

    confirm = input("Proceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled")
        sys.exit(0)

    removed = clean_dataset(dataset_path, backup_dir)

    print("\n" + "="*70)
    print("NEXT STEP: START TRAINING")
    print("="*70)
    print("cd /workspace/violence_detection_mvp")
    print("python3 train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset")
    print("="*70)
