#!/usr/bin/env python3
"""
Remove videos with corrupted h264 encoding that process slowly
These videos cause 10x slowdown during feature extraction
"""

import cv2
import sys
import time
from pathlib import Path
from tqdm import tqdm
import shutil
import warnings
import os

# Suppress h264 warnings
warnings.filterwarnings('ignore')
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

def test_video_speed(video_path, timeout_sec=2.0):
    """
    Test if video processes at acceptable speed
    Returns: (is_fast, processing_time)
    """
    try:
        start_time = time.time()

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return False, 0, "Cannot open"

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return False, 0, "0 frames"

        # Test reading 10 frames
        frames_to_test = min(10, total_frames)
        frame_indices = [int(i * total_frames / frames_to_test) for i in range(frames_to_test)]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                cap.release()
                return False, 0, "Cannot read frame"

            # Check if we've exceeded timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                cap.release()
                return False, elapsed, f"SLOW - {elapsed:.2f}s for {len(frame_indices)} frames"

        cap.release()
        elapsed = time.time() - start_time

        # If it took more than timeout_sec for 10 frames, it's too slow
        if elapsed > timeout_sec:
            return False, elapsed, f"SLOW - {elapsed:.2f}s"

        return True, elapsed, "OK"

    except Exception as e:
        return False, 0, f"Error: {str(e)}"

def clean_slow_videos(dataset_path, backup_dir, timeout_sec=2.0):
    """Remove slow-processing videos from dataset"""

    dataset_path = Path(dataset_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("REMOVING SLOW-PROCESSING VIDEOS")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Backup:  {backup_dir}")
    print(f"Timeout: {timeout_sec}s for 10 frames")
    print("="*80 + "\n")

    slow_videos = []
    total_videos = 0

    # Process all splits
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue

        print(f"\n{'='*80}")
        print(f"Processing {split.upper()}")
        print(f"{'='*80}\n")

        for class_name in ['violent', 'nonviolent']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            videos = list(class_dir.glob('*.mp4'))
            print(f"\n{class_name}: Testing {len(videos):,} videos...")

            for video in tqdm(videos, desc=f"{split}/{class_name}"):
                total_videos += 1
                is_fast, proc_time, error = test_video_speed(video, timeout_sec)

                if not is_fast:
                    # Move to backup
                    backup_class_dir = backup_dir / split / class_name
                    backup_class_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = backup_class_dir / video.name

                    shutil.move(str(video), str(backup_path))
                    slow_videos.append({
                        'path': str(video),
                        'split': split,
                        'class': class_name,
                        'time': proc_time,
                        'error': error
                    })

                    tqdm.write(f"  ✗ REMOVED: {video.name} ({error})")

    # Summary
    print(f"\n{'='*80}")
    print("CLEANUP COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos scanned: {total_videos:,}")
    print(f"Slow videos removed: {len(slow_videos):,}")
    print(f"Fast videos remaining: {total_videos - len(slow_videos):,}")
    print(f"{'='*80}\n")

    if slow_videos:
        print("Slow videos by split:")
        for split in ['train', 'val', 'test']:
            split_slow = [v for v in slow_videos if v['split'] == split]
            if split_slow:
                print(f"  {split}: {len(split_slow)} removed")

        # Calculate average slowdown
        avg_time = sum(v['time'] for v in slow_videos if v['time'] > 0) / max(1, len(slow_videos))
        print(f"\nAverage processing time of slow videos: {avg_time:.2f}s")
        print(f"Expected speedup: {avg_time / timeout_sec:.1f}x faster training")

        # Save list
        slow_list = backup_dir / 'slow_videos.txt'
        with open(slow_list, 'w') as f:
            for v in slow_videos:
                f.write(f"{v['path']}\t{v['time']:.2f}s\t{v['error']}\n")

        print(f"\n📝 Full list saved to: {slow_list}")

    print(f"\n✅ Dataset is now optimized for fast training!")
    print(f"✅ No more 10x slowdowns during feature extraction!")
    print(f"\n{'='*80}\n")

    return len(slow_videos)

if __name__ == "__main__":
    dataset_path = "/workspace/organized_dataset"
    backup_dir = "/workspace/slow_videos_removed"

    print("\n⚠️  WARNING: This will MOVE slow-processing videos to backup directory")
    print(f"⚠️  Backup location: {backup_dir}")
    print(f"⚠️  Videos that take >2s to read 10 frames will be removed\n")

    confirm = input("Proceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled")
        sys.exit(0)

    removed = clean_slow_videos(dataset_path, backup_dir, timeout_sec=2.0)

    print("\n" + "="*80)
    print("NEXT STEP: START TRAINING")
    print("="*80)
    print("cd /workspace/violence_detection_mvp")
    print("python3 train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset")
    print("="*80)
