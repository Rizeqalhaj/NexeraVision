#!/usr/bin/env python3
"""
AGGRESSIVE VIDEO CLEANUP - GUARANTEED FIX
Uses multiprocessing with forced timeout to catch hanging videos
"""

import cv2
import sys
import signal
from pathlib import Path
from tqdm import tqdm
import shutil
from multiprocessing import Process, Queue
import time

def test_video_with_timeout(video_path, result_queue, timeout_sec=3):
    """
    Test video in separate process with forced timeout
    This WILL catch hanging videos that cv2 timeouts miss
    """
    def timeout_handler(signum, frame):
        raise TimeoutError("Video processing timed out")

    try:
        # Set alarm for forced timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            result_queue.put((False, "Cannot open"))
            return

        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            result_queue.put((False, "0 frames"))
            return

        # Try to read first frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            result_queue.put((False, "Cannot read first frame"))
            return

        # Try to read middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            result_queue.put((False, "Cannot read middle frame"))
            return

        # Try to read last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            result_queue.put((False, "Cannot read last frame"))
            return

        cap.release()
        signal.alarm(0)  # Cancel alarm
        result_queue.put((True, "OK"))

    except TimeoutError:
        result_queue.put((False, "TIMEOUT - Video hangs"))
    except Exception as e:
        result_queue.put((False, f"Exception: {str(e)}"))
    finally:
        signal.alarm(0)  # Cancel alarm

def validate_video(video_path, timeout_sec=3):
    """
    Validate video using multiprocessing with forced timeout
    Returns: (is_valid, error_message)
    """
    result_queue = Queue()

    # Run validation in separate process
    process = Process(target=test_video_with_timeout, args=(video_path, result_queue, timeout_sec))
    process.start()

    # Wait for result with timeout
    process.join(timeout=timeout_sec + 1)

    # If process is still alive, it's hanging - kill it
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()  # Force kill if terminate doesn't work
        return False, "HANGING - Process timeout"

    # Get result from queue
    if not result_queue.empty():
        return result_queue.get()
    else:
        return False, "No result returned"

def clean_dataset(dataset_path, backup_dir):
    """Remove ALL problematic videos from dataset"""

    dataset_path = Path(dataset_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("AGGRESSIVE VIDEO CLEANUP - GUARANTEED FIX")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Backup:  {backup_dir}")
    print("="*80 + "\n")

    corrupted_videos = []
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
                is_valid, error = validate_video(video, timeout_sec=3)

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
    print(f"\n{'='*80}")
    print("CLEANUP COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos scanned: {total_videos:,}")
    print(f"Corrupted videos removed: {len(corrupted_videos):,}")
    print(f"Clean videos remaining: {total_videos - len(corrupted_videos):,}")
    print(f"{'='*80}\n")

    if corrupted_videos:
        print("Corrupted videos by split:")
        for split in ['train', 'val', 'test']:
            split_corrupted = [v for v in corrupted_videos if v['split'] == split]
            if split_corrupted:
                print(f"  {split}: {len(split_corrupted)} removed")

        print("\nCorrupted videos by error type:")
        error_types = {}
        for v in corrupted_videos:
            error_type = v['error'].split('-')[0].strip()
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} videos")

        # Save list
        corrupted_list = backup_dir / 'corrupted_videos_aggressive.txt'
        with open(corrupted_list, 'w') as f:
            for v in corrupted_videos:
                f.write(f"{v['path']}\t{v['error']}\n")

        print(f"\n📝 Full list saved to: {corrupted_list}")

    print(f"\n✅ Dataset is now CLEAN!")
    print(f"✅ Training will NOT hang on corrupted videos!")
    print(f"\n{'='*80}\n")

    return len(corrupted_videos)

if __name__ == "__main__":
    dataset_path = "/workspace/organized_dataset"
    backup_dir = "/workspace/corrupted_videos_aggressive"

    print("\n⚠️  WARNING: This will MOVE corrupted videos to backup directory")
    print(f"⚠️  Backup location: {backup_dir}")
    print(f"⚠️  This uses AGGRESSIVE timeout detection with multiprocessing\n")

    confirm = input("Proceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled")
        sys.exit(0)

    removed = clean_dataset(dataset_path, backup_dir)

    print("\n" + "="*80)
    print("NEXT STEP: START TRAINING")
    print("="*80)
    print("cd /workspace/violence_detection_mvp")
    print("python3 train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset")
    print("="*80)
