#!/usr/bin/env python3
"""
Clean Corrupted Videos from Dataset - OPTIMIZED FOR 192 CPU CORES

Identifies and removes videos that fail to:
1. Open with cv2
2. Have sufficient frames
3. Decode properly

Uses parallel processing with 192 workers for maximum speed.
"""

import cv2
from pathlib import Path
from tqdm import tqdm
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def check_video(video_path):
    """Check if video is readable and has enough frames"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 20:
            cap.release()
            return False, f"Only {total_frames} frames"

        # Try to read first and last frames
        ret1, _ = cap.read()
        if not ret1:
            cap.release()
            return False, "Cannot read first frame"

        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret2, _ = cap.read()
        if not ret2:
            cap.release()
            return False, "Cannot read last frame"

        cap.release()
        return True, None

    except Exception as e:
        return False, f"Error: {e}"

def check_video_wrapper(video_path):
    """Wrapper for parallel processing"""
    is_valid, error = check_video(video_path)
    return video_path, is_valid, error

def clean_dataset(dataset_path, remove_corrupted=False, num_workers=None):
    """Scan dataset and optionally remove corrupted videos - PARALLEL"""

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    dataset_path = Path(dataset_path)

    results = {
        'train_violent': {'total': 0, 'corrupted': [], 'valid': 0},
        'train_nonviolent': {'total': 0, 'corrupted': [], 'valid': 0},
        'val_violent': {'total': 0, 'corrupted': [], 'valid': 0},
        'val_nonviolent': {'total': 0, 'corrupted': [], 'valid': 0},
        'test_violent': {'total': 0, 'corrupted': [], 'valid': 0},
        'test_nonviolent': {'total': 0, 'corrupted': [], 'valid': 0},
    }

    print("="*80)
    print("SCANNING DATASET FOR CORRUPTED VIDEOS (192-CORE PARALLEL)")
    print("="*80)
    print(f"CPU Workers: {num_workers}")
    print()

    for split in ['train', 'val', 'test']:
        for class_name in ['Violent', 'NonViolent']:
            key = f'{split}_{class_name.lower()}'
            video_dir = dataset_path / split / class_name

            if not video_dir.exists():
                print(f"⚠️  Directory not found: {video_dir}")
                continue

            # Check for all video formats
            videos = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi')) + list(video_dir.glob('*.webm'))
            results[key]['total'] = len(videos)

            print(f"\n📁 {split}/{class_name}: {len(videos)} videos")

            # PARALLEL PROCESSING WITH 192 WORKERS
            print(f"  🚀 Checking with {num_workers} parallel workers...")

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(check_video_wrapper, video): video for video in videos}

                for future in tqdm(as_completed(futures), total=len(videos), desc=f"  Scanning"):
                    video_path, is_valid, error = future.result()

                    if is_valid:
                        results[key]['valid'] += 1
                    else:
                        results[key]['corrupted'].append((video_path, error))

    # Print summary
    print("\n" + "="*80)
    print("SCAN RESULTS")
    print("="*80)

    total_corrupted = 0
    total_videos = 0

    for key, data in results.items():
        split, class_name = key.split('_')
        corrupted_count = len(data['corrupted'])
        total_corrupted += corrupted_count
        total_videos += data['total']

        if corrupted_count > 0:
            print(f"\n⚠️  {split}/{class_name}:")
            print(f"   Total: {data['total']}")
            print(f"   Valid: {data['valid']}")
            print(f"   Corrupted: {corrupted_count} ({corrupted_count/data['total']*100:.1f}%)")
        else:
            print(f"\n✅ {split}/{class_name}: All {data['total']} videos valid")

    print(f"\n{'='*80}")
    print(f"OVERALL: {total_corrupted}/{total_videos} corrupted ({total_corrupted/total_videos*100:.1f}%)")
    print(f"{'='*80}")

    # Optionally remove corrupted videos
    if remove_corrupted and total_corrupted > 0:
        print(f"\n⚠️  REMOVING {total_corrupted} CORRUPTED VIDEOS...")

        corrupted_dir = dataset_path / 'corrupted_videos'
        corrupted_dir.mkdir(exist_ok=True)

        for key, data in results.items():
            for video_path, error in data['corrupted']:
                # Move to corrupted directory instead of deleting
                dest = corrupted_dir / video_path.name
                shutil.move(str(video_path), str(dest))

        print(f"✅ Moved to: {corrupted_dir}")

    return results

if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Training"  # Updated default path
    remove = "--no-remove" not in sys.argv  # AUTO REMOVE BY DEFAULT

    print(f"Dataset: {dataset_path}")
    print(f"Remove corrupted: {remove} (AUTO ENABLED)")
    print()

    results = clean_dataset(dataset_path, remove_corrupted=remove)

    if not remove:
        print("\n" + "="*80)
        print("Note: Use --no-remove flag to skip removal")
        print("="*80)
