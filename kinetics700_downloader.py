#!/usr/bin/env python3
"""
Custom Kinetics-700 Downloader
Downloads fight/combat videos from Kinetics-700 dataset using YouTube video IDs
"""

import os
import sys
import subprocess
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time

# Kinetics-700 annotation URLs
KINETICS_700_TRAIN_URL = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020/train.csv"
KINETICS_700_VAL_URL = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020/val.csv"
KINETICS_700_TEST_URL = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020/test.csv"

# Fight/combat classes to download
FIGHT_CLASSES = [
    "boxing",
    "wrestling",
    "punching person (boxing)",
    "side kick",
    "high kick",
    "drop kicking",
    "arm wrestling",
    "capoeira",
    "fencing (sport)",
    "kickboxing",
    "martial arts",
    "slapping",
    "headbutting",
    "punching bag",
    "sword fighting",
    "tai chi",
    "training with punching bag",
    "catching or throwing baseball",
    "catching or throwing softball"
]

def download_annotations(output_dir):
    """Download Kinetics-700 annotation CSV files"""
    print("📥 Downloading Kinetics-700 annotations...")

    annotations = {}

    for split, url in [("train", KINETICS_700_TRAIN_URL),
                       ("val", KINETICS_700_VAL_URL),
                       ("test", KINETICS_700_TEST_URL)]:
        print(f"  Downloading {split}.csv...")
        try:
            response = requests.get(url, timeout=30)
            csv_path = output_dir / f"{split}.csv"
            with open(csv_path, 'wb') as f:
                f.write(response.content)

            df = pd.read_csv(csv_path)
            annotations[split] = df
            print(f"  ✅ {split}: {len(df)} videos")
        except Exception as e:
            print(f"  ❌ Failed to download {split}: {e}")
            annotations[split] = pd.DataFrame()

    return annotations

def filter_fight_classes(annotations, fight_classes):
    """Filter annotations for fight/combat classes only"""
    print(f"\n🥊 Filtering for {len(fight_classes)} fight classes...")

    all_fight_videos = []

    for split, df in annotations.items():
        if df.empty:
            continue

        # Filter for fight classes
        fight_df = df[df['label'].isin(fight_classes)]

        if len(fight_df) > 0:
            fight_df['split'] = split
            all_fight_videos.append(fight_df)
            print(f"  {split}: {len(fight_df)} fight videos")

    if not all_fight_videos:
        print("❌ No fight videos found!")
        return pd.DataFrame()

    combined = pd.concat(all_fight_videos, ignore_index=True)
    print(f"\n✅ Total fight videos: {len(combined)}")

    return combined

def download_video(args):
    """Download a single video using yt-dlp"""
    youtube_id, label, time_start, time_end, output_dir = args

    # Create class directory
    class_dir = output_dir / label.replace('/', '_').replace(' ', '_')
    class_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_file = class_dir / f"{youtube_id}_{int(time_start)}_{int(time_end)}.mp4"

    # Skip if already downloaded
    if output_file.exists():
        return True, youtube_id, "already_exists"

    # YouTube URL
    video_url = f"https://www.youtube.com/watch?v={youtube_id}"

    # yt-dlp command with trimming
    cmd = [
        'yt-dlp',
        '--quiet',
        '--no-warnings',
        f'--download-sections',
        f'*{time_start}-{time_end}',
        '-f', 'best[height<=480]',  # 480p max to save bandwidth
        '-o', str(output_file),
        video_url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)

        if result.returncode == 0 and output_file.exists():
            return True, youtube_id, "success"
        else:
            return False, youtube_id, "download_failed"

    except subprocess.TimeoutExpired:
        return False, youtube_id, "timeout"
    except Exception as e:
        return False, youtube_id, str(e)

def download_videos_parallel(fight_videos, output_dir, num_workers=8):
    """Download videos in parallel"""
    print(f"\n🚀 Starting parallel download with {num_workers} workers...")
    print(f"📁 Output directory: {output_dir}")
    print("")

    # Prepare download arguments
    download_args = []
    for _, row in fight_videos.iterrows():
        download_args.append((
            row['youtube_id'],
            row['label'],
            row['time_start'],
            row['time_end'],
            output_dir
        ))

    # Track statistics
    stats = {
        'success': 0,
        'failed': 0,
        'already_exists': 0
    }

    # Download with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_video, args) for args in download_args]

        with tqdm(total=len(download_args), desc="Downloading videos") as pbar:
            for future in concurrent.futures.as_completed(futures):
                success, youtube_id, status = future.result()

                if status == "already_exists":
                    stats['already_exists'] += 1
                elif success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1

                pbar.update(1)
                pbar.set_postfix({
                    'success': stats['success'],
                    'failed': stats['failed'],
                    'exists': stats['already_exists']
                })

    return stats

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Kinetics-700 Fight Videos Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/phase3/kinetics',
                       help='Output directory for videos')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel download workers')
    parser.add_argument('--classes-file', help='File with custom class list (one per line)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load custom classes if provided
    fight_classes = FIGHT_CLASSES
    if args.classes_file and os.path.exists(args.classes_file):
        with open(args.classes_file, 'r') as f:
            fight_classes = [line.strip() for line in f if line.strip()]
        print(f"📋 Loaded {len(fight_classes)} custom classes from {args.classes_file}")

    print("="*80)
    print("KINETICS-700 FIGHT VIDEOS DOWNLOADER")
    print("="*80)
    print(f"Fight classes: {len(fight_classes)}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {args.num_workers}")
    print("")

    # Step 1: Download annotations
    annotations = download_annotations(output_dir)

    # Step 2: Filter for fight classes
    fight_videos = filter_fight_classes(annotations, fight_classes)

    if fight_videos.empty:
        print("❌ No videos to download!")
        return 1

    # Step 3: Download videos
    stats = download_videos_parallel(fight_videos, output_dir, args.num_workers)

    # Final statistics
    print("")
    print("="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"✅ Successfully downloaded: {stats['success']}")
    print(f"📁 Already existed: {stats['already_exists']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📊 Total videos: {stats['success'] + stats['already_exists']}")
    print(f"📈 Success rate: {stats['success'] / len(fight_videos) * 100:.1f}%")
    print("")

    # Count actual files
    video_count = sum(1 for _ in output_dir.rglob('*.mp4'))
    print(f"💾 Videos on disk: {video_count}")
    print(f"📁 Location: {output_dir}")
    print("")

    return 0

if __name__ == "__main__":
    sys.exit(main())
