#!/usr/bin/env python3
"""
Custom Kinetics-700 Downloader - FIXED VERSION
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

# Fight/combat keywords for flexible matching
FIGHT_KEYWORDS = [
    "box", "fight", "punch", "kick", "wrestl", "martial", "combat",
    "slap", "headbutt", "karate", "judo", "taekwondo", "muay thai",
    "mma", "ufc", "capoeira", "fenc", "sword", "tai chi", "kung fu"
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

            # Debug: show columns and first few rows
            print(f"  ✅ {split}: {len(df)} videos")
            print(f"     Columns: {list(df.columns)}")

        except Exception as e:
            print(f"  ❌ Failed to download {split}: {e}")
            annotations[split] = pd.DataFrame()

    return annotations

def find_label_column(df):
    """Find which column contains the class labels"""
    possible_names = ['label', 'class', 'category', 'action', 'activity']

    for col in df.columns:
        if col.lower() in possible_names:
            return col

    # If not found, return first string column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col

    return None

def filter_fight_classes(annotations, fight_keywords):
    """Filter annotations for fight/combat classes using flexible keyword matching"""
    print(f"\n🥊 Filtering for fight/combat videos using keywords...")
    print(f"Keywords: {', '.join(fight_keywords)}")
    print("")

    all_fight_videos = []
    found_classes = set()

    for split, df in annotations.items():
        if df.empty:
            continue

        # Find label column
        label_col = find_label_column(df)

        if label_col is None:
            print(f"  ❌ {split}: Could not find label column")
            print(f"     Available columns: {list(df.columns)}")
            continue

        print(f"  {split}: Using '{label_col}' column for class labels")

        # Show sample of classes
        unique_classes = df[label_col].unique()
        print(f"  {split}: Found {len(unique_classes)} unique classes")
        print(f"  Sample classes: {list(unique_classes[:10])}")

        # Filter using keywords (case-insensitive)
        fight_mask = df[label_col].str.lower().str.contains('|'.join(fight_keywords), na=False)
        fight_df = df[fight_mask].copy()

        if len(fight_df) > 0:
            fight_df['split'] = split
            all_fight_videos.append(fight_df)

            # Track found classes
            for cls in fight_df[label_col].unique():
                found_classes.add(cls)

            print(f"  ✅ {split}: {len(fight_df)} fight videos found")
        else:
            print(f"  ⚠️  {split}: No fight videos found")

    if not all_fight_videos:
        print("\n❌ No fight videos found in any split!")
        print("\nTry checking the CSV files manually to see available classes.")
        return pd.DataFrame(), None

    combined = pd.concat(all_fight_videos, ignore_index=True)

    print(f"\n✅ Total fight videos: {len(combined)}")
    print(f"\nFight classes found ({len(found_classes)}):")
    for cls in sorted(found_classes):
        count = combined[combined[label_col] == cls].shape[0]
        print(f"  - {cls}: {count} videos")
    print("")

    return combined, label_col

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

    # yt-dlp command - simplified for better compatibility
    cmd = [
        'yt-dlp',
        '--quiet',
        '--no-warnings',
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

def download_videos_parallel(fight_videos, label_col, output_dir, num_workers=8):
    """Download videos in parallel"""
    print(f"\n🚀 Starting parallel download with {num_workers} workers...")
    print(f"📁 Output directory: {output_dir}")
    print("")

    # Prepare download arguments
    download_args = []

    # Check required columns
    required_cols = ['youtube_id', 'time_start', 'time_end']
    missing_cols = [col for col in required_cols if col not in fight_videos.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(fight_videos.columns)}")
        return {'success': 0, 'failed': 0, 'already_exists': 0}

    for _, row in fight_videos.iterrows():
        download_args.append((
            row['youtube_id'],
            row[label_col],
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
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode - show all classes without downloading')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("KINETICS-700 FIGHT VIDEOS DOWNLOADER (FIXED)")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Debug mode: {args.debug}")
    print("")

    # Step 1: Download annotations
    annotations = download_annotations(output_dir)

    # Step 2: Filter for fight classes
    fight_videos, label_col = filter_fight_classes(annotations, FIGHT_KEYWORDS)

    if fight_videos.empty:
        print("❌ No videos to download!")

        # Debug: save all classes to file for manual inspection
        print("\n💡 Saving all classes to 'all_kinetics_classes.txt' for inspection...")
        with open(output_dir / 'all_kinetics_classes.txt', 'w') as f:
            for split, df in annotations.items():
                if not df.empty:
                    label_col_temp = find_label_column(df)
                    if label_col_temp:
                        f.write(f"\n=== {split.upper()} ===\n")
                        for cls in sorted(df[label_col_temp].unique()):
                            count = (df[label_col_temp] == cls).sum()
                            f.write(f"{cls}: {count}\n")

        print("Check 'all_kinetics_classes.txt' to see all available classes")
        return 1

    if args.debug:
        print("\n🔍 DEBUG MODE - Not downloading, just showing what would be downloaded")
        return 0

    # Step 3: Download videos
    stats = download_videos_parallel(fight_videos, label_col, output_dir, args.num_workers)

    # Final statistics
    total_videos = len(fight_videos)

    print("")
    print("="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"✅ Successfully downloaded: {stats['success']}")
    print(f"📁 Already existed: {stats['already_exists']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📊 Total videos: {stats['success'] + stats['already_exists']}")

    if total_videos > 0:
        print(f"📈 Success rate: {stats['success'] / total_videos * 100:.1f}%")

    print("")

    # Count actual files
    video_count = sum(1 for _ in output_dir.rglob('*.mp4'))
    print(f"💾 Videos on disk: {video_count}")
    print(f"📁 Location: {output_dir}")
    print("")

    return 0

if __name__ == "__main__":
    sys.exit(main())
