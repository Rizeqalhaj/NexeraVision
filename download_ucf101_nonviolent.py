#!/usr/bin/env python3
"""
UCF101 Non-Violent Video Dataset Downloader
Downloads and extracts non-violent action categories from UCF101 dataset

UCF101 Dataset:
- 13,320 videos total
- 101 action categories
- ~60+ non-violent categories expected
- Expected yield: 8,000-10,000 non-violence videos

Requirements:
    pip install kaggle
    kaggle API token in ~/.kaggle/kaggle.json
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import List, Set

# UCF101 Action Categories - Non-Violent Only
NON_VIOLENT_CATEGORIES = [
    # Sports (Non-Contact)
    'Basketball', 'BasketballDunk', 'Billiards', 'Bowling', 'CliffDiving',
    'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
    'HammerThrow', 'HighJump', 'HorseRiding', 'HorseRace', 'IceDancing',
    'JavelinThrow', 'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking',
    'LongJump', 'Nunchucks', 'ParallelBars', 'PoleVault', 'Rafting',
    'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'Shotput', 'SkateBoarding',
    'Skiing', 'Skijet', 'SoccerJuggling', 'StillRings', 'SumoWrestling',
    'Surfing', 'Swing', 'TableTennisShot', 'TennisSwing', 'ThrowDiscus',
    'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking',
    'WalkingWithDog', 'YoYo',

    # Daily Activities
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling',
    'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Biking', 'Billiards',
    'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling',
    'BrushingTeeth', 'CleanAndJerk', 'CuttingInKitchen', 'Drumming',
    'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrontCrawl',
    'Haircut', 'HandstandPushups', 'HandstandWalking', 'HeadMassage',
    'Knitting', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor',
    'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol',
    'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
    'PlayingTabla', 'PlayingViolin', 'PullUps', 'Punch', 'PushUps',
    'Salsa', 'ShavingBeard', 'SkyDiving', 'SoccerPenalty', 'TaiChi',
    'WritingOnBoard',
]


def check_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    print(f"\n{'='*70}")
    print("Checking Kaggle API Setup")
    print(f"{'='*70}\n")

    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'

    if not kaggle_config.exists():
        print("❌ Kaggle API token not found!")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    print("✅ Kaggle API token found")

    # Check if kaggle is installed
    try:
        import kaggle
        print("✅ Kaggle package installed")
        return True
    except ImportError:
        print("❌ Kaggle package not installed")
        print("Install with: pip install kaggle")
        return False


def download_ucf101_dataset(output_dir: Path):
    """Download UCF101 dataset from Kaggle"""
    print(f"\n{'='*70}")
    print("Downloading UCF101 Dataset from Kaggle")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # UCF101 dataset identifier on Kaggle
    dataset = "pevogam/ucf101"

    print(f"📥 Downloading dataset: {dataset}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⏳ This may take 10-30 minutes (dataset is ~6.5 GB)...\n")

    try:
        # Download using kaggle CLI
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', dataset,
            '-p', str(output_dir),
            '--unzip'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        print(f"\n✅ Dataset downloaded successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed!")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def extract_nonviolent_videos(ucf_dir: Path, output_dir: Path):
    """Extract non-violent videos from UCF101 dataset"""
    print(f"\n{'='*70}")
    print("Extracting Non-Violent Videos")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find UCF101 video directory
    # UCF101 structure: UCF-101/category/video.avi
    ucf_videos_dir = None
    for candidate in ['UCF-101', 'ucf101', 'videos']:
        test_dir = ucf_dir / candidate
        if test_dir.exists():
            ucf_videos_dir = test_dir
            break

    if not ucf_videos_dir:
        print(f"❌ UCF-101 video directory not found in {ucf_dir}")
        print("Searching for video files...")
        # Try to find any .avi files
        ucf_videos_dir = ucf_dir

    print(f"📂 Source directory: {ucf_videos_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🎯 Target categories: {len(NON_VIOLENT_CATEGORIES)}\n")

    total_copied = 0
    total_skipped = 0
    category_counts = {}

    for category in NON_VIOLENT_CATEGORIES:
        category_dir = ucf_videos_dir / category

        if not category_dir.exists():
            print(f"⚠️  Category not found: {category}")
            total_skipped += 1
            continue

        # Get all videos in category
        videos = list(category_dir.glob('*.avi'))

        if not videos:
            print(f"⚠️  No videos found in: {category}")
            total_skipped += 1
            continue

        # Create output category directory
        output_category_dir = output_dir / category
        output_category_dir.mkdir(parents=True, exist_ok=True)

        # Copy videos
        copied = 0
        for video in videos:
            dest = output_category_dir / video.name
            if not dest.exists():
                shutil.copy2(video, dest)
                copied += 1

        total_copied += copied
        category_counts[category] = copied
        print(f"✓ {category:30s}: {copied:4d} videos")

    print(f"\n{'='*70}")
    print("Extraction Complete")
    print(f"{'='*70}")
    print(f"✅ Total videos extracted: {total_copied:,}")
    print(f"✅ Categories processed: {len(category_counts)}")
    print(f"⚠️  Categories skipped: {total_skipped}")
    print(f"{'='*70}\n")

    # Save category counts
    stats_file = output_dir / 'extraction_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'total_videos': total_copied,
            'categories_processed': len(category_counts),
            'categories_skipped': total_skipped,
            'category_counts': category_counts
        }, f, indent=2)

    print(f"📊 Statistics saved to: {stats_file}")

    return total_copied, category_counts


def verify_extraction(output_dir: Path):
    """Verify extracted videos and show statistics"""
    print(f"\n{'='*70}")
    print("Verification Report")
    print(f"{'='*70}\n")

    total_videos = 0
    total_size = 0
    categories = []

    for category_dir in sorted(output_dir.iterdir()):
        if category_dir.is_dir() and category_dir.name != '__pycache__':
            videos = list(category_dir.glob('*.avi'))
            video_count = len(videos)
            category_size = sum(v.stat().st_size for v in videos)

            total_videos += video_count
            total_size += category_size
            categories.append(category_dir.name)

    print(f"📊 Dataset Statistics:")
    print(f"   Videos:     {total_videos:,}")
    print(f"   Categories: {len(categories)}")
    print(f"   Total size: {total_size / (1024**3):.2f} GB")
    print(f"\n{'='*70}\n")

    # Show current dataset balance
    print("📈 Dataset Balance Status:")
    print(f"   Violence:     14,000 videos")
    print(f"   Non-violence: 10,454 + {total_videos:,} = {10454 + total_videos:,} videos")

    if (10454 + total_videos) >= 14000:
        print(f"   ✅ BALANCED! ({10454 + total_videos:,} ≥ 14,000)")
        print(f"   📊 Final dataset: ~{14000 + 10454 + total_videos:,} total videos")
        print(f"   🎯 Expected accuracy: 93-95%")
    else:
        needed = 14000 - (10454 + total_videos)
        print(f"   ⚠️  Still need {needed:,} more non-violence videos")

    print(f"\n{'='*70}\n")


def main():
    print("="*70)
    print("UCF101 Non-Violent Video Dataset Extractor")
    print("="*70)
    print("Goal: Extract 8,000-10,000 non-violence videos from UCF101")
    print("Current: 14K violence, 10,454 non-violence (need 3,546 more)\n")

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python download_ucf101_nonviolent.py [download_dir] [output_dir]")
            print()
            print("Arguments:")
            print("  download_dir: Where to download UCF101 (default: ./ucf101_download)")
            print("  output_dir:   Where to save extracted videos (default: ./ucf101_nonviolent)")
            print()
            print("Examples:")
            print("  python download_ucf101_nonviolent.py")
            print("  python download_ucf101_nonviolent.py ./data/ucf101 ./data/nonviolent")
            sys.exit(0)

        download_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('./ucf101_nonviolent')
    else:
        download_dir = Path('./ucf101_download')
        output_dir = Path('./ucf101_nonviolent')

    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Download directory: {download_dir.absolute()}")
    print(f"  Output directory:   {output_dir.absolute()}")
    print(f"  Non-violent categories: {len(NON_VIOLENT_CATEGORIES)}")
    print(f"{'='*70}")

    # Check Kaggle setup
    if not check_kaggle_setup():
        sys.exit(1)

    confirm = input("\nProceed with download and extraction? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Download dataset
    print("\n" + "="*70)
    print("PHASE 1: Download UCF101 Dataset")
    print("="*70)

    if not download_ucf101_dataset(download_dir):
        print("\n❌ Download failed! Check your Kaggle API setup.")
        sys.exit(1)

    # Extract non-violent videos
    print("\n" + "="*70)
    print("PHASE 2: Extract Non-Violent Videos")
    print("="*70)

    total_videos, category_counts = extract_nonviolent_videos(download_dir, output_dir)

    if total_videos == 0:
        print("\n❌ No videos extracted! Check dataset structure.")
        sys.exit(1)

    # Verify extraction
    verify_extraction(output_dir)

    # Show next steps
    print(f"{'='*70}")
    print("Next Steps")
    print(f"{'='*70}")
    print(f"1. Organize dataset for training:")
    print(f"   python organize_final_dataset.py {output_dir} ./organized_dataset")
    print()
    print(f"2. Start training with balanced dataset:")
    print(f"   cd violence_detection_mvp")
    print(f"   python train_rtx5000_dual_IMPROVED.py --dataset-path ../organized_dataset")
    print()
    print(f"Expected results:")
    print(f"  - Balanced dataset: ~{14000 + 10454 + total_videos:,} total videos")
    print(f"  - Training accuracy: 93-95%")
    print(f"  - Models saved to: violence_detection_mvp/models/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
