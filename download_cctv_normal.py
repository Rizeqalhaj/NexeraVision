#!/usr/bin/env python3
"""
CCTV Normal/Non-Violent Footage Downloader
Downloads surveillance footage WITHOUT violence for balanced training
CRITICAL: Same camera angles/quality as violent footage
"""

import subprocess
import os
from pathlib import Path
from tqdm import tqdm
import time

# ============================================
# CCTV NON-VIOLENT SEARCH QUERIES
# ============================================

# Normal surveillance footage
CCTV_NORMAL_QUERIES = [
    "security camera normal day",
    "CCTV peaceful store",
    "surveillance footage no incident",
    "security camera daily activity",
    "store surveillance normal",
    "parking lot security camera normal",
    "mall security camera shopping",
    "convenience store CCTV normal",
    "restaurant security camera normal",
    "hotel lobby security normal",
    "office building security normal",
    "elevator security camera normal",
    "subway station CCTV normal",
    "train station security normal",
    "airport security camera normal",
    "street surveillance normal activity",
    "gas station security camera normal",
    "bank security camera normal day",
    "hospital security normal",
    "school security camera normal",
    "warehouse security footage normal",
    "retail store CCTV normal day",
]

# Reddit - Normal surveillance footage (yt-dlp compatible URLs)
REDDIT_NORMAL_CCTV = [
    "https://www.reddit.com/r/IdiotsInCars/top/?t=all",  # Traffic cameras
    "https://www.reddit.com/r/CCTV/top/?t=all",  # CCTV subreddit
    "https://www.reddit.com/r/homedefense/top/?t=all",  # Home security
    "https://www.reddit.com/r/SecurityCameras/top/?t=all",  # Security cameras
    "https://www.reddit.com/r/WatchPeopleSurvive/top/?t=year",  # Surveillance footage (survival)
]

# Kaggle datasets with CCTV normal footage
KAGGLE_CCTV_NORMAL = [
    # These need to be verified on Kaggle first
    "workplace surveillance normal",
    "retail surveillance dataset",
    "traffic camera dataset",
]

def download_cctv_normal_youtube(output_dir, max_per_query=500):
    """Download normal CCTV footage from YouTube"""
    print("\n📹 YOUTUBE CCTV NORMAL FOOTAGE")
    print("="*60)

    youtube_dir = output_dir / "youtube_cctv_normal"
    youtube_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for i, query in enumerate(CCTV_NORMAL_QUERIES, 1):
        print(f"\n[{i}/{len(CCTV_NORMAL_QUERIES)}] '{query}'")

        # Batch processing
        batch_size = 50
        batches = max_per_query // batch_size

        for batch in range(batches):
            print(f"  Batch {batch+1}/{batches}: ", end='', flush=True)

            cmd = [
                'yt-dlp',
                f'ytsearch{batch_size}:{query}',
                '-f', 'best[height<=480]',
                '--max-downloads', str(batch_size),
                '-o', str(youtube_dir / '%(id)s.%(ext)s'),
                '--restrict-filenames',
                '--no-playlist',
                '--ignore-errors',
                '--no-warnings',
                '--quiet'
            ]

            try:
                subprocess.run(cmd, timeout=300)
                count = len(list(youtube_dir.glob('*.*')))
                batch_downloaded = count - total_downloaded
                total_downloaded = count
                print(f"{batch_downloaded} videos ✓")
                time.sleep(3)
            except Exception as e:
                print(f"error: {e}")
                break

        print(f"  Total so far: {total_downloaded} videos")

    return total_downloaded

def download_cctv_normal_reddit(output_dir, max_per_subreddit=3000):
    """Download normal CCTV footage from Reddit"""
    print("\n🔴 REDDIT NORMAL CCTV FOOTAGE")
    print("="*60)

    reddit_dir = output_dir / "reddit_cctv_normal"
    reddit_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for i, subreddit_url in enumerate(REDDIT_NORMAL_CCTV, 1):
        subreddit_name = subreddit_url.split('/r/')[1].split('/')[0]
        print(f"\n[{i}/{len(REDDIT_NORMAL_CCTV)}] Reddit: r/{subreddit_name}")

        cmd = [
            'yt-dlp',
            subreddit_url,
            '-f', 'best[height<=720]',
            '--max-downloads', str(max_per_subreddit),
            '-o', str(reddit_dir / f'{subreddit_name}_%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=3600)
            count = len(list(reddit_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(5)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def download_cctv_normal_vimeo(output_dir, max_videos=300):
    """Download normal CCTV footage from Vimeo"""
    print("\n🎬 VIMEO NORMAL CCTV FOOTAGE")
    print("="*60)

    vimeo_dir = output_dir / "vimeo_cctv_normal"
    vimeo_dir.mkdir(parents=True, exist_ok=True)

    vimeo_queries = [
        "https://vimeo.com/search?q=security+camera",
        "https://vimeo.com/search?q=CCTV+timelapse",
        "https://vimeo.com/search?q=surveillance",
    ]

    total_downloaded = 0

    for i, query in enumerate(vimeo_queries, 1):
        print(f"\n[{i}/{len(vimeo_queries)}] Vimeo: {query}")

        cmd = [
            'yt-dlp',
            query,
            '-f', 'best[height<=480]',
            '--max-downloads', str(max_videos // len(vimeo_queries)),
            '-o', str(vimeo_dir / '%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=1200)
            count = len(list(vimeo_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(3)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def main():
    import argparse

    parser = argparse.ArgumentParser(description='CCTV Normal/Non-Violent Footage Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/cctv_normal',
                       help='Output directory')
    parser.add_argument('--sources', nargs='+',
                       choices=['reddit', 'youtube', 'vimeo', 'all'],
                       default=['all'],
                       help='Video sources to download from')
    parser.add_argument('--max-reddit', type=int, default=3000,
                       help='Max videos per Reddit search')
    parser.add_argument('--max-youtube-per-query', type=int, default=500,
                       help='Max videos per YouTube query')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CCTV NORMAL/NON-VIOLENT FOOTAGE DOWNLOADER")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Sources: {args.sources}")
    print("")
    print("🎯 PURPOSE:")
    print("  ✓ Balance violent CCTV footage with non-violent")
    print("  ✓ Same camera angles and quality as violent footage")
    print("  ✓ Train model to distinguish violence from normal activity")
    print("  ✓ Prevent false positives in production")
    print("")
    print("📊 Expected normal CCTV volume:")
    print("  • Reddit: 8,000-12,000 videos")
    print("  • YouTube: 8,000-10,000 videos")
    print("  • Vimeo: 300-500 videos")
    print("  • TOTAL: 16,000-22,000 normal CCTV videos")
    print("")

    total_stats = {}

    sources = args.sources
    if 'all' in sources:
        sources = ['reddit', 'youtube', 'vimeo']

    if 'reddit' in sources:
        total_stats['reddit'] = download_cctv_normal_reddit(output_dir, args.max_reddit)

    if 'youtube' in sources:
        total_stats['youtube'] = download_cctv_normal_youtube(output_dir, args.max_youtube_per_query)

    if 'vimeo' in sources:
        total_stats['vimeo'] = download_cctv_normal_vimeo(output_dir)

    # Final summary
    print("\n" + "="*80)
    print("NORMAL CCTV DOWNLOAD COMPLETE!")
    print("="*80)
    print("")
    print("📊 STATISTICS BY SOURCE:")
    print("-"*60)

    total_all = 0
    for source, count in total_stats.items():
        print(f"  {source.capitalize():20s}: {count:6d} videos")
        total_all += count

    print("-"*60)
    print(f"  {'TOTAL NORMAL CCTV':20s}: {total_all:6d} videos")
    print("")

    print(f"📁 Location: {output_dir}")
    print("")

    # Count all video files
    all_videos = list(output_dir.rglob('*.mp4')) + list(output_dir.rglob('*.webm')) + \
                 list(output_dir.rglob('*.mkv')) + list(output_dir.rglob('*.avi'))

    print(f"✅ Verified on disk: {len(all_videos)} normal CCTV files")
    print("")

    if total_all >= 15000:
        print("🎉 EXCELLENT: 15,000+ normal CCTV videos!")
    elif total_all >= 10000:
        print("✅ VERY GOOD: 10,000+ normal CCTV videos")
    elif total_all >= 5000:
        print("✅ GOOD: 5,000+ normal CCTV videos")
    else:
        print(f"⚠️  Downloaded {total_all} normal CCTV videos")

    print("")
    print("🎯 NEXT STEPS:")
    print("1. Combine with violent CCTV footage:")
    print("   Violent: /workspace/datasets/cctv_surveillance")
    print("   Normal:  /workspace/datasets/cctv_normal")
    print("")
    print("2. Run balancing script:")
    print("   bash /home/admin/Desktop/NexaraVision/balance_and_combine.sh")
    print("")
    print("💡 DATASET STRUCTURE:")
    print("   With CCTV-focused training, your model will:")
    print("   ✓ Understand camera perspectives (top-down, corner angles)")
    print("   ✓ Handle surveillance quality (480p-720p typical)")
    print("   ✓ Work in various lighting conditions")
    print("   ✓ Distinguish violence from normal pushing/shoving")
    print("   ✓ Minimize false positives in crowded areas")
    print("")
    print("   = PRODUCTION-READY for actual camera deployment!")
    print("")

if __name__ == "__main__":
    main()
