#!/usr/bin/env python3
"""
Multi-Platform Fight Video Downloader
Downloads fight videos from alternative platforms (not YouTube)
Supports: Vimeo, Dailymotion, Reddit, Twitter, Instagram, Bilibili, Internet Archive
"""

import subprocess
import os
from pathlib import Path
from tqdm import tqdm
import time

# ============================================
# PLATFORM-SPECIFIC SEARCH QUERIES
# ============================================

# Vimeo - High-quality MMA/boxing content
VIMEO_QUERIES = [
    "vimeo.com/search?q=UFC+fight",
    "vimeo.com/search?q=MMA+knockout",
    "vimeo.com/search?q=boxing+match",
    "vimeo.com/search?q=kickboxing",
    "vimeo.com/search?q=muay+thai+fight",
]

# Dailymotion - Popular sports platform
DAILYMOTION_QUERIES = [
    "UFC fight highlights",
    "MMA knockouts",
    "Boxing matches",
    "Street fight compilation",
    "Wrestling highlights",
    "Combat sports",
]

# Reddit - r/fightporn, r/StreetFights, r/MMA (yt-dlp compatible URLs)
REDDIT_SUBREDDITS = [
    "https://www.reddit.com/r/fightporn/top/?t=all",
    "https://www.reddit.com/r/StreetFights/top/?t=all",
    "https://www.reddit.com/r/MMA/top/?t=year",
    "https://www.reddit.com/r/PublicFreakout/top/?t=year",
    "https://www.reddit.com/r/ActualFreakouts/top/?t=all",
    "https://www.reddit.com/r/DocumentedFights/top/?t=all",
]

# Internet Archive - Public domain fight footage
ARCHIVE_SEARCHES = [
    "boxing match",
    "wrestling match",
    "martial arts demonstration",
    "UFC fight",
    "combat sports",
]

def download_from_vimeo(output_dir, max_videos=500):
    """Download fight videos from Vimeo"""
    print("\n🎬 VIMEO DOWNLOADS")
    print("="*60)

    vimeo_dir = output_dir / "vimeo"
    vimeo_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for i, query in enumerate(VIMEO_QUERIES, 1):
        print(f"\n[{i}/{len(VIMEO_QUERIES)}] Vimeo: {query}")

        cmd = [
            'yt-dlp',
            query,
            '-f', 'best[height<=480]',
            '--max-downloads', str(max_videos // len(VIMEO_QUERIES)),
            '-o', str(vimeo_dir / '%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=1800)  # 30 min timeout
            count = len(list(vimeo_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(3)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def download_from_dailymotion(output_dir, max_videos=500):
    """Download fight videos from Dailymotion"""
    print("\n📺 DAILYMOTION DOWNLOADS")
    print("="*60)

    dm_dir = output_dir / "dailymotion"
    dm_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for i, query in enumerate(DAILYMOTION_QUERIES, 1):
        print(f"\n[{i}/{len(DAILYMOTION_QUERIES)}] Dailymotion: '{query}'")

        # Dailymotion search URL
        search_url = f"https://www.dailymotion.com/search/{query.replace(' ', '%20')}"

        cmd = [
            'yt-dlp',
            search_url,
            '-f', 'best[height<=480]',
            '--max-downloads', str(max_videos // len(DAILYMOTION_QUERIES)),
            '-o', str(dm_dir / '%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=1800)
            count = len(list(dm_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(3)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def download_from_reddit(output_dir, max_videos=1000):
    """Download fight videos from Reddit"""
    print("\n🔴 REDDIT DOWNLOADS")
    print("="*60)

    reddit_dir = output_dir / "reddit"
    reddit_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for i, subreddit_url in enumerate(REDDIT_SUBREDDITS, 1):
        subreddit_name = subreddit_url.split('/r/')[1].split('/')[0]
        print(f"\n[{i}/{len(REDDIT_SUBREDDITS)}] Reddit: r/{subreddit_name}")

        cmd = [
            'yt-dlp',
            subreddit_url,
            '-f', 'best[height<=480]',
            '--max-downloads', str(max_videos // len(REDDIT_SUBREDDITS)),
            '-o', str(reddit_dir / f'{subreddit_name}_%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=2400)  # 40 min timeout
            count = len(list(reddit_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(5)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def download_from_archive(output_dir, max_videos=500):
    """Download fight videos from Internet Archive"""
    print("\n📚 INTERNET ARCHIVE DOWNLOADS")
    print("="*60)

    archive_dir = output_dir / "archive_org"
    archive_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for i, query in enumerate(ARCHIVE_SEARCHES, 1):
        print(f"\n[{i}/{len(ARCHIVE_SEARCHES)}] Archive.org: '{query}'")

        # Internet Archive search
        search_url = f"https://archive.org/search.php?query={query.replace(' ', '+')}&and[]=mediatype%3A%22movies%22"

        cmd = [
            'yt-dlp',
            search_url,
            '-f', 'best[height<=480]',
            '--max-downloads', str(max_videos // len(ARCHIVE_SEARCHES)),
            '-o', str(archive_dir / '%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=1800)
            count = len(list(archive_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(3)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def download_from_bilibili(output_dir, max_videos=300):
    """Download fight videos from Bilibili (Chinese platform)"""
    print("\n🇨🇳 BILIBILI DOWNLOADS")
    print("="*60)

    bili_dir = output_dir / "bilibili"
    bili_dir.mkdir(parents=True, exist_ok=True)

    bilibili_queries = [
        "UFC 比赛",  # UFC matches
        "拳击",      # Boxing
        "格斗",      # Fighting
        "MMA",
        "泰拳",      # Muay Thai
    ]

    total_downloaded = 0

    for i, query in enumerate(bilibili_queries, 1):
        print(f"\n[{i}/{len(bilibili_queries)}] Bilibili: '{query}'")

        search_url = f"https://search.bilibili.com/all?keyword={query}"

        cmd = [
            'yt-dlp',
            search_url,
            '-f', 'best[height<=480]',
            '--max-downloads', str(max_videos // len(bilibili_queries)),
            '-o', str(bili_dir / '%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=1800)
            count = len(list(bili_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(3)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Platform Fight Video Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/multiplatform_fights',
                       help='Output directory')
    parser.add_argument('--platforms', nargs='+',
                       choices=['vimeo', 'dailymotion', 'reddit', 'archive', 'bilibili', 'all'],
                       default=['all'],
                       help='Platforms to download from')
    parser.add_argument('--max-per-platform', type=int, default=500,
                       help='Max videos per platform')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MULTI-PLATFORM FIGHT VIDEO DOWNLOADER")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Platforms: {args.platforms}")
    print(f"Max per platform: {args.max_per_platform}")
    print("")
    print("Supported platforms:")
    print("  • Vimeo: High-quality MMA/boxing content")
    print("  • Dailymotion: Popular sports platform")
    print("  • Reddit: r/fightporn, r/StreetFights, r/MMA")
    print("  • Internet Archive: Public domain fight footage")
    print("  • Bilibili: Chinese combat sports content")
    print("")

    total_stats = {}

    platforms = args.platforms
    if 'all' in platforms:
        platforms = ['vimeo', 'dailymotion', 'reddit', 'archive', 'bilibili']

    # Download from each platform
    if 'vimeo' in platforms:
        total_stats['vimeo'] = download_from_vimeo(output_dir, args.max_per_platform)

    if 'dailymotion' in platforms:
        total_stats['dailymotion'] = download_from_dailymotion(output_dir, args.max_per_platform)

    if 'reddit' in platforms:
        total_stats['reddit'] = download_from_reddit(output_dir, args.max_per_platform)

    if 'archive' in platforms:
        total_stats['archive'] = download_from_archive(output_dir, args.max_per_platform)

    if 'bilibili' in platforms:
        total_stats['bilibili'] = download_from_bilibili(output_dir, args.max_per_platform)

    # Final summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print("")
    print("📊 STATISTICS BY PLATFORM:")
    print("-"*60)

    total_all = 0
    for platform, count in total_stats.items():
        print(f"  {platform.capitalize():20s}: {count:6d} videos")
        total_all += count

    print("-"*60)
    print(f"  {'TOTAL':20s}: {total_all:6d} videos")
    print("")

    print(f"📁 Location: {output_dir}")
    print("")

    # Count all video files
    all_videos = list(output_dir.rglob('*.mp4')) + list(output_dir.rglob('*.webm')) + \
                 list(output_dir.rglob('*.mkv')) + list(output_dir.rglob('*.avi'))

    print(f"✅ Verified on disk: {len(all_videos)} video files")
    print("")

    if total_all >= 2000:
        print("🎉 EXCELLENT: 2,000+ videos from alternative platforms!")
    elif total_all >= 1000:
        print("✅ GOOD: 1,000+ videos downloaded")
    else:
        print(f"⚠️  Downloaded {total_all} videos")

    print("")
    print("🔄 NEXT STEPS:")
    print("1. Combine with Phase 1 Kaggle datasets")
    print("2. Run balancing script:")
    print("   bash /home/admin/Desktop/NexaraVision/balance_and_combine.sh")
    print("")

if __name__ == "__main__":
    main()
