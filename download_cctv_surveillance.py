#!/usr/bin/env python3
"""
CCTV Surveillance Footage Downloader
Specifically for camera-angle fight/violence detection
CRITICAL for production deployment - trains model on actual camera perspectives
"""

import subprocess
import os
from pathlib import Path
from tqdm import tqdm
import time

# ============================================
# CCTV/SURVEILLANCE SPECIFIC SEARCH QUERIES
# ============================================

# High-value CCTV fight searches
CCTV_FIGHT_QUERIES = [
    "CCTV fight caught on camera",
    "security camera fight",
    "surveillance footage fight",
    "store security camera fight",
    "parking lot security fight",
    "street camera fight footage",
    "gas station fight camera",
    "convenience store fight CCTV",
    "mall security camera fight",
    "restaurant fight caught on camera",
    "hotel lobby fight security",
    "bar fight security footage",
    "nightclub security camera fight",
    "school security camera fight",
    "hospital security fight",
    "office building security fight",
    "elevator fight caught on camera",
    "subway security camera fight",
    "train station CCTV fight",
    "bus fight security camera",
    "airport security fight",
    "casino security camera fight",
]

# CCTV non-violent (for comparison/balance)
CCTV_NORMAL_QUERIES = [
    "security camera normal day",
    "CCTV normal activity",
    "store surveillance no incident",
    "parking lot security normal",
    "surveillance footage peaceful",
    "security camera daily routine",
]

# Reddit - Best source for CCTV footage
# Use top/hot posts from fight-focused subreddits (yt-dlp compatible)
REDDIT_CCTV_SUBREDDITS = [
    "https://www.reddit.com/r/fightporn/top/?t=all",
    "https://www.reddit.com/r/fightporn/top/?t=year",
    "https://www.reddit.com/r/StreetFights/top/?t=all",
    "https://www.reddit.com/r/PublicFreakout/top/?t=year",
    "https://www.reddit.com/r/ActualFreakouts/top/?t=all",
]

# Liveleak alternatives (Liveleak shutdown 2021)
LIVELEAK_ALTERNATIVES = [
    "https://www.reddit.com/r/CrazyFuckingVideos/top/?t=year",
    "https://www.reddit.com/r/AbruptChaos/top/?t=year",
    "https://www.reddit.com/r/DocumentedFights/top/?t=all",
]

def download_cctv_fights_youtube(output_dir, max_per_query=500):
    """Download CCTV fight footage from YouTube"""
    print("\n📹 YOUTUBE CCTV FIGHT FOOTAGE")
    print("="*60)

    youtube_dir = output_dir / "youtube_cctv"
    youtube_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for i, query in enumerate(CCTV_FIGHT_QUERIES, 1):
        print(f"\n[{i}/{len(CCTV_FIGHT_QUERIES)}] '{query}'")

        # YouTube search with batching to avoid rate limiting
        batch_size = 50  # Smaller batches for reliability
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
                subprocess.run(cmd, timeout=300)  # 5 min per batch
                count = len(list(youtube_dir.glob('*.*')))
                batch_downloaded = count - total_downloaded
                total_downloaded = count
                print(f"{batch_downloaded} videos ✓")
                time.sleep(3)  # Rate limiting prevention
            except Exception as e:
                print(f"error: {e}")
                break

        print(f"  Total so far: {total_downloaded} videos")

    return total_downloaded

def download_cctv_reddit(output_dir, max_per_subreddit=2000):
    """Download CCTV footage from Reddit - BEST SOURCE"""
    print("\n🔴 REDDIT CCTV/SURVEILLANCE FOOTAGE")
    print("="*60)

    reddit_dir = output_dir / "reddit_cctv"
    reddit_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    all_subreddits = REDDIT_CCTV_SUBREDDITS + LIVELEAK_ALTERNATIVES

    for i, subreddit_url in enumerate(all_subreddits, 1):
        # Extract search term for naming
        if 'search' in subreddit_url:
            search_term = subreddit_url.split('q=')[1] if 'q=' in subreddit_url else 'unknown'
            search_term = search_term.replace('+', '_').replace('/', '_')
        else:
            search_term = subreddit_url.split('/r/')[1].split('/')[0]

        print(f"\n[{i}/{len(all_subreddits)}] Reddit: {search_term}")

        cmd = [
            'yt-dlp',
            subreddit_url,
            '-f', 'best[height<=720]',  # CCTV quality is usually lower
            '--max-downloads', str(max_per_subreddit),
            '-o', str(reddit_dir / f'{search_term}_%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=3600)  # 1 hour per subreddit
            count = len(list(reddit_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(5)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def download_cctv_vimeo(output_dir, max_videos=300):
    """Download CCTV footage from Vimeo"""
    print("\n🎬 VIMEO CCTV FOOTAGE")
    print("="*60)

    vimeo_dir = output_dir / "vimeo_cctv"
    vimeo_dir.mkdir(parents=True, exist_ok=True)

    vimeo_queries = [
        "https://vimeo.com/search?q=CCTV+fight",
        "https://vimeo.com/search?q=security+camera+fight",
        "https://vimeo.com/search?q=surveillance+footage",
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

def download_cctv_dailymotion(output_dir, max_videos=500):
    """Download CCTV footage from Dailymotion"""
    print("\n📺 DAILYMOTION CCTV FOOTAGE")
    print("="*60)

    dm_dir = output_dir / "dailymotion_cctv"
    dm_dir.mkdir(parents=True, exist_ok=True)

    dm_queries = [
        "CCTV fight",
        "security camera fight",
        "surveillance footage fight",
        "caught on camera fight",
    ]

    total_downloaded = 0

    for i, query in enumerate(dm_queries, 1):
        print(f"\n[{i}/{len(dm_queries)}] Dailymotion: '{query}'")

        search_url = f"https://www.dailymotion.com/search/{query.replace(' ', '%20')}"

        cmd = [
            'yt-dlp',
            search_url,
            '-f', 'best[height<=480]',
            '--max-downloads', str(max_videos // len(dm_queries)),
            '-o', str(dm_dir / '%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            subprocess.run(cmd, timeout=1200)
            count = len(list(dm_dir.glob('*.*')))
            downloaded = count - total_downloaded
            total_downloaded = count
            print(f"   ✅ Downloaded: {downloaded} videos")
            time.sleep(3)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return total_downloaded

def main():
    import argparse

    parser = argparse.ArgumentParser(description='CCTV Surveillance Fight Footage Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/cctv_surveillance',
                       help='Output directory')
    parser.add_argument('--sources', nargs='+',
                       choices=['reddit', 'youtube', 'vimeo', 'dailymotion', 'all'],
                       default=['all'],
                       help='Video sources to download from')
    parser.add_argument('--max-reddit', type=int, default=2000,
                       help='Max videos per Reddit search')
    parser.add_argument('--max-youtube-per-query', type=int, default=500,
                       help='Max videos per YouTube query')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CCTV SURVEILLANCE FIGHT FOOTAGE DOWNLOADER")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Sources: {args.sources}")
    print("")
    print("🎯 WHY CCTV FOOTAGE IS CRITICAL:")
    print("  ✓ Matches your production deployment (camera angles)")
    print("  ✓ Real-world surveillance perspectives")
    print("  ✓ Typical camera quality and lighting")
    print("  ✓ Authentic violence detection scenarios")
    print("")
    print("📊 Expected CCTV footage volume:")
    print("  • Reddit: 10,000-15,000 videos (BEST SOURCE)")
    print("  • YouTube: 5,000-10,000 videos")
    print("  • Vimeo: 300-500 videos")
    print("  • Dailymotion: 500-1,000 videos")
    print("  • TOTAL: 15,000-26,000 CCTV videos")
    print("")

    total_stats = {}

    sources = args.sources
    if 'all' in sources:
        sources = ['reddit', 'youtube', 'vimeo', 'dailymotion']

    # Reddit is the BEST source for CCTV footage
    if 'reddit' in sources:
        total_stats['reddit'] = download_cctv_reddit(output_dir, args.max_reddit)

    # YouTube has good volume
    if 'youtube' in sources:
        total_stats['youtube'] = download_cctv_fights_youtube(output_dir, args.max_youtube_per_query)

    # Vimeo for quality
    if 'vimeo' in sources:
        total_stats['vimeo'] = download_cctv_vimeo(output_dir)

    # Dailymotion for additional content
    if 'dailymotion' in sources:
        total_stats['dailymotion'] = download_cctv_dailymotion(output_dir)

    # Final summary
    print("\n" + "="*80)
    print("CCTV DOWNLOAD COMPLETE!")
    print("="*80)
    print("")
    print("📊 STATISTICS BY SOURCE:")
    print("-"*60)

    total_all = 0
    for source, count in total_stats.items():
        print(f"  {source.capitalize():20s}: {count:6d} videos")
        total_all += count

    print("-"*60)
    print(f"  {'TOTAL CCTV FOOTAGE':20s}: {total_all:6d} videos")
    print("")

    print(f"📁 Location: {output_dir}")
    print("")

    # Count all video files
    all_videos = list(output_dir.rglob('*.mp4')) + list(output_dir.rglob('*.webm')) + \
                 list(output_dir.rglob('*.mkv')) + list(output_dir.rglob('*.avi'))

    print(f"✅ Verified on disk: {len(all_videos)} CCTV video files")
    print("")

    if total_all >= 15000:
        print("🎉 EXCELLENT: 15,000+ CCTV videos!")
        print("   This will give you PRODUCTION-GRADE accuracy for camera deployment")
    elif total_all >= 10000:
        print("✅ VERY GOOD: 10,000+ CCTV videos")
        print("   Excellent for real-world camera deployment")
    elif total_all >= 5000:
        print("✅ GOOD: 5,000+ CCTV videos")
    else:
        print(f"⚠️  Downloaded {total_all} CCTV videos")

    print("")
    print("🎯 NEXT STEPS:")
    print("1. These CCTV videos are VIOLENT (fights caught on camera)")
    print("2. You need matching CCTV NON-VIOLENT footage for balance")
    print("3. Run: python3 download_cctv_normal.py  # For non-violent CCTV")
    print("4. Then combine all datasets:")
    print("   bash /home/admin/Desktop/NexaraVision/balance_and_combine.sh")
    print("")
    print("💡 PRO TIP:")
    print("   Training on CCTV footage = Better accuracy for camera deployment")
    print("   Your model will understand:")
    print("   - Camera angles and perspectives")
    print("   - Typical surveillance quality")
    print("   - Real-world lighting conditions")
    print("   - Actual violence scenarios in monitored areas")
    print("")

if __name__ == "__main__":
    main()
