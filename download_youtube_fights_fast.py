#!/usr/bin/env python3
"""
YouTube Fight Videos Downloader - FAST VERSION
Downloads in smaller batches to avoid rate limiting
"""

import subprocess
import os
from pathlib import Path
from tqdm import tqdm
import time

# Fight-related YouTube search queries
FIGHT_SEARCH_QUERIES = [
    "UFC fight highlights 2023",
    "MMA knockouts compilation",
    "Boxing match highlights",
    "Street fight caught on camera",
    "Wrestling highlights",
    "Martial arts demonstration",
    "Kickboxing highlights",
    "Muay Thai fight",
    "Karate tournament",
    "Judo competition",
    "Taekwondo sparring",
    "Combat sports highlights",
    "Fight compilation",
    "Hockey fight",
    "Brawl caught on camera",
    "Self defense real situation",
    "Combat training",
    "Sparring session",
    "Fighting championship",
    "Combat footage"
]

def download_from_search(query, output_dir, max_videos=100):
    """Download videos from YouTube search - smaller batches"""
    print(f"\n🔍 '{query}' (max {max_videos} videos)")

    # Create query-specific directory
    query_dir = output_dir / query.replace(' ', '_').replace('/', '_')
    query_dir.mkdir(parents=True, exist_ok=True)

    # Use smaller batch - download 10 at a time
    batch_size = 10
    total_downloaded = 0

    for batch_start in range(0, max_videos, batch_size):
        batch_num = (batch_start // batch_size) + 1
        print(f"  Batch {batch_num}/{(max_videos//batch_size)}: ", end='', flush=True)

        cmd = [
            'yt-dlp',
            f'ytsearch{batch_size}:{query}',
            '-f', 'best[height<=480]',
            '--max-downloads', str(batch_size),
            '-o', str(query_dir / '%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--no-playlist',
            '--ignore-errors',
            '--no-warnings',
            '--quiet'
        ]

        try:
            result = subprocess.run(cmd, timeout=180, capture_output=True)  # 3 min timeout per batch

            # Count downloaded
            count = len(list(query_dir.glob('*.mp4'))) + len(list(query_dir.glob('*.webm')))
            batch_count = count - total_downloaded
            total_downloaded = count

            print(f"{batch_count} videos ✓")

            # Small delay between batches to avoid rate limiting
            time.sleep(2)

        except subprocess.TimeoutExpired:
            print("timeout ⏱")
            break
        except Exception as e:
            print(f"error: {e}")
            break

    print(f"  ✅ Total: {total_downloaded} videos")
    return total_downloaded

def main():
    import argparse

    parser = argparse.ArgumentParser(description='YouTube Fight Videos Downloader (Fast)')
    parser.add_argument('--output-dir', default='/workspace/datasets/youtube_fights',
                       help='Output directory')
    parser.add_argument('--videos-per-query', type=int, default=100,
                       help='Max videos per search query (recommend 50-100)')
    parser.add_argument('--queries', type=int, default=20,
                       help='Number of search queries to use')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("YOUTUBE FIGHT VIDEOS DOWNLOADER (FAST)")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Videos per query: {args.videos_per_query}")
    print(f"Search queries: {args.queries}")
    print(f"Expected total: {args.videos_per_query * args.queries} videos")
    print(f"Download strategy: Small batches (10 videos) to avoid rate limits")
    print("")

    total_downloaded = 0
    queries_to_use = FIGHT_SEARCH_QUERIES[:args.queries]

    for i, query in enumerate(queries_to_use, 1):
        print(f"\n[{i}/{len(queries_to_use)}] Processing...")
        count = download_from_search(query, output_dir, args.videos_per_query)
        total_downloaded += count

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"Total videos downloaded: {total_downloaded}")
    print(f"Location: {output_dir}")
    print("")

    # Count all videos
    all_videos = list(output_dir.rglob('*.mp4')) + list(output_dir.rglob('*.webm'))
    print(f"Videos on disk: {len(all_videos)}")
    print("")

if __name__ == "__main__":
    main()
