#!/usr/bin/env python3
"""
YouTube Fight Videos Downloader
Downloads fight/combat videos directly from YouTube using search and playlists
"""

import subprocess
import os
from pathlib import Path
from tqdm import tqdm

# Fight-related YouTube search queries
FIGHT_SEARCH_QUERIES = [
    "UFC fight highlights",
    "MMA knockouts compilation",
    "Boxing match highlights",
    "Street fight caught on camera",
    "Wrestling match highlights",
    "Martial arts demonstration",
    "Kickboxing match",
    "Muay Thai fight",
    "Karate tournament",
    "Judo competition",
    "Taekwondo sparring",
    "Combat sports highlights",
    "Fight compilation",
    "Hockey fight",
    "Brawl caught on camera",
    "Self defense demonstration",
    "Combat training",
    "Sparring session",
    "Fighting championship",
    "Combat footage"
]

def download_from_search(query, output_dir, max_videos=500):
    """Download videos from YouTube search"""
    print(f"\n🔍 Searching: '{query}'")
    print(f"   Max videos: {max_videos}")

    # Create query-specific directory
    query_dir = output_dir / query.replace(' ', '_').replace('/', '_')
    query_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'yt-dlp',
        f'ytsearch{max_videos}:{query}',
        '-f', 'best[height<=480]',  # 480p to save bandwidth
        '--max-downloads', str(max_videos),
        '-o', str(query_dir / '%(id)s.%(ext)s'),
        '--restrict-filenames',
        '--no-playlist',
        '--ignore-errors',
        '--quiet',
        '--progress'
    ]

    try:
        subprocess.run(cmd, timeout=3600)  # 1 hour timeout per query

        # Count downloaded
        count = len(list(query_dir.glob('*.mp4'))) + len(list(query_dir.glob('*.webm')))
        print(f"   ✅ Downloaded: {count} videos")
        return count

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def main():
    import argparse

    parser = argparse.ArgumentParser(description='YouTube Fight Videos Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/youtube_fights',
                       help='Output directory')
    parser.add_argument('--videos-per-query', type=int, default=500,
                       help='Max videos per search query')
    parser.add_argument('--queries', type=int, default=20,
                       help='Number of search queries to use')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("YOUTUBE FIGHT VIDEOS DOWNLOADER")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Videos per query: {args.videos_per_query}")
    print(f"Search queries: {args.queries}")
    print(f"Expected total: {args.videos_per_query * args.queries} videos")
    print("")

    total_downloaded = 0

    # Use specified number of queries
    queries_to_use = FIGHT_SEARCH_QUERIES[:args.queries]

    for i, query in enumerate(queries_to_use, 1):
        print(f"\n[{i}/{len(queries_to_use)}] Processing query...")
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
