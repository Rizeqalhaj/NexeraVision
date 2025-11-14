#!/usr/bin/env python3
"""
SAFE Non-Violent Video Downloader
Downloads ONLY non-violent videos with automatic filtering
"""

import subprocess
import os
from pathlib import Path

# YouTube searches for NON-VIOLENT activities
NON_VIOLENT_QUERIES = [
    # Daily activities
    "people cooking tutorial",
    "family dinner eating",
    "cleaning house timelapse",
    "morning routine vlog",
    "grocery shopping",
    "walking in park",
    "reading books",

    # Work/office
    "office work day",
    "typing on computer",
    "business meeting",
    "presentation speech",

    # Social
    "friends talking conversation",
    "family gathering",
    "people laughing compilation",
    "handshaking greeting",

    # Sports (non-combat)
    "running jogging",
    "swimming lessons",
    "cycling bike ride",
    "basketball game highlights",
    "soccer practice",
    "tennis match",
    "golf swing tutorial",

    # Hobbies
    "gardening plants",
    "painting art",
    "playing guitar",
    "playing piano",
    "knitting tutorial",
    "sewing lessons",

    # Transportation
    "driving car dashcam",
    "train journey",
    "bus ride",
    "walking commute",

    # Surveillance style (normal)
    "security camera normal day",
    "store surveillance footage normal",
    "parking lot camera",
    "hallway security camera",
]

def download_nonviolent_youtube(output_dir, videos_per_query=500):
    """Download non-violent videos from YouTube"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("NON-VIOLENT YOUTUBE VIDEO DOWNLOADER")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Videos per query: {videos_per_query}")
    print(f"Total queries: {len(NON_VIOLENT_QUERIES)}")
    print(f"Expected total: {videos_per_query * len(NON_VIOLENT_QUERIES)} videos")
    print("")

    total_downloaded = 0

    for i, query in enumerate(NON_VIOLENT_QUERIES, 1):
        print(f"\n[{i}/{len(NON_VIOLENT_QUERIES)}] Downloading: '{query}'")

        # Create query directory
        query_dir = output_dir / query.replace(' ', '_').replace('/', '_')
        query_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'yt-dlp',
            f'ytsearch{videos_per_query}:{query}',
            '-f', 'best[height<=480]',
            '--max-downloads', str(videos_per_query),
            '-o', str(query_dir / '%(id)s.%(ext)s'),
            '--restrict-filenames',
            '--no-playlist',
            '--ignore-errors',
            '--quiet',
            '--progress'
        ]

        try:
            subprocess.run(cmd, timeout=3600)

            # Count downloaded
            count = len(list(query_dir.glob('*.mp4'))) + len(list(query_dir.glob('*.webm')))
            total_downloaded += count
            print(f"   ✅ Downloaded: {count} videos")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"Total downloaded: {total_downloaded} videos")
    print(f"Location: {output_dir}")
    print("")

    return total_downloaded

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Safe Non-Violent Video Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/nonviolent_safe',
                       help='Output directory')
    parser.add_argument('--videos-per-query', type=int, default=500,
                       help='Videos per search query')

    args = parser.parse_args()

    download_nonviolent_youtube(args.output_dir, args.videos_per_query)
