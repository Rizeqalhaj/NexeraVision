#!/usr/bin/env python3
"""
Download non-violence videos from scraped links
Supports YouTube, Dailymotion, Vimeo, Pexels, Pixabay, Archive.org
"""

import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

output_dir = Path("/workspace/nonviolence_videos")
output_dir.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING NON-VIOLENCE VIDEOS")
print("="*80)
print()

# Load scraped links
youtube_file = Path("youtube_nonviolence_cctv_links.json")
multi_file = Path("multiplatform_nonviolence_links.json")

all_links = []

if youtube_file.exists():
    with open(youtube_file) as f:
        data = json.load(f)
        youtube_links = data.get('links', [])
        all_links.extend(youtube_links)
        print(f"✅ Loaded {len(youtube_links)} YouTube links")

if multi_file.exists():
    with open(multi_file) as f:
        data = json.load(f)
        for platform, links in data.items():
            all_links.extend(links)
            print(f"✅ Loaded {len(links)} {platform} links")

print(f"\n📹 Total videos to download: {len(all_links):,}")
print()

def download_video(args):
    url, index = args

    try:
        # Determine platform
        if 'youtube.com' in url or 'youtu.be' in url:
            output_file = output_dir / f"youtube_normal_{index:05d}.mp4"
        elif 'dailymotion.com' in url:
            output_file = output_dir / f"dailymotion_normal_{index:05d}.mp4"
        elif 'vimeo.com' in url:
            output_file = output_dir / f"vimeo_normal_{index:05d}.mp4"
        elif 'pexels.com' in url:
            output_file = output_dir / f"pexels_normal_{index:05d}.mp4"
        elif 'pixabay.com' in url:
            output_file = output_dir / f"pixabay_normal_{index:05d}.mp4"
        elif 'archive.org' in url:
            output_file = output_dir / f"archive_normal_{index:05d}.mp4"
        else:
            output_file = output_dir / f"other_normal_{index:05d}.mp4"

        # Skip if exists
        if output_file.exists():
            return {'status': 'skip', 'url': url}

        # Download
        cmd = [
            'yt-dlp',
            '--format', 'best[ext=mp4]/best',
            '--output', str(output_file),
            '--no-playlist',
            '--quiet',
            '--no-warnings',
            url
        ]

        result = subprocess.run(cmd, timeout=300, capture_output=True)

        if result.returncode == 0 and output_file.exists():
            return {'status': 'success', 'url': url, 'file': str(output_file)}
        else:
            return {'status': 'failed', 'url': url}

    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'url': url}
    except Exception as e:
        return {'status': 'error', 'url': url, 'error': str(e)[:100]}

# Download in parallel
print("Starting downloads with 8 parallel workers...")
print()

success_count = 0
fail_count = 0
skip_count = 0

with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(download_video, [(url, i) for i, url in enumerate(all_links, 1)])

    for i, result in enumerate(results, 1):
        if result['status'] == 'success':
            success_count += 1
            print(f"[{i}/{len(all_links)}] ✅ Downloaded")
        elif result['status'] == 'skip':
            skip_count += 1
        else:
            fail_count += 1
            print(f"[{i}/{len(all_links)}] ❌ Failed")

        # Progress update every 50
        if i % 50 == 0:
            print(f"\nProgress: {i}/{len(all_links)} | Success: {success_count} | Failed: {fail_count} | Skipped: {skip_count}\n")

# Summary
print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
print()
print(f"✅ Successfully downloaded: {success_count:,} videos")
print(f"⏭️  Skipped (already exist): {skip_count:,} videos")
print(f"❌ Failed: {fail_count:,} videos")
print()
print(f"📁 Saved to: {output_dir}")
print("="*80)
