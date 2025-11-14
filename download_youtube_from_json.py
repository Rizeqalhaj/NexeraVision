#!/usr/bin/env python3
"""
Download YouTube videos from scraped JSON files
Supports both youtube_nonviolence_cctv_links.json and 1youtube_nonviolence_cctv_links.json
"""

import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
JSON_FILES = [
    "/workspace/youtube_nonviolence_cctv_links.json",
    "/workspace/1youtube_nonviolence_cctv_links.json"
]

OUTPUT_DIR = Path("/workspace/youtube_nonviolence_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 5  # Parallel downloads
RETRY_ATTEMPTS = 2

print("=" * 80)
print("YOUTUBE NON-VIOLENCE VIDEO DOWNLOADER")
print("=" * 80)
print()

# Load all URLs from JSON files
all_urls = set()

for json_file in JSON_FILES:
    json_path = Path(json_file)

    if not json_path.exists():
        print(f"⚠️  File not found: {json_file}")
        continue

    print(f"Loading: {json_file}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            if 'links' in data:
                links = data['links']
            elif 'urls' in data:
                links = data['urls']
            else:
                # Try to find any list in the dict
                for key, value in data.items():
                    if isinstance(value, list):
                        links = value
                        break
                else:
                    links = []
        elif isinstance(data, list):
            links = data
        else:
            print(f"❌ Unknown JSON format in {json_file}")
            continue

        # Add to set (deduplicates automatically)
        before_count = len(all_urls)
        all_urls.update(links)
        new_count = len(all_urls) - before_count

        print(f"  ✓ Loaded {len(links)} URLs ({new_count} new, {len(links) - new_count} duplicates)")

    except Exception as e:
        print(f"❌ Error loading {json_file}: {e}")
        continue

print()
print(f"Total unique URLs: {len(all_urls)}")
print()

if len(all_urls) == 0:
    print("❌ No URLs found to download")
    sys.exit(1)

# Check for yt-dlp
try:
    result = subprocess.run(['yt-dlp', '--version'],
                          capture_output=True, text=True, timeout=5)
    print(f"✓ yt-dlp version: {result.stdout.strip()}")
except:
    print("❌ yt-dlp not found. Installing...")
    subprocess.run(['pip', 'install', '-U', 'yt-dlp'], check=True)
    print("✓ yt-dlp installed")

print()
print("=" * 80)
print("DOWNLOADING VIDEOS")
print("=" * 80)
print()

# Convert to list for indexing
url_list = list(all_urls)

# Track progress
success_count = 0
failed_count = 0
skipped_count = 0
failed_urls = []

def download_video(url, index, total):
    """Download single video with retry logic"""
    global success_count, failed_count, skipped_count

    try:
        # Generate filename based on video ID
        if '/shorts/' in url:
            video_id = url.split('/shorts/')[1].split('?')[0]
        elif 'watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0]
        else:
            video_id = f"video_{index:05d}"

        output_template = str(OUTPUT_DIR / f"{video_id}.%(ext)s")

        # Check if already downloaded
        existing = list(OUTPUT_DIR.glob(f"{video_id}.*"))
        if existing:
            skipped_count += 1
            return f"[{index}/{total}] ⏭️  Skipped (exists): {video_id}"

        # Download with yt-dlp
        cmd = [
            'yt-dlp',
            '--format', 'best[height<=720]',  # Max 720p to save space
            '--output', output_template,
            '--no-playlist',
            '--quiet',
            '--no-warnings',
            '--retries', str(RETRY_ATTEMPTS),
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            success_count += 1
            return f"[{index}/{total}] ✓ Downloaded: {video_id}"
        else:
            failed_count += 1
            failed_urls.append(url)
            error = result.stderr[:100] if result.stderr else "Unknown error"
            return f"[{index}/{total}] ❌ Failed: {video_id} ({error})"

    except subprocess.TimeoutExpired:
        failed_count += 1
        failed_urls.append(url)
        return f"[{index}/{total}] ⏱️  Timeout: {url}"
    except Exception as e:
        failed_count += 1
        failed_urls.append(url)
        return f"[{index}/{total}] ❌ Error: {str(e)[:50]}"

# Download with thread pool
start_time = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(download_video, url, idx + 1, len(url_list)): url
        for idx, url in enumerate(url_list)
    }

    for future in as_completed(futures):
        result = future.result()
        print(result)

        # Progress update every 50 videos
        total_processed = success_count + failed_count + skipped_count
        if total_processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            eta = (len(url_list) - total_processed) / rate if rate > 0 else 0

            print()
            print(f"Progress: {total_processed}/{len(url_list)} "
                  f"({total_processed*100//len(url_list)}%) | "
                  f"Success: {success_count} | Failed: {failed_count} | Skipped: {skipped_count} | "
                  f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min")
            print()

elapsed_time = time.time() - start_time

print()
print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print()
print(f"✓ Success: {success_count}")
print(f"❌ Failed: {failed_count}")
print(f"⏭️  Skipped: {skipped_count}")
print(f"⏱️  Time: {elapsed_time/60:.1f} minutes")
print(f"📁 Output: {OUTPUT_DIR}")
print()

# Count downloaded files
downloaded_files = list(OUTPUT_DIR.glob("*.*"))
total_size = sum(f.stat().st_size for f in downloaded_files)
total_size_gb = total_size / (1024**3)

print(f"Total videos: {len(downloaded_files)}")
print(f"Total size: {total_size_gb:.2f} GB")
print()

# Save failed URLs
if failed_urls:
    failed_file = OUTPUT_DIR / "failed_urls.txt"
    with open(failed_file, 'w') as f:
        for url in failed_urls:
            f.write(f"{url}\n")
    print(f"Failed URLs saved to: {failed_file}")
    print()

print("Next steps:")
print("  1. Check videos: ls -lh /workspace/youtube_nonviolence_videos/ | head -20")
print("  2. Count videos: ls /workspace/youtube_nonviolence_videos/*.* | wc -l")
print("  3. Retry failed: python3 download_youtube_from_json.py (will skip existing)")
print()
