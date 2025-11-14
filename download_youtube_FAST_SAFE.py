#!/usr/bin/env python3
"""
FAST & SAFE YouTube Downloader
- Parallel downloads (10 workers)
- Automatic retry with exponential backoff
- Resume capability (skips existing)
- Progress saving every 100 videos
- Rate limiting to avoid bans
- Memory efficient
"""

import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================
JSON_FILE = "/workspace/youtube_nonviolence_cctv_links.json"
OUTPUT_DIR = Path("/workspace/youtube_nonviolence_videos")
PROGRESS_FILE = OUTPUT_DIR / "download_progress.json"

MAX_WORKERS = 10  # Parallel downloads (increase for speed)
RETRY_ATTEMPTS = 3  # Retry failed downloads
RATE_LIMIT_DELAY = 0.5  # Delay between downloads (seconds)
SAVE_PROGRESS_EVERY = 100  # Save progress every N videos

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PROGRESS TRACKING
# ============================================================================
progress_lock = threading.Lock()
progress_data = {
    'success': 0,
    'failed': 0,
    'skipped': 0,
    'failed_urls': []
}

def load_progress():
    """Load previous progress if exists"""
    global progress_data
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress_data = json.load(f)
            print(f"✓ Loaded progress: {progress_data['success']} successful, {progress_data['failed']} failed")
        except:
            pass

def save_progress():
    """Save current progress"""
    with progress_lock:
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except:
            pass

# ============================================================================
# MAIN
# ============================================================================
print("=" * 80)
print("FAST & SAFE YOUTUBE DOWNLOADER")
print("=" * 80)
print()

# Load URLs
json_path = Path(JSON_FILE)
if not json_path.exists():
    print(f"❌ File not found: {JSON_FILE}")
    sys.exit(1)

print(f"Loading: {JSON_FILE}")
with open(json_path, 'r') as f:
    data = json.load(f)

if isinstance(data, dict):
    urls = data.get('links', data.get('urls', []))
elif isinstance(data, list):
    urls = data
else:
    print("❌ Unknown JSON format")
    sys.exit(1)

print(f"✓ Loaded {len(urls)} URLs")
print()

# Load previous progress
load_progress()

# Check for yt-dlp
print("Checking yt-dlp...")
try:
    result = subprocess.run(['yt-dlp', '--version'],
                          capture_output=True, text=True, timeout=5)
    print(f"✓ yt-dlp version: {result.stdout.strip()}")
except:
    print("Installing yt-dlp...")
    subprocess.run(['pip', 'install', '-U', 'yt-dlp'], check=True)

print()
print("=" * 80)
print("DOWNLOADING")
print("=" * 80)
print(f"Workers: {MAX_WORKERS}")
print(f"Output: {OUTPUT_DIR}")
print()

start_time = time.time()

def download_video(url, index, total):
    """Download single video with retry and rate limiting"""
    try:
        # Extract video ID
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
            with progress_lock:
                progress_data['skipped'] += 1
            return 'skipped', video_id

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        # Download with yt-dlp
        for attempt in range(RETRY_ATTEMPTS):
            try:
                cmd = [
                    'yt-dlp',
                    '--format', 'best[height<=720]/best',  # Max 720p
                    '--output', output_template,
                    '--no-playlist',
                    '--quiet',
                    '--no-warnings',
                    '--socket-timeout', '30',
                    '--retries', '3',
                    '--fragment-retries', '3',
                    url
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    with progress_lock:
                        progress_data['success'] += 1

                        # Save progress periodically
                        if progress_data['success'] % SAVE_PROGRESS_EVERY == 0:
                            save_progress()

                    return 'success', video_id

                # Exponential backoff on retry
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)

            except subprocess.TimeoutExpired:
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    break

        # All retries failed
        with progress_lock:
            progress_data['failed'] += 1
            progress_data['failed_urls'].append(url)
        return 'failed', video_id

    except Exception as e:
        with progress_lock:
            progress_data['failed'] += 1
            progress_data['failed_urls'].append(url)
        return 'error', str(e)[:50]

# Download with thread pool
last_print_time = time.time()
print_interval = 2  # Print update every 2 seconds

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(download_video, url, idx + 1, len(urls)): (idx + 1, url)
        for idx, url in enumerate(urls)
    }

    for future in as_completed(futures):
        idx, url = futures[future]
        status, info = future.result()

        # Print progress updates (not every single video to reduce spam)
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            with progress_lock:
                total_processed = progress_data['success'] + progress_data['failed'] + progress_data['skipped']
                percent = (total_processed * 100) // len(urls)
                elapsed = current_time - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining = len(urls) - total_processed
                eta_seconds = remaining / rate if rate > 0 else 0

                print(f"\rProgress: {total_processed}/{len(urls)} ({percent}%) | "
                      f"✓ {progress_data['success']} | "
                      f"❌ {progress_data['failed']} | "
                      f"⏭️  {progress_data['skipped']} | "
                      f"⚡ {rate:.1f}/s | "
                      f"ETA: {eta_seconds/60:.1f}m", end='', flush=True)

            last_print_time = current_time

# Final newline
print()

# Save final progress
save_progress()

elapsed_time = time.time() - start_time

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print()
print(f"✓ Success: {progress_data['success']}")
print(f"❌ Failed: {progress_data['failed']}")
print(f"⏭️  Skipped: {progress_data['skipped']}")
print(f"⏱️  Time: {elapsed_time/60:.1f} minutes")
print()

# Count final files
downloaded_files = list(OUTPUT_DIR.glob("*.mp4")) + list(OUTPUT_DIR.glob("*.webm"))
total_size = sum(f.stat().st_size for f in downloaded_files)
total_size_gb = total_size / (1024**3)

print(f"Total videos: {len(downloaded_files)}")
print(f"Total size: {total_size_gb:.2f} GB")
print(f"📁 Location: {OUTPUT_DIR}")
print()

# Save failed URLs
if progress_data['failed_urls']:
    failed_file = OUTPUT_DIR / "failed_urls.txt"
    with open(failed_file, 'w') as f:
        for url in progress_data['failed_urls']:
            f.write(f"{url}\n")
    print(f"⚠️  {len(progress_data['failed_urls'])} failed URLs saved to: {failed_file}")
    print()

print("Tips:")
print("  • To resume: Just run this script again (skips existing)")
print("  • To retry failed: Delete failed_urls.txt and re-run")
print("  • Check videos: ls -lh /workspace/youtube_nonviolence_videos/ | head -20")
print()
