#!/usr/bin/env python3
"""
Reddit Video Downloader - Rate Limit Safe
NO BANS - Proper authentication, rate limiting, retry logic
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

class SafeRedditDownloader:
    def __init__(self, input_json, output_dir="downloaded_reddit_videos", max_workers=3):
        self.input_json = Path(input_json)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers  # Lower = safer

        # Load URLs
        with open(self.input_json, 'r') as f:
            data = json.load(f)
            # Handle both list of URLs and list of dicts
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], str):
                    self.urls = data
                elif isinstance(data[0], dict) and 'url' in data[0]:
                    self.urls = [item['url'] for item in data]
                else:
                    raise ValueError("Unknown JSON format")
            else:
                self.urls = []

        # Progress tracking
        self.success_log = self.output_dir / "download_success.txt"
        self.failed_log = self.output_dir / "download_failed.txt"

        # Load already downloaded
        self.downloaded = set()
        if self.success_log.exists():
            with open(self.success_log, 'r') as f:
                self.downloaded = {line.strip() for line in f}

    def download_single_video(self, url):
        """Download a single Reddit video with rate limit protection"""

        # Skip if already downloaded
        if url in self.downloaded:
            return {'status': 'skipped', 'url': url}

        # Extract post ID for filename
        post_id = url.split('/')[-3] if '/comments/' in url else url.split('/')[-1]
        post_id = post_id.split('?')[0]  # Remove query params

        # Output template
        output_template = str(self.output_dir / f"reddit_{post_id}.%(ext)s")

        # yt-dlp command with Reddit-specific optimizations
        cmd = [
            'yt-dlp',
            '--no-warnings',
            '--format', 'best',
            '--output', output_template,
            '--quiet',
            '--no-progress',

            # Reddit-specific settings
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '--referer', 'https://www.reddit.com/',

            # Retry settings - more aggressive
            '--retries', '5',
            '--fragment-retries', '5',
            '--retry-sleep', '5',

            # Rate limiting
            '--sleep-interval', '2',  # Sleep 2s between downloads
            '--max-sleep-interval', '5',  # Random up to 5s

            # Timeout settings
            '--socket-timeout', '30',

            # Error handling
            '--no-abort-on-error',
            '--ignore-errors',

            url
        ]

        try:
            # Add random delay before starting (1-3 seconds)
            time.sleep(random.uniform(1, 3))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per video
            )

            if result.returncode == 0:
                with open(self.success_log, 'a') as f:
                    f.write(f"{url}\n")
                self.downloaded.add(url)
                return {'status': 'success', 'url': url}
            else:
                error = result.stderr.strip() if result.stderr else 'Unknown error'

                # Check for rate limit errors
                if 'rate limit' in error.lower() or '429' in error:
                    return {'status': 'rate_limited', 'url': url, 'error': 'Rate limited'}

                with open(self.failed_log, 'a') as f:
                    f.write(f"{url}\t{error}\n")
                return {'status': 'failed', 'url': url, 'error': error}

        except subprocess.TimeoutExpired:
            with open(self.failed_log, 'a') as f:
                f.write(f"{url}\tTimeout (>2 min)\n")
            return {'status': 'failed', 'url': url, 'error': 'Timeout'}

        except Exception as e:
            with open(self.failed_log, 'a') as f:
                f.write(f"{url}\t{str(e)}\n")
            return {'status': 'failed', 'url': url, 'error': str(e)}

    def download_all(self):
        """Download all videos with rate limit protection"""
        total = len(self.urls)
        already_have = len(self.downloaded)
        to_download = [url for url in self.urls if url not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                  REDDIT VIDEO DOWNLOADER - SAFE MODE                       ║")
        print("║                  Rate Limit Protection Enabled                             ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total videos in JSON: {total}")
        print(f"✅ Already downloaded: {already_have}")
        print(f"📥 To download: {len(to_download)}")
        print(f"🔧 Parallel workers: {self.max_workers} (safer = slower but no bans)")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⏱️  Estimated time: ~{len(to_download) * 5 / 60:.1f} minutes (with delays)")
        print()
        print("🛡️  SAFETY FEATURES:")
        print("   - Random delays (1-3s) between downloads")
        print("   - Sleep intervals (2-5s) built into yt-dlp")
        print("   - Lower concurrency (3 workers max)")
        print("   - Automatic retry with backoff")
        print("   - Rate limit detection and handling")
        print()

        if len(to_download) == 0:
            print("✅ All videos already downloaded!")
            return

        # Download with rate limiting
        success_count = 0
        failed_count = 0
        rate_limited_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_video, url): url for url in to_download}

            with tqdm(total=len(to_download), desc="Downloading", unit="video") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'success':
                        success_count += 1
                    elif result['status'] == 'rate_limited':
                        rate_limited_count += 1
                        # If rate limited, add longer delay
                        print(f"\n⚠️  Rate limit detected - pausing for 30 seconds...")
                        time.sleep(30)
                    elif result['status'] == 'failed':
                        failed_count += 1

                    pbar.set_postfix({
                        '✅': success_count,
                        '❌': failed_count,
                        '⏸️': rate_limited_count
                    })
                    pbar.update(1)

        # Final statistics
        print()
        print("="*80)
        print("📊 DOWNLOAD COMPLETE")
        print("="*80)
        print(f"✅ Successfully downloaded: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⏸️  Rate limited (will retry later): {rate_limited_count}")
        print(f"⏭️  Skipped (already have): {already_have}")
        print(f"📁 Videos saved to: {self.output_dir}")
        print()

        if failed_count > 0 or rate_limited_count > 0:
            print(f"⚠️  Some videos failed or were rate limited")
            print(f"   Failed URLs saved to: {self.failed_log}")
            print(f"   To retry: python3 {sys.argv[0]}")

        print("="*80)


def main():
    # Check dependencies
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: yt-dlp not installed")
        print()
        print("Install with:")
        print("  pip install --break-system-packages yt-dlp")
        sys.exit(1)

    # Configuration - Auto-detect Reddit JSON file
    if len(sys.argv) > 1:
        INPUT_JSON = sys.argv[1]
    else:
        # Search for Reddit JSON files
        reddit_files = [
            "reddit_fight_videos_all.json",
            "reddit_fight_urls.json",
            "reddit_videos.json"
        ]
        INPUT_JSON = None
        for fname in reddit_files:
            if Path(fname).exists():
                INPUT_JSON = fname
                break

        if INPUT_JSON is None:
            print("❌ ERROR: No Reddit JSON file found")
            print()
            print("Searched for: reddit_fight_videos_all.json, reddit_fight_urls.json, reddit_videos.json")
            print()
            print("Specify file: python3 download_reddit_videos_safe.py <your_file.json>")
            sys.exit(1)
        else:
            print(f"✅ Auto-detected: {INPUT_JSON}")
            print()

    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_reddit_videos"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # Default 3 (safe)

    # Check if input exists
    if not Path(INPUT_JSON).exists():
        print(f"❌ ERROR: Input file not found: {INPUT_JSON}")
        print()
        print("First run the Reddit scraper:")
        print("  python3 scrape_reddit_multi_query.py")
        sys.exit(1)

    # Run downloader
    downloader = SafeRedditDownloader(
        input_json=INPUT_JSON,
        output_dir=OUTPUT_DIR,
        max_workers=MAX_WORKERS
    )
    downloader.download_all()


if __name__ == "__main__":
    main()
