#!/usr/bin/env python3
"""
Reddit Video Downloader using gallery-dl
Works better than yt-dlp for Reddit - handles v.redd.it properly
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

class GalleryDLRedditDownloader:
    def __init__(self, input_json, output_dir="downloaded_reddit_videos", max_workers=2):
        self.input_json = Path(input_json)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers

        # Load URLs
        with open(self.input_json, 'r') as f:
            data = json.load(f)
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

        self.downloaded = set()
        if self.success_log.exists():
            with open(self.success_log, 'r') as f:
                self.downloaded = {line.strip() for line in f}

    def download_single_video(self, url):
        """Download using gallery-dl (better for Reddit)"""

        if url in self.downloaded:
            return {'status': 'skipped', 'url': url}

        # Random delay to avoid rate limits
        time.sleep(random.uniform(2, 5))

        # gallery-dl command
        cmd = [
            'gallery-dl',
            '--no-skip',
            '--no-part',
            '--dest', str(self.output_dir),
            '--filename', '{id}_{num}.{extension}',

            # Rate limiting
            '--sleep', '3',  # 3s between requests
            '--sleep-request', '2',  # 2s between API calls

            # Retry settings
            '--retries', '5',
            '--timeout', '30',

            # Quiet mode
            '--quiet',

            url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                with open(self.success_log, 'a') as f:
                    f.write(f"{url}\n")
                self.downloaded.add(url)
                return {'status': 'success', 'url': url}
            else:
                error = result.stderr.strip() if result.stderr else 'Unknown error'

                # Check for rate limit
                if 'rate' in error.lower() or '429' in error:
                    return {'status': 'rate_limited', 'url': url, 'error': 'Rate limited'}

                with open(self.failed_log, 'a') as f:
                    f.write(f"{url}\t{error}\n")
                return {'status': 'failed', 'url': url, 'error': error}

        except subprocess.TimeoutExpired:
            with open(self.failed_log, 'a') as f:
                f.write(f"{url}\tTimeout\n")
            return {'status': 'failed', 'url': url, 'error': 'Timeout'}
        except Exception as e:
            with open(self.failed_log, 'a') as f:
                f.write(f"{url}\t{str(e)}\n")
            return {'status': 'failed', 'url': url, 'error': str(e)}

    def download_all(self):
        """Download all videos"""
        total = len(self.urls)
        already_have = len(self.downloaded)
        to_download = [url for url in self.urls if url not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              REDDIT VIDEO DOWNLOADER - gallery-dl                          ║")
        print("║              Better for Reddit than yt-dlp                                 ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total videos: {total}")
        print(f"✅ Already downloaded: {already_have}")
        print(f"📥 To download: {len(to_download)}")
        print(f"🔧 Workers: {self.max_workers}")
        print(f"📁 Output: {self.output_dir}")
        print(f"⏱️  Estimated time: ~{len(to_download) * 6 / 60:.1f} minutes")
        print()

        if len(to_download) == 0:
            print("✅ All videos already downloaded!")
            return

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
                        print(f"\n⚠️  Rate limited - pausing 60 seconds...")
                        time.sleep(60)
                    elif result['status'] == 'failed':
                        failed_count += 1

                    pbar.set_postfix({'✅': success_count, '❌': failed_count, '⏸️': rate_limited_count})
                    pbar.update(1)

        print()
        print("="*80)
        print("📊 DOWNLOAD COMPLETE")
        print("="*80)
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⏸️  Rate limited: {rate_limited_count}")
        print(f"📁 Saved to: {self.output_dir}")
        print("="*80)


def main():
    # Check if gallery-dl is installed
    try:
        subprocess.run(['gallery-dl', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: gallery-dl not installed")
        print()
        print("Install with:")
        print("  pip install --break-system-packages gallery-dl")
        sys.exit(1)

    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "reddit_fight_videos_all.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_reddit_videos"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    if not Path(INPUT_JSON).exists():
        print(f"❌ ERROR: File not found: {INPUT_JSON}")
        sys.exit(1)

    downloader = GalleryDLRedditDownloader(INPUT_JSON, OUTPUT_DIR, MAX_WORKERS)
    downloader.download_all()


if __name__ == "__main__":
    main()
