#!/usr/bin/env python3
"""
Download TikTok videos collected via TikTokApi
Uses yt-dlp to download from URLs
"""

import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class TikTokApiDownloader:
    def __init__(self, input_json, output_dir="tiktok_downloaded", max_workers=6):
        self.input_json = Path(input_json)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers

        # Load scraped videos
        with open(self.input_json, 'r') as f:
            self.video_data = json.load(f)

        # Track progress
        self.success_log = self.output_dir / "download_success.txt"
        self.failed_log = self.output_dir / "download_failed.txt"

        # Load already downloaded
        self.downloaded = set()
        if self.success_log.exists():
            with open(self.success_log, 'r') as f:
                self.downloaded = {line.strip() for line in f}

    def download_single_video(self, video_info):
        """Download a single TikTok video"""
        url = video_info['url']
        video_id = video_info['video_id']
        hashtag = video_info.get('hashtag', 'unknown')
        author = video_info.get('author', 'unknown')

        # Skip if already downloaded
        if url in self.downloaded:
            return {'status': 'skipped', 'url': url}

        # Output template
        output_template = str(self.output_dir / f"tiktok_{hashtag}_{author}_{video_id}.%(ext)s")

        # yt-dlp command for TikTok
        cmd = [
            'yt-dlp',
            '--no-warnings',
            '--format', 'best',
            '--output', output_template,
            '--quiet',
            '--no-progress',
            '--retries', '3',
            '--fragment-retries', '3',
            '--retry-sleep', '3',
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout per video
            )

            if result.returncode == 0:
                # Log success
                with open(self.success_log, 'a') as f:
                    f.write(f"{url}\n")
                self.downloaded.add(url)
                return {'status': 'success', 'url': url}
            else:
                error = result.stderr.strip() if result.stderr else 'Unknown error'
                with open(self.failed_log, 'a') as f:
                    f.write(f"{url}\t{error}\n")
                return {'status': 'failed', 'url': url, 'error': error}

        except subprocess.TimeoutExpired:
            with open(self.failed_log, 'a') as f:
                f.write(f"{url}\tTimeout (>3 min)\n")
            return {'status': 'failed', 'url': url, 'error': 'Timeout'}

        except Exception as e:
            with open(self.failed_log, 'a') as f:
                f.write(f"{url}\t{str(e)}\n")
            return {'status': 'failed', 'url': url, 'error': str(e)}

    def download_all(self):
        """Download all videos in parallel"""
        total = len(self.video_data)
        already_have = len(self.downloaded)
        to_download = [v for v in self.video_data if v['url'] not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                    TIKTOK VIDEO DOWNLOADER (API)                           ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total videos in JSON: {total}")
        print(f"✅ Already downloaded: {already_have}")
        print(f"📥 To download: {len(to_download)}")
        print(f"🔧 Parallel workers: {self.max_workers}")
        print(f"📁 Output directory: {self.output_dir}")
        print()

        if len(to_download) == 0:
            print("✅ All videos already downloaded!")
            return

        # Download in parallel
        success_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.download_single_video, video): video for video in to_download}

            # Progress bar
            with tqdm(total=len(to_download), desc="Downloading", unit="video") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'success':
                        success_count += 1
                    elif result['status'] == 'failed':
                        failed_count += 1

                    pbar.set_postfix({'✅': success_count, '❌': failed_count})
                    pbar.update(1)

                    # Small delay to avoid rate limits
                    time.sleep(0.2)

        # Final statistics
        print()
        print("="*80)
        print("📊 DOWNLOAD COMPLETE")
        print("="*80)
        print(f"✅ Successfully downloaded: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⏭️  Skipped (already have): {already_have}")
        print(f"📁 Videos saved to: {self.output_dir}")
        print()

        if failed_count > 0:
            print(f"⚠️  Failed URLs saved to: {self.failed_log}")

        print("="*80)


def main():
    import sys

    # Check dependencies
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: yt-dlp not installed")
        print()
        print("Install with:")
        print("  pip install --break-system-packages yt-dlp")
        sys.exit(1)

    # Configuration
    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "tiktok_videos_api.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "tiktok_downloaded"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    # Check if input exists
    if not Path(INPUT_JSON).exists():
        print(f"❌ ERROR: Input file not found: {INPUT_JSON}")
        print()
        print("First run: python3 collect_tiktok_api.py")
        print("Then run: python3 download_tiktok_api_videos.py")
        sys.exit(1)

    # Run downloader
    downloader = TikTokApiDownloader(
        input_json=INPUT_JSON,
        output_dir=OUTPUT_DIR,
        max_workers=MAX_WORKERS
    )
    downloader.download_all()


if __name__ == "__main__":
    main()
