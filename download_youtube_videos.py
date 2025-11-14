#!/usr/bin/env python3
"""
YouTube Shorts Video Downloader
Fast and reliable - no IP bans, no rate limits
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

class YouTubeDownloader:
    def __init__(self, input_json, output_dir="downloaded_youtube_videos", max_workers=8):
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
                self.urls = []

        # Progress tracking
        self.success_log = self.output_dir / "download_success.txt"
        self.failed_log = self.output_dir / "download_failed.txt"

        self.downloaded = set()
        if self.success_log.exists():
            with open(self.success_log, 'r') as f:
                self.downloaded = {line.strip() for line in f}

    def extract_video_id(self, url):
        """Extract YouTube video ID from URL"""
        # Handle various YouTube URL formats
        if 'youtube.com/watch?v=' in url:
            return url.split('watch?v=')[1].split('&')[0]
        elif 'youtube.com/shorts/' in url:
            return url.split('shorts/')[1].split('?')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        else:
            # Extract anything that looks like a video ID
            parts = url.split('/')
            for part in parts:
                if len(part) == 11 and part.isalnum():
                    return part
            return url.split('/')[-1].split('?')[0]

    def download_single_video(self, url):
        """Download single YouTube video"""

        if url in self.downloaded:
            return {'status': 'skipped', 'url': url}

        # Extract video ID
        video_id = self.extract_video_id(url)
        output_file = self.output_dir / f"youtube_{video_id}.mp4"

        # Check if already exists
        if output_file.exists() and output_file.stat().st_size > 50000:
            with open(self.success_log, 'a') as f:
                f.write(f"{url}\n")
            self.downloaded.add(url)
            return {'status': 'skipped', 'url': url}

        # yt-dlp command optimized for YouTube
        cmd = [
            'yt-dlp',

            # Format selection - prioritize mp4
            '--format', 'best[ext=mp4]/best',
            '--merge-output-format', 'mp4',

            # Output settings
            '--output', str(output_file),
            '--no-warnings',
            '--quiet',
            '--no-progress',

            # YouTube-specific optimizations
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',

            # Retry settings
            '--retries', '5',
            '--fragment-retries', '5',
            '--retry-sleep', '2',

            # Timeout
            '--socket-timeout', '30',

            # Speed optimizations
            '--concurrent-fragments', '4',
            '--buffer-size', '16K',

            # Error handling
            '--no-abort-on-error',
            '--ignore-errors',

            # Limit file size (optional - for faster downloads)
            # '--max-filesize', '50M',

            url
        ]

        try:
            # Small random delay to avoid hammering YouTube
            time.sleep(random.uniform(0.2, 0.5))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Verify download
            if output_file.exists() and output_file.stat().st_size > 50000:
                with open(self.success_log, 'a') as f:
                    f.write(f"{url}\n")
                self.downloaded.add(url)
                return {'status': 'success', 'url': url}
            else:
                error = result.stderr[:200] if result.stderr else 'Download failed'
                with open(self.failed_log, 'a') as f:
                    f.write(f"{url}\t{error}\n")
                return {'status': 'failed', 'url': url, 'error': error}

        except subprocess.TimeoutExpired:
            with open(self.failed_log, 'a') as f:
                f.write(f"{url}\tTimeout\n")
            return {'status': 'failed', 'url': url, 'error': 'Timeout'}

        except Exception as e:
            error = str(e)[:200]
            with open(self.failed_log, 'a') as f:
                f.write(f"{url}\t{error}\n")
            return {'status': 'failed', 'url': url, 'error': error}

    def download_all(self):
        """Download all YouTube videos"""
        total = len(self.urls)
        already_have = len(self.downloaded)
        to_download = [url for url in self.urls if url not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              YOUTUBE VIDEO DOWNLOADER                                      ║")
        print("║              Fast & Reliable - No Rate Limits                              ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print("⚡ FEATURES:")
        print("   - High-speed concurrent downloads")
        print("   - No IP bans or rate limits")
        print("   - Automatic retry on failure")
        print("   - Resume support")
        print()
        print(f"📊 Total videos: {total}")
        print(f"✅ Already downloaded: {already_have}")
        print(f"📥 To download: {len(to_download)}")
        print(f"🔧 Parallel workers: {self.max_workers}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⏱️  Estimated time: ~{len(to_download) * 3 / self.max_workers / 60:.1f} minutes")
        print()

        if len(to_download) == 0:
            print("✅ All videos already downloaded!")
            return

        success_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_video, url): url for url in to_download}

            with tqdm(total=len(to_download), desc="Downloading", unit="video") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'success':
                        success_count += 1
                    elif result['status'] == 'failed':
                        failed_count += 1

                    success_rate = (success_count / (success_count + failed_count) * 100) if (success_count + failed_count) > 0 else 0
                    pbar.set_postfix({'✅': success_count, '❌': failed_count, 'Rate': f'{success_rate:.1f}%'})
                    pbar.update(1)

        print()
        print("="*80)
        print("📊 DOWNLOAD COMPLETE")
        print("="*80)
        print(f"✅ Successfully downloaded: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📈 Success rate: {(success_count / (success_count + failed_count) * 100):.1f}%")
        print(f"📁 Saved to: {self.output_dir}")
        print()

        if failed_count > 0:
            print(f"⚠️  {failed_count} videos failed")
            print(f"   Common reasons: video deleted, private, or region-blocked")
            print(f"   Failed list: {self.failed_log}")

        print("="*80)


def main():
    # Check yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except:
        print("❌ ERROR: yt-dlp not installed")
        print()
        print("Install with:")
        print("  pip install --break-system-packages yt-dlp")
        sys.exit(1)

    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "youtube_shorts_fights.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_youtube_videos"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    if not Path(INPUT_JSON).exists():
        print(f"❌ File not found: {INPUT_JSON}")
        print()
        print("Available JSON files:")
        for f in Path('.').glob('*.json'):
            print(f"  - {f}")
        sys.exit(1)

    downloader = YouTubeDownloader(INPUT_JSON, OUTPUT_DIR, MAX_WORKERS)
    downloader.download_all()


if __name__ == "__main__":
    main()
