#!/usr/bin/env python3
"""
Download YouTube Shorts videos
Uses yt-dlp optimized for Shorts format
"""

import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class YouTubeShortsDownloader:
    def __init__(self, input_json, output_dir="downloaded_youtube_shorts", max_workers=8):
        self.input_json = Path(input_json)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers

        # Load scraped shorts
        with open(self.input_json, 'r') as f:
            self.shorts_data = json.load(f)

        # Track progress
        self.success_log = self.output_dir / "download_success.txt"
        self.failed_log = self.output_dir / "download_failed.txt"

        # Load already downloaded
        self.downloaded = set()
        if self.success_log.exists():
            with open(self.success_log, 'r') as f:
                self.downloaded = {line.strip() for line in f}

    def download_single_short(self, short_info):
        """Download a single YouTube Short"""
        url = short_info['url']
        title = short_info.get('title', 'Unknown')
        video_id = short_info['videoId']

        # Skip if already downloaded
        if url in self.downloaded:
            return {'status': 'skipped', 'url': url}

        # Sanitize filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]

        # Output template
        output_template = str(self.output_dir / f"short_{video_id}_{safe_title}.%(ext)s")

        # yt-dlp command optimized for Shorts
        cmd = [
            'yt-dlp',
            '--no-warnings',
            '--format', 'best[height<=1080][ext=mp4]/best[ext=mp4]/best',  # Shorts are usually vertical 1080p
            '--output', output_template,
            '--quiet',
            '--no-progress',
            '--retries', '3',
            '--fragment-retries', '3',
            '--retry-sleep', '2',
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per short
            )

            if result.returncode == 0:
                # Log success
                with open(self.success_log, 'a') as f:
                    f.write(f"{url}\n")
                self.downloaded.add(url)
                return {'status': 'success', 'url': url, 'title': title}
            else:
                error = result.stderr.strip() if result.stderr else 'Unknown error'
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
        """Download all shorts in parallel"""
        total = len(self.shorts_data)
        already_have = len(self.downloaded)
        to_download = [s for s in self.shorts_data if s['url'] not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                    YOUTUBE SHORTS DOWNLOADER                               ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total shorts in JSON: {total}")
        print(f"✅ Already downloaded: {already_have}")
        print(f"📥 To download: {len(to_download)}")
        print(f"🔧 Parallel workers: {self.max_workers}")
        print(f"📁 Output directory: {self.output_dir}")
        print()

        if len(to_download) == 0:
            print("✅ All shorts already downloaded!")
            return

        # Download in parallel
        success_count = 0
        failed_count = 0
        skipped_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.download_single_short, short): short for short in to_download}

            # Progress bar
            with tqdm(total=len(to_download), desc="Downloading", unit="short") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'success':
                        success_count += 1
                        pbar.set_postfix({'✅': success_count, '❌': failed_count})
                    elif result['status'] == 'failed':
                        failed_count += 1
                        pbar.set_postfix({'✅': success_count, '❌': failed_count})
                    else:
                        skipped_count += 1

                    pbar.update(1)

                    # Small delay
                    time.sleep(0.05)

        # Final statistics
        print()
        print("="*80)
        print("📊 DOWNLOAD COMPLETE")
        print("="*80)
        print(f"✅ Successfully downloaded: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⏭️  Skipped (already have): {skipped_count + already_have}")
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
        print("  pip install yt-dlp")
        print("  # or")
        print("  pip install --break-system-packages yt-dlp")
        sys.exit(1)

    # Configuration
    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "youtube_shorts_fights.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_youtube_shorts"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    # Check if input exists
    if not Path(INPUT_JSON).exists():
        print(f"❌ ERROR: Input file not found: {INPUT_JSON}")
        print()
        print("Usage:")
        print(f"  python3 {sys.argv[0]} <input_json> [output_dir] [max_workers]")
        print()
        print("Example:")
        print(f"  python3 {sys.argv[0]} youtube_shorts_fights.json downloaded_youtube_shorts 8")
        sys.exit(1)

    # Run downloader
    downloader = YouTubeShortsDownloader(
        input_json=INPUT_JSON,
        output_dir=OUTPUT_DIR,
        max_workers=MAX_WORKERS
    )
    downloader.download_all()


if __name__ == "__main__":
    main()
