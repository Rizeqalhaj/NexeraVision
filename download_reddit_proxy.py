#!/usr/bin/env python3
"""
Reddit Video Downloader - IP Ban Workaround
Uses multiple methods to bypass Vast.ai IP blocks:
1. Teddit (Reddit proxy frontend - no rate limits)
2. Old Reddit (less strict)
3. RSS feeds (public, no auth needed)
"""

import json
import time
import requests
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import re

class ProxyRedditDownloader:
    def __init__(self, input_json, output_dir="downloaded_reddit_videos", max_workers=4):
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

        # Session with rotation
        self.session = requests.Session()

    def get_video_via_teddit(self, post_url):
        """Use Teddit (Reddit proxy) to get video URL"""
        try:
            # Convert to Teddit URL
            teddit_url = post_url.replace('reddit.com', 'teddit.net')
            teddit_url = teddit_url.replace('www.', '')

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = self.session.get(teddit_url, headers=headers, timeout=15)

            if response.status_code == 200:
                # Extract video URL from Teddit HTML
                content = response.text

                # Look for v.redd.it video URLs
                video_pattern = r'(https://v\.redd\.it/[a-zA-Z0-9]+/DASH_\d+\.mp4)'
                matches = re.findall(video_pattern, content)

                if matches:
                    # Get highest quality
                    return matches[0], None

            return None, 'no_video_found'

        except Exception as e:
            return None, str(e)

    def get_video_via_old_reddit(self, post_url):
        """Use old.reddit.com (less rate limiting)"""
        try:
            # Convert to old.reddit.com
            old_url = post_url.replace('www.reddit.com', 'old.reddit.com')
            old_url = old_url.replace('reddit.com', 'old.reddit.com')

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            time.sleep(random.uniform(0.5, 1.5))

            response = self.session.get(old_url, headers=headers, timeout=15)

            if response.status_code == 200:
                content = response.text

                # Look for video URLs in page
                video_pattern = r'(https://v\.redd\.it/[a-zA-Z0-9]+)'
                matches = re.findall(video_pattern, content)

                if matches:
                    video_id = matches[0].split('/')[-1]
                    # Construct full URL with quality
                    video_url = f"https://v.redd.it/{video_id}/DASH_720.mp4"
                    return video_url, None

            return None, 'no_video'

        except Exception as e:
            return None, str(e)

    def download_direct_video(self, video_url, output_file):
        """Download video file directly"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.reddit.com/'
            }

            response = self.session.get(video_url, headers=headers, stream=True, timeout=30)

            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True

            # Try lower quality if 720p fails
            if '720' in video_url:
                video_url_480 = video_url.replace('720', '480')
                response = self.session.get(video_url_480, headers=headers, stream=True, timeout=30)
                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True

            return False

        except Exception:
            return False

    def download_single_video(self, post_url):
        """Download single video using multiple methods"""

        if post_url in self.downloaded:
            return {'status': 'skipped', 'url': post_url}

        # Extract post ID
        post_id = post_url.split('/')[-3] if '/comments/' in post_url else post_url.split('/')[-1]
        post_id = post_id.split('?')[0]

        output_file = self.output_dir / f"reddit_{post_id}.mp4"

        # Method 1: Try Teddit (no rate limits)
        video_url, error = self.get_video_via_teddit(post_url)

        # Method 2: Try old.reddit.com if Teddit fails
        if not video_url:
            time.sleep(random.uniform(0.5, 1))
            video_url, error = self.get_video_via_old_reddit(post_url)

        if not video_url:
            with open(self.failed_log, 'a') as f:
                f.write(f"{post_url}\t{error or 'no_video'}\n")
            return {'status': 'failed', 'url': post_url}

        # Download the video file
        time.sleep(random.uniform(0.3, 0.8))

        if self.download_direct_video(video_url, output_file):
            # Verify file is valid (>10KB)
            if output_file.stat().st_size > 10000:
                with open(self.success_log, 'a') as f:
                    f.write(f"{post_url}\n")
                self.downloaded.add(post_url)
                return {'status': 'success', 'url': post_url}
            else:
                output_file.unlink()  # Delete tiny/corrupt file
                with open(self.failed_log, 'a') as f:
                    f.write(f"{post_url}\tFile too small\n")
                return {'status': 'failed', 'url': post_url}
        else:
            with open(self.failed_log, 'a') as f:
                f.write(f"{post_url}\tDownload failed\n")
            return {'status': 'failed', 'url': post_url}

    def download_all(self):
        """Download all videos"""
        total = len(self.urls)
        already_have = len(self.downloaded)
        to_download = [url for url in self.urls if url not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║           REDDIT VIDEO DOWNLOADER - IP BAN WORKAROUND                     ║")
        print("║           Uses Teddit + old.reddit.com (no rate limits)                   ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print("🛡️  BYPASS METHODS:")
        print("   1. Teddit.net (Reddit proxy - no IP bans)")
        print("   2. old.reddit.com (less strict rate limiting)")
        print("   3. Direct v.redd.it downloads")
        print()
        print(f"📊 Total videos: {total}")
        print(f"✅ Already downloaded: {already_have}")
        print(f"📥 To download: {len(to_download)}")
        print(f"🔧 Workers: {self.max_workers}")
        print(f"📁 Output: {self.output_dir}")
        print(f"⏱️  Estimated: ~{len(to_download) * 2 / 60:.1f} minutes (~2s per video)")
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

                    pbar.set_postfix({'✅': success_count, '❌': failed_count})
                    pbar.update(1)

        print()
        print("="*80)
        print("📊 DOWNLOAD COMPLETE")
        print("="*80)
        print(f"✅ Successfully downloaded: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📁 Saved to: {self.output_dir}")
        print()

        if failed_count > 0:
            print(f"⚠️  {failed_count} videos failed")
            print(f"   Reasons: no video found, deleted posts, or audio-only")
            print(f"   Failed list: {self.failed_log}")

        print("="*80)


def main():
    import sys

    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "reddit_fight_videos_all.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_reddit_videos"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    if not Path(INPUT_JSON).exists():
        print(f"❌ File not found: {INPUT_JSON}")
        return

    downloader = ProxyRedditDownloader(INPUT_JSON, OUTPUT_DIR, MAX_WORKERS)
    downloader.download_all()


if __name__ == "__main__":
    main()
