#!/usr/bin/env python3
"""
Reddit Video Downloader - Direct API approach
Bypasses rate limits by using Reddit API properly
"""

import json
import time
import requests
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

class RedditAPIDownloader:
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
                self.urls = []

        # Progress tracking
        self.success_log = self.output_dir / "download_success.txt"
        self.failed_log = self.output_dir / "download_failed.txt"

        self.downloaded = set()
        if self.success_log.exists():
            with open(self.success_log, 'r') as f:
                self.downloaded = {line.strip() for line in f}

        # Session with retry logic
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_reddit_video_url(self, post_url):
        """Extract direct video URL from Reddit post"""
        try:
            # Convert to JSON API endpoint
            json_url = post_url.rstrip('/') + '.json'

            time.sleep(random.uniform(1, 3))  # Rate limiting

            response = self.session.get(json_url, timeout=15)

            if response.status_code == 429:
                return None, 'rate_limited'

            if response.status_code != 200:
                return None, f'HTTP {response.status_code}'

            data = response.json()

            # Extract video URL from JSON
            if isinstance(data, list) and len(data) > 0:
                post_data = data[0]['data']['children'][0]['data']

                # Check for v.redd.it hosted video
                if 'secure_media' in post_data and post_data['secure_media']:
                    if 'reddit_video' in post_data['secure_media']:
                        video_url = post_data['secure_media']['reddit_video']['fallback_url']
                        return video_url, None

                # Check for external video (imgur, gfycat, etc)
                if 'url' in post_data:
                    url = post_data['url']
                    if any(ext in url.lower() for ext in ['.mp4', '.webm', '.gif']):
                        return url, None

            return None, 'no_video'

        except Exception as e:
            return None, str(e)

    def download_video_file(self, video_url, output_file):
        """Download video file directly"""
        try:
            response = self.session.get(video_url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True
        except Exception:
            return False

    def download_single_video(self, post_url):
        """Download single Reddit video"""

        if post_url in self.downloaded:
            return {'status': 'skipped', 'url': post_url}

        # Extract post ID
        post_id = post_url.split('/')[-3] if '/comments/' in post_url else post_url.split('/')[-1]
        post_id = post_id.split('?')[0]

        output_file = self.output_dir / f"reddit_{post_id}.mp4"

        # Get video URL from Reddit API
        video_url, error = self.get_reddit_video_url(post_url)

        if error == 'rate_limited':
            return {'status': 'rate_limited', 'url': post_url}

        if error or not video_url:
            with open(self.failed_log, 'a') as f:
                f.write(f"{post_url}\t{error or 'no_video'}\n")
            return {'status': 'failed', 'url': post_url, 'error': error or 'no_video'}

        # Download video file
        time.sleep(random.uniform(1, 2))

        if self.download_video_file(video_url, output_file):
            with open(self.success_log, 'a') as f:
                f.write(f"{post_url}\n")
            self.downloaded.add(post_url)
            return {'status': 'success', 'url': post_url}
        else:
            with open(self.failed_log, 'a') as f:
                f.write(f"{post_url}\tDownload failed\n")
            return {'status': 'failed', 'url': post_url, 'error': 'download_failed'}

    def download_all(self):
        """Download all videos"""
        total = len(self.urls)
        already_have = len(self.downloaded)
        to_download = [url for url in self.urls if url not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              REDDIT VIDEO DOWNLOADER - Direct API                          ║")
        print("║              Uses Reddit JSON API - More reliable                          ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total videos: {total}")
        print(f"✅ Already downloaded: {already_have}")
        print(f"📥 To download: {len(to_download)}")
        print(f"🔧 Workers: {self.max_workers}")
        print(f"📁 Output: {self.output_dir}")
        print(f"⏱️  Estimated: ~{len(to_download) * 4 / 60:.1f} minutes")
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
                        print(f"\n⚠️  Rate limited - pausing 60s...")
                        time.sleep(60)
                    elif result['status'] == 'failed':
                        failed_count += 1

                    pbar.set_postfix({'✅': success_count, '❌': failed_count, '⏸️': rate_limited_count})
                    pbar.update(1)

        print()
        print("="*80)
        print("📊 COMPLETE")
        print("="*80)
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⏸️  Rate limited: {rate_limited_count}")
        print("="*80)


def main():
    import sys
    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "reddit_fight_videos_all.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_reddit_videos"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    if not Path(INPUT_JSON).exists():
        print(f"❌ File not found: {INPUT_JSON}")
        return

    downloader = RedditAPIDownloader(INPUT_JSON, OUTPUT_DIR, MAX_WORKERS)
    downloader.download_all()


if __name__ == "__main__":
    main()
