#!/usr/bin/env python3
"""
Reddit Video Downloader - OPTIMIZED & FIXED
Higher success rate with better error handling
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import requests

class OptimizedRedditDownloader:
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

    def download_method_1_ytdlp(self, url, output_file):
        """Method 1: yt-dlp with optimal Reddit settings"""
        cmd = [
            'yt-dlp',
            '--proxy', 'socks5://127.0.0.1:9050',

            # Best format selection for Reddit
            '--format', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',

            # Output
            '--output', str(output_file),
            '--no-warnings',
            '--quiet',
            '--no-progress',

            # Headers
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '--referer', 'https://www.reddit.com/',
            '--add-header', 'Accept:*/*',
            '--add-header', 'Accept-Language:en-US,en;q=0.9',
            '--add-header', 'Accept-Encoding:gzip, deflate, br',

            # Retry
            '--retries', '10',
            '--fragment-retries', '10',
            '--retry-sleep', '2',

            # Timeout
            '--socket-timeout', '30',

            # Force ffmpeg
            '--ffmpeg-location', '/usr/bin/ffmpeg',

            # Skip unavailable fragments
            '--no-abort-on-error',
            '--ignore-errors',

            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if output_file.exists() and output_file.stat().st_size > 50000:
                return True, None

            return False, result.stderr[:200] if result.stderr else 'No file created'

        except Exception as e:
            return False, str(e)[:200]

    def download_method_2_direct(self, url, output_file):
        """Method 2: Direct Reddit API + v.redd.it download"""
        try:
            # Get post JSON
            json_url = url.rstrip('/') + '.json'

            session = requests.Session()
            session.proxies = {
                'http': 'socks5://127.0.0.1:9050',
                'https': 'socks5://127.0.0.1:9050'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }

            response = session.get(json_url, headers=headers, timeout=15)

            if response.status_code != 200:
                return False, f'API returned {response.status_code}'

            data = response.json()

            if not isinstance(data, list) or len(data) == 0:
                return False, 'Invalid JSON structure'

            post_data = data[0]['data']['children'][0]['data']

            # Extract video URL
            video_url = None

            # Method A: secure_media.reddit_video
            if 'secure_media' in post_data and post_data['secure_media']:
                if 'reddit_video' in post_data['secure_media']:
                    video_url = post_data['secure_media']['reddit_video']['fallback_url']

            # Method B: preview.reddit_video_preview
            if not video_url and 'preview' in post_data:
                if 'reddit_video_preview' in post_data['preview']:
                    video_url = post_data['preview']['reddit_video_preview']['fallback_url']

            # Method C: media.reddit_video
            if not video_url and 'media' in post_data and post_data['media']:
                if 'reddit_video' in post_data['media']:
                    video_url = post_data['media']['reddit_video']['fallback_url']

            if not video_url:
                return False, 'No video URL found in post'

            # Download video
            video_response = session.get(video_url, headers=headers, stream=True, timeout=30)

            if video_response.status_code != 200:
                return False, f'Video download failed: {video_response.status_code}'

            with open(output_file, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if output_file.stat().st_size > 50000:
                return True, None
            else:
                output_file.unlink()
                return False, 'Downloaded file too small'

        except Exception as e:
            return False, str(e)[:200]

    def download_method_3_nottor(self, url, output_file):
        """Method 3: yt-dlp without TOR (fallback)"""
        cmd = [
            'yt-dlp',
            '--format', 'best',
            '--output', str(output_file),
            '--quiet',
            '--no-warnings',
            '--retries', '5',
            '--socket-timeout', '20',
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)

            if output_file.exists() and output_file.stat().st_size > 50000:
                return True, None

            return False, 'Failed without TOR'

        except Exception as e:
            return False, str(e)[:200]

    def download_single_video(self, post_url):
        """Download single video with 3 methods"""

        if post_url in self.downloaded:
            return {'status': 'skipped', 'url': post_url}

        # Extract post ID
        post_id = post_url.split('/')[-3] if '/comments/' in post_url else post_url.split('/')[-1]
        post_id = post_id.split('?')[0]

        output_file = self.output_dir / f"reddit_{post_id}.mp4"

        # Random delay
        time.sleep(random.uniform(0.5, 1.5))

        # Try Method 1: yt-dlp with TOR
        success, error = self.download_method_1_ytdlp(post_url, output_file)

        if success:
            with open(self.success_log, 'a') as f:
                f.write(f"{post_url}\n")
            self.downloaded.add(post_url)
            return {'status': 'success', 'url': post_url, 'method': 'ytdlp'}

        # Try Method 2: Direct API
        time.sleep(random.uniform(0.5, 1))
        success, error2 = self.download_method_2_direct(post_url, output_file)

        if success:
            with open(self.success_log, 'a') as f:
                f.write(f"{post_url}\n")
            self.downloaded.add(post_url)
            return {'status': 'success', 'url': post_url, 'method': 'direct'}

        # Try Method 3: yt-dlp without TOR (last resort)
        time.sleep(random.uniform(0.5, 1))
        success, error3 = self.download_method_3_nottor(post_url, output_file)

        if success:
            with open(self.success_log, 'a') as f:
                f.write(f"{post_url}\n")
            self.downloaded.add(post_url)
            return {'status': 'success', 'url': post_url, 'method': 'nottor'}

        # All methods failed
        with open(self.failed_log, 'a') as f:
            f.write(f"{post_url}\tMethod1: {error} | Method2: {error2} | Method3: {error3}\n")

        return {'status': 'failed', 'url': post_url, 'error': error2}

    def download_all(self):
        """Download all videos"""
        total = len(self.urls)
        already_have = len(self.downloaded)
        to_download = [url for url in self.urls if url not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║         REDDIT VIDEO DOWNLOADER - OPTIMIZED & FIXED                       ║")
        print("║         3 download methods for maximum success rate                       ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print("🔧 DOWNLOAD METHODS (in order):")
        print("   1️⃣  yt-dlp with TOR proxy (best quality, IP bypass)")
        print("   2️⃣  Direct Reddit API + v.redd.it (reliable fallback)")
        print("   3️⃣  yt-dlp without TOR (last resort)")
        print()
        print(f"📊 Total videos: {total}")
        print(f"✅ Already downloaded: {already_have}")
        print(f"📥 To download: {len(to_download)}")
        print(f"🔧 Workers: {self.max_workers}")
        print(f"📁 Output: {self.output_dir}")
        print(f"⏱️  Estimated: ~{len(to_download) * 3 / 60:.1f} minutes")
        print()

        if len(to_download) == 0:
            print("✅ All videos already downloaded!")
            return

        success_count = 0
        failed_count = 0
        method_stats = {'ytdlp': 0, 'direct': 0, 'nottor': 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_video, url): url for url in to_download}

            with tqdm(total=len(to_download), desc="Downloading", unit="video") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'success':
                        success_count += 1
                        method = result.get('method', 'unknown')
                        method_stats[method] = method_stats.get(method, 0) + 1
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
        print()
        print("📊 Method breakdown:")
        print(f"   Method 1 (yt-dlp+TOR): {method_stats.get('ytdlp', 0)} videos")
        print(f"   Method 2 (Direct API): {method_stats.get('direct', 0)} videos")
        print(f"   Method 3 (yt-dlp only): {method_stats.get('nottor', 0)} videos")
        print()
        print(f"📁 Saved to: {self.output_dir}")
        print()

        if failed_count > 0:
            print(f"⚠️  {failed_count} videos failed")
            print(f"   Common reasons: deleted posts, no video content, removed by mods")
            print(f"   Failed list: {self.failed_log}")

        print("="*80)


def main():
    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "reddit_fight_videos_all.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_reddit_videos"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    if not Path(INPUT_JSON).exists():
        print(f"❌ File not found: {INPUT_JSON}")
        sys.exit(1)

    downloader = OptimizedRedditDownloader(INPUT_JSON, OUTPUT_DIR, MAX_WORKERS)
    downloader.download_all()


if __name__ == "__main__":
    main()
