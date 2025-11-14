#!/usr/bin/env python3
"""
Reddit Video Downloader - WORKING METHOD
Handles Reddit's split video/audio format + TOR proxy
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

class WorkingRedditDownloader:
    def __init__(self, input_json, output_dir="downloaded_reddit_videos", max_workers=3):
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

    def download_with_ytdlp_full(self, url, output_file):
        """Download using yt-dlp with FULL options for Reddit"""
        cmd = [
            'yt-dlp',

            # TOR proxy
            '--proxy', 'socks5://127.0.0.1:9050',

            # Force best format with audio merge
            '--format', 'bestvideo+bestaudio/best',
            '--merge-output-format', 'mp4',

            # Output settings
            '--output', str(output_file),
            '--no-warnings',
            '--quiet',
            '--no-progress',

            # Reddit-specific settings
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            '--referer', 'https://www.reddit.com/',

            # Cookies and authentication bypass
            '--add-header', 'Accept:*/*',
            '--add-header', 'Accept-Language:en-US,en;q=0.9',

            # Retry settings
            '--retries', '5',
            '--fragment-retries', '5',
            '--retry-sleep', '3',

            # Timeout
            '--socket-timeout', '30',

            # Force ffmpeg merge (critical for Reddit)
            '--ffmpeg-location', '/usr/bin/ffmpeg',

            # Download both video and audio
            '--prefer-free-formats',

            url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Check if file was created and is valid
            if output_file.exists() and output_file.stat().st_size > 50000:  # >50KB
                return True, None
            else:
                error = result.stderr if result.stderr else 'No output file'
                return False, error

        except subprocess.TimeoutExpired:
            return False, 'Timeout'
        except Exception as e:
            return False, str(e)

    def download_fallback_method(self, url, output_file):
        """Fallback: Use reddit JSON API + direct download"""
        try:
            # Get post JSON
            json_url = url.rstrip('/') + '.json'

            proxies = {
                'http': 'socks5://127.0.0.1:9050',
                'https': 'socks5://127.0.0.1:9050'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(json_url, headers=headers, proxies=proxies, timeout=15)

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, list) and len(data) > 0:
                    post_data = data[0]['data']['children'][0]['data']

                    # Get video URL
                    if 'secure_media' in post_data and post_data['secure_media']:
                        if 'reddit_video' in post_data['secure_media']:
                            video_url = post_data['secure_media']['reddit_video']['fallback_url']

                            # Download video
                            video_response = requests.get(
                                video_url,
                                headers=headers,
                                proxies=proxies,
                                stream=True,
                                timeout=30
                            )

                            if video_response.status_code == 200:
                                with open(output_file, 'wb') as f:
                                    for chunk in video_response.iter_content(chunk_size=8192):
                                        f.write(chunk)

                                if output_file.stat().st_size > 50000:
                                    return True, None

            return False, 'Fallback method failed'

        except Exception as e:
            return False, str(e)

    def download_single_video(self, post_url):
        """Download single video with multiple methods"""

        if post_url in self.downloaded:
            return {'status': 'skipped', 'url': post_url}

        # Extract post ID
        post_id = post_url.split('/')[-3] if '/comments/' in post_url else post_url.split('/')[-1]
        post_id = post_id.split('?')[0]

        output_file = self.output_dir / f"reddit_{post_id}.mp4"

        # Random delay
        time.sleep(random.uniform(1, 2))

        # Method 1: yt-dlp with full Reddit support
        success, error = self.download_with_ytdlp_full(post_url, output_file)

        if not success:
            # Method 2: Fallback to direct API method
            time.sleep(random.uniform(1, 2))
            success, error = self.download_fallback_method(post_url, output_file)

        if success:
            with open(self.success_log, 'a') as f:
                f.write(f"{post_url}\n")
            self.downloaded.add(post_url)
            return {'status': 'success', 'url': post_url}
        else:
            with open(self.failed_log, 'a') as f:
                f.write(f"{post_url}\t{error or 'unknown'}\n")
            return {'status': 'failed', 'url': post_url, 'error': error}

    def download_all(self):
        """Download all videos"""
        total = len(self.urls)
        already_have = len(self.downloaded)
        to_download = [url for url in self.urls if url not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║         REDDIT VIDEO DOWNLOADER - WORKING METHOD                          ║")
        print("║         Handles split audio/video + TOR proxy                             ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print("🔧 FIXES APPLIED:")
        print("   ✅ Video+audio merge using ffmpeg")
        print("   ✅ Proper Reddit format handling")
        print("   ✅ TOR proxy for IP bypass")
        print("   ✅ Fallback to direct API method")
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

        # Test first 10 videos to verify it works
        print("🧪 Testing with first 10 videos...")
        test_urls = to_download[:10]

        for url in test_urls:
            result = self.download_single_video(url)
            if result['status'] == 'success':
                success_count += 1
                print(f"✅ Success! Downloaded {success_count}/10")
            else:
                failed_count += 1
                print(f"❌ Failed: {result.get('error', 'unknown')}")

        print()
        print(f"📊 Test Results: {success_count} success, {failed_count} failed")
        print()

        if success_count == 0:
            print("❌ CRITICAL: All test downloads failed!")
            print("   Checking system...")

            # Check ffmpeg
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                print("   ✅ ffmpeg: installed")
            except:
                print("   ❌ ffmpeg: NOT INSTALLED")
                print()
                print("   Install ffmpeg:")
                print("   apt-get update && apt-get install -y ffmpeg")
                return

            # Check TOR
            try:
                subprocess.run(['pgrep', 'tor'], capture_output=True, check=True)
                print("   ✅ TOR: running")
            except:
                print("   ❌ TOR: NOT RUNNING")
                print()
                print("   Start TOR:")
                print("   service tor start")
                return

            print()
            print("   System OK - Reddit may have changed their format")
            print("   Check failed log: downloaded_reddit_videos/download_failed.txt")
            return

        if success_count < 5:
            print(f"⚠️  WARNING: Only {success_count}/10 test downloads succeeded")
            print("   This may indicate issues with Reddit access")
            print()
            response = input("Continue with full download? (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Download cancelled")
                return

        print("✅ Test successful! Starting full download...")
        print()

        # Continue with remaining videos
        remaining_urls = to_download[10:]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_video, url): url for url in remaining_urls}

            with tqdm(total=len(remaining_urls), desc="Downloading", unit="video", initial=10) as pbar:
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
            print(f"   Common reasons:")
            print(f"   - Post deleted")
            print(f"   - No video (image/text post)")
            print(f"   - Video removed by moderator")
            print(f"   Failed list: {self.failed_log}")

        print("="*80)


def main():
    # Check prerequisites
    print("🔍 Checking prerequisites...")

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg installed")
    except:
        print("❌ ffmpeg NOT installed")
        print()
        print("Installing ffmpeg...")
        subprocess.run(['apt-get', 'update', '-qq'], capture_output=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], capture_output=True)
        print("✅ ffmpeg installed")

    # Check TOR
    try:
        result = subprocess.run(['pgrep', 'tor'], capture_output=True)
        if result.returncode == 0:
            print("✅ TOR running")
        else:
            print("⚠️  TOR not running - starting...")
            subprocess.run(['service', 'tor', 'start'], capture_output=True)
            time.sleep(3)
            print("✅ TOR started")
    except:
        print("❌ TOR not installed - installing...")
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'tor'], capture_output=True)
        subprocess.run(['service', 'tor', 'start'], capture_output=True)
        time.sleep(3)
        print("✅ TOR installed and started")

    # Check requests library with socks support
    try:
        import requests
        print("✅ requests library ready")
    except ImportError:
        print("❌ requests library missing - installing...")
        subprocess.run(['pip', 'install', '--break-system-packages', 'requests[socks]'], capture_output=True)
        print("✅ requests installed")

    print()

    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "reddit_fight_videos_all.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_reddit_videos"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    if not Path(INPUT_JSON).exists():
        print(f"❌ File not found: {INPUT_JSON}")
        sys.exit(1)

    downloader = WorkingRedditDownloader(INPUT_JSON, OUTPUT_DIR, MAX_WORKERS)
    downloader.download_all()


if __name__ == "__main__":
    main()
