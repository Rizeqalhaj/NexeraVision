#!/usr/bin/env python3
"""
Reddit Video Downloader - TOR Proxy Method
Bypasses IP bans by rotating through TOR exit nodes
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

class TorRedditDownloader:
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

        self.download_count = 0

    def renew_tor_ip(self):
        """Request new TOR identity (new IP)"""
        try:
            subprocess.run(['killall', '-HUP', 'tor'], capture_output=True)
            time.sleep(2)  # Wait for new circuit
        except:
            pass

    def download_with_tor(self, url, output_file):
        """Download using yt-dlp with TOR proxy"""
        cmd = [
            'yt-dlp',
            '--proxy', 'socks5://127.0.0.1:9050',  # TOR proxy
            '--no-warnings',
            '--format', 'best',
            '--output', str(output_file),
            '--quiet',
            '--no-progress',
            '--retries', '3',
            '--socket-timeout', '30',
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            return result.returncode == 0
        except:
            return False

    def download_single_video(self, post_url):
        """Download single video through TOR"""

        if post_url in self.downloaded:
            return {'status': 'skipped', 'url': post_url}

        # Extract post ID
        post_id = post_url.split('/')[-3] if '/comments/' in post_url else post_url.split('/')[-1]
        post_id = post_id.split('?')[0]

        output_file = self.output_dir / f"reddit_{post_id}.mp4"

        # Rotate TOR IP every 10 downloads
        self.download_count += 1
        if self.download_count % 10 == 0:
            self.renew_tor_ip()

        # Random delay
        time.sleep(random.uniform(1, 3))

        if self.download_with_tor(post_url, output_file):
            # Verify file exists and is valid
            if output_file.exists() and output_file.stat().st_size > 10000:
                with open(self.success_log, 'a') as f:
                    f.write(f"{post_url}\n")
                self.downloaded.add(post_url)
                return {'status': 'success', 'url': post_url}
            else:
                if output_file.exists():
                    output_file.unlink()
                with open(self.failed_log, 'a') as f:
                    f.write(f"{post_url}\tFile invalid\n")
                return {'status': 'failed', 'url': post_url}
        else:
            with open(self.failed_log, 'a') as f:
                f.write(f"{post_url}\tDownload failed\n")
            return {'status': 'failed', 'url': post_url}

    def download_all(self):
        """Download all videos through TOR"""
        total = len(self.urls)
        already_have = len(self.downloaded)
        to_download = [url for url in self.urls if url not in self.downloaded]

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║           REDDIT VIDEO DOWNLOADER - TOR PROXY METHOD                      ║")
        print("║           Bypasses IP bans using TOR network                              ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print("🛡️  TOR FEATURES:")
        print("   - Routes through TOR network (anonymous)")
        print("   - IP rotation every 10 downloads")
        print("   - Bypasses all IP-based blocks")
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
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📁 Saved to: {self.output_dir}")
        print("="*80)


def check_tor():
    """Check if TOR is installed and running"""
    # Check if TOR is installed
    try:
        result = subprocess.run(['which', 'tor'], capture_output=True)
        if result.returncode != 0:
            return False, "TOR not installed"
    except:
        return False, "TOR not found"

    # Check if TOR is running
    try:
        result = subprocess.run(['pgrep', 'tor'], capture_output=True)
        if result.returncode != 0:
            return False, "TOR not running"
    except:
        return False, "Cannot check TOR status"

    return True, "TOR ready"


def setup_tor():
    """Setup TOR if not installed"""
    print("🔧 Setting up TOR...")
    print()

    # Install TOR
    print("📦 Installing TOR...")
    subprocess.run(['apt-get', 'update', '-qq'], capture_output=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'tor'], capture_output=True)

    # Start TOR
    print("🚀 Starting TOR service...")
    subprocess.run(['service', 'tor', 'start'], capture_output=True)

    time.sleep(3)

    # Verify
    tor_ok, msg = check_tor()
    if tor_ok:
        print("✅ TOR is ready!")
        print()
        return True
    else:
        print(f"❌ TOR setup failed: {msg}")
        return False


def main():
    # Check TOR status
    tor_ok, msg = check_tor()

    if not tor_ok:
        print(f"⚠️  TOR status: {msg}")
        print()
        response = input("Install and setup TOR now? (yes/no): ")
        if response.lower() == 'yes':
            if not setup_tor():
                print("❌ Cannot proceed without TOR")
                sys.exit(1)
        else:
            print("❌ TOR is required for this method")
            sys.exit(1)

    INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "reddit_fight_videos_all.json"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "downloaded_reddit_videos"
    MAX_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    if not Path(INPUT_JSON).exists():
        print(f"❌ File not found: {INPUT_JSON}")
        sys.exit(1)

    downloader = TorRedditDownloader(INPUT_JSON, OUTPUT_DIR, MAX_WORKERS)
    downloader.download_all()


if __name__ == "__main__":
    main()
