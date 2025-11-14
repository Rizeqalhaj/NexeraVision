#!/usr/bin/env python3
"""
TikTok Video Collection using yt-dlp
Uses yt-dlp's native TikTok support to collect videos from hashtags
No Playwright needed - yt-dlp handles everything
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

class TikTokYtdlpCollector:
    def __init__(self, output_dir="tiktok_videos_ytdlp", cookies_file=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cookies_file = cookies_file

        # TikTok hashtags - CCTV focused
        self.hashtags = [
            "cctv",
            "cctvfootage",
            "securitycamera",
            "securityfootage",
            "surveillance",
            "surveillancecamera",
            "caughtoncamera",
            "caughtoncctv",
            "cctvfight",
            "securitycamerafight",
            "fight",
            "streetfight",
            "fightvideos",
            "fightcaughtoncamera",
            "knockout",
            "brawl",
            "fighting",
            "realfight",
            "violentfight",
            "brutalfight",
            "publicfight",
            "parkingfight",
            "gasstationfight",
            "storefight",
            "barfight",
        ]

        self.log_file = self.output_dir / "collection_log.txt"
        self.success_file = self.output_dir / "successful_hashtags.txt"

    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def collect_from_hashtag(self, hashtag, max_videos=300):
        """Collect videos from a single TikTok hashtag using yt-dlp"""

        url = f"https://www.tiktok.com/tag/{hashtag}"

        self.log(f"\n{'='*80}")
        self.log(f"📍 Hashtag: #{hashtag}")
        self.log(f"🌐 URL: {url}")
        self.log(f"🎯 Target: {max_videos} videos")
        self.log(f"{'='*80}\n")

        # Output template for this hashtag
        output_template = str(self.output_dir / f"tiktok_{hashtag}_%(id)s.%(ext)s")

        # Build yt-dlp command
        cmd = [
            'yt-dlp',
            '--no-warnings',
            '--ignore-errors',  # Continue on errors
            '--format', 'best',
            '--output', output_template,
            '--max-downloads', str(max_videos),
            '--no-playlist',  # Don't treat as playlist
            '--write-info-json',  # Save metadata
            '--write-description',  # Save description
            '--no-overwrites',  # Skip if already downloaded
            '--retries', '3',
            '--fragment-retries', '3',
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        ]

        # Add cookies if provided
        if self.cookies_file and Path(self.cookies_file).exists():
            cmd.extend(['--cookies', str(self.cookies_file)])
            self.log("🍪 Using cookie file for authentication")
        else:
            self.log("⚠️  No cookies - trying without authentication")

        # Add URL
        cmd.append(url)

        try:
            self.log("▶️  Starting collection...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per hashtag
            )

            # Parse output for download count
            output = result.stdout + result.stderr

            if "Downloading" in output or "has already been downloaded" in output:
                # Count successful downloads
                downloads = output.count("Downloading") + output.count("has already been downloaded")

                self.log(f"✅ Completed: Collected ~{downloads} videos from #{hashtag}")

                # Log success
                with open(self.success_file, 'a') as f:
                    f.write(f"{hashtag}\t{downloads}\t{datetime.now().isoformat()}\n")

                return downloads
            else:
                self.log(f"⚠️  No videos found for #{hashtag}")
                self.log(f"   Output: {output[:200]}")
                return 0

        except subprocess.TimeoutExpired:
            self.log(f"❌ Timeout for #{hashtag} (>30 min)")
            return 0
        except Exception as e:
            self.log(f"❌ Error for #{hashtag}: {str(e)}")
            return 0

    def collect_all_hashtags(self, max_per_hashtag=300):
        """Collect videos from all hashtags"""

        start_time = datetime.now()

        self.log("╔════════════════════════════════════════════════════════════════════════════╗")
        self.log("║              TIKTOK VIDEO COLLECTOR (YT-DLP)                               ║")
        self.log("║              Direct download without Playwright                            ║")
        self.log("╚════════════════════════════════════════════════════════════════════════════╝")
        self.log("")
        self.log(f"📊 Total hashtags: {len(self.hashtags)}")
        self.log(f"🎯 Max per hashtag: {max_per_hashtag}")
        self.log(f"📊 Expected total: {len(self.hashtags) * max_per_hashtag} videos")
        self.log(f"📁 Output directory: {self.output_dir}")
        self.log(f"🍪 Cookies: {'Yes' if self.cookies_file else 'No (may limit results)'}")
        self.log("")

        total_collected = 0
        successful_hashtags = 0

        for idx, hashtag in enumerate(self.hashtags, 1):
            self.log(f"\n📍 Progress: {idx}/{len(self.hashtags)} hashtags")

            try:
                count = self.collect_from_hashtag(hashtag, max_per_hashtag)

                if count > 0:
                    total_collected += count
                    successful_hashtags += 1

                # Delay between hashtags to avoid rate limiting
                if idx < len(self.hashtags):
                    delay = 60  # 1 minute between hashtags
                    self.log(f"\n💤 Resting {delay}s before next hashtag...\n")
                    time.sleep(delay)

            except KeyboardInterrupt:
                self.log("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                self.log(f"❌ Error on hashtag '{hashtag}': {str(e)}")
                continue

        # Final statistics
        runtime = (datetime.now() - start_time).total_seconds() / 60

        self.log("\n" + "="*80)
        self.log("📊 COLLECTION COMPLETE")
        self.log("="*80)
        self.log(f"✅ Total videos collected: ~{total_collected}")
        self.log(f"✅ Successful hashtags: {successful_hashtags}/{len(self.hashtags)}")
        self.log(f"⏱️  Total runtime: {runtime:.1f} minutes ({runtime/60:.1f} hours)")
        self.log(f"📁 Videos saved to: {self.output_dir}")
        self.log("="*80)

        # Count actual files
        video_files = list(self.output_dir.glob("*.mp4")) + list(self.output_dir.glob("*.webm"))
        self.log(f"\n📊 Actual video files on disk: {len(video_files)}")


def main():
    import sys

    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: yt-dlp not installed")
        print()
        print("Install with:")
        print("  pip install --break-system-packages yt-dlp")
        sys.exit(1)

    # Configuration
    OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "tiktok_videos_ytdlp"
    COOKIES_FILE = sys.argv[2] if len(sys.argv) > 2 else None
    MAX_PER_HASHTAG = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    # Check cookies file
    if COOKIES_FILE and not Path(COOKIES_FILE).exists():
        print(f"⚠️  Warning: Cookie file not found: {COOKIES_FILE}")
        print("   Continuing without cookies (may limit results)")
        print()
        COOKIES_FILE = None

    print()
    print("Usage:")
    print(f"  python3 {sys.argv[0]} [output_dir] [cookies_file] [max_per_hashtag]")
    print()
    print("Example:")
    print(f"  python3 {sys.argv[0]} tiktok_videos cookies_tiktok.txt 300")
    print()
    print("Current settings:")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Cookies: {COOKIES_FILE if COOKIES_FILE else 'None'}")
    print(f"  Max per hashtag: {MAX_PER_HASHTAG}")
    print()

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        sys.exit(0)

    # Run collector
    collector = TikTokYtdlpCollector(
        output_dir=OUTPUT_DIR,
        cookies_file=COOKIES_FILE
    )
    collector.collect_all_hashtags(max_per_hashtag=MAX_PER_HASHTAG)


if __name__ == "__main__":
    main()
