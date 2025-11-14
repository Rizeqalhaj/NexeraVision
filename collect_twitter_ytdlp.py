#!/usr/bin/env python3
"""
Twitter Video Collection using yt-dlp
Uses yt-dlp's native Twitter support to collect videos from searches
No Playwright needed - yt-dlp handles everything
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

class TwitterYtdlpCollector:
    def __init__(self, output_dir="twitter_videos_ytdlp", cookies_file=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cookies_file = cookies_file

        # Search queries - CCTV focused
        self.queries = [
            "cctv fight",
            "security camera fight",
            "surveillance footage fight",
            "caught on camera fight",
            "cctv violence",
            "security camera violence",
            "cctv brawl",
            "cctv assault",
            "surveillance fight",
            "caught on cctv",
            "security footage violence",
            "cctv street fight",
            "security cam fight",
            "cctv attack",
            "surveillance camera violence",
            "fight caught on camera",
            "street fight",
            "fight video",
            "knockout",
            "brawl",
            "public fight",
            "real fight",
            "violent fight",
            "brutal fight",
            "bar fight",
            "parking lot fight",
        ]

        self.log_file = self.output_dir / "collection_log.txt"
        self.success_file = self.output_dir / "successful_queries.txt"

    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def collect_from_query(self, query, max_videos=200):
        """Collect videos from a single Twitter search using yt-dlp"""

        # Twitter search URL with video filter
        query_encoded = query.replace(' ', '%20')
        url = f"https://twitter.com/search?q={query_encoded}%20filter%3Avideos&f=live"

        self.log(f"\n{'='*80}")
        self.log(f"📍 Query: \"{query}\"")
        self.log(f"🌐 URL: {url}")
        self.log(f"🎯 Target: {max_videos} videos")
        self.log(f"{'='*80}\n")

        # Sanitize query for filename
        safe_query = "".join(c if c.isalnum() else '_' for c in query)
        output_template = str(self.output_dir / f"twitter_{safe_query}_%(id)s.%(ext)s")

        # Build yt-dlp command
        cmd = [
            'yt-dlp',
            '--no-warnings',
            '--ignore-errors',  # Continue on errors
            '--format', 'best',
            '--output', output_template,
            '--max-downloads', str(max_videos),
            '--write-info-json',  # Save metadata
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
            self.log("⚠️  No cookies - trying without authentication (may fail)")

        # Add URL
        cmd.append(url)

        try:
            self.log("▶️  Starting collection...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout per query
            )

            # Parse output for download count
            output = result.stdout + result.stderr

            if "Downloading" in output or "has already been downloaded" in output:
                # Count successful downloads
                downloads = output.count("Downloading") + output.count("has already been downloaded")

                self.log(f"✅ Completed: Collected ~{downloads} videos from \"{query}\"")

                # Log success
                with open(self.success_file, 'a') as f:
                    f.write(f"{query}\t{downloads}\t{datetime.now().isoformat()}\n")

                return downloads
            else:
                self.log(f"⚠️  No videos found for \"{query}\"")
                if "Sign in" in output or "login" in output.lower():
                    self.log(f"   ⚠️  Twitter requires login - provide cookies file!")
                else:
                    self.log(f"   Output: {output[:200]}")
                return 0

        except subprocess.TimeoutExpired:
            self.log(f"❌ Timeout for \"{query}\" (>15 min)")
            return 0
        except Exception as e:
            self.log(f"❌ Error for \"{query}\": {str(e)}")
            return 0

    def collect_all_queries(self, max_per_query=200):
        """Collect videos from all queries"""

        start_time = datetime.now()

        self.log("╔════════════════════════════════════════════════════════════════════════════╗")
        self.log("║              TWITTER VIDEO COLLECTOR (YT-DLP)                              ║")
        self.log("║              Direct download without Playwright                            ║")
        self.log("╚════════════════════════════════════════════════════════════════════════════╝")
        self.log("")
        self.log(f"📊 Total queries: {len(self.queries)}")
        self.log(f"🎯 Max per query: {max_per_query}")
        self.log(f"📊 Expected total: {len(self.queries) * max_per_query} videos")
        self.log(f"📁 Output directory: {self.output_dir}")
        self.log(f"🍪 Cookies: {'Yes' if self.cookies_file else 'No (will likely fail)'}")
        self.log("")

        if not self.cookies_file:
            self.log("⚠️  WARNING: Twitter requires authentication!")
            self.log("   Without cookies, this will likely fail.")
            self.log("   See TIKTOK_TWITTER_LOGIN_SOLUTION.md for cookie export guide")
            self.log("")

        total_collected = 0
        successful_queries = 0

        for idx, query in enumerate(self.queries, 1):
            self.log(f"\n📍 Progress: {idx}/{len(self.queries)} queries")

            try:
                count = self.collect_from_query(query, max_per_query)

                if count > 0:
                    total_collected += count
                    successful_queries += 1

                # Delay between queries to avoid rate limiting
                if idx < len(self.queries):
                    delay = 45  # 45 seconds between queries
                    self.log(f"\n💤 Resting {delay}s before next query...\n")
                    time.sleep(delay)

            except KeyboardInterrupt:
                self.log("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                self.log(f"❌ Error on query '{query}': {str(e)}")
                continue

        # Final statistics
        runtime = (datetime.now() - start_time).total_seconds() / 60

        self.log("\n" + "="*80)
        self.log("📊 COLLECTION COMPLETE")
        self.log("="*80)
        self.log(f"✅ Total videos collected: ~{total_collected}")
        self.log(f"✅ Successful queries: {successful_queries}/{len(self.queries)}")
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
    OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "twitter_videos_ytdlp"
    COOKIES_FILE = sys.argv[2] if len(sys.argv) > 2 else None
    MAX_PER_QUERY = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    # Check cookies file
    if COOKIES_FILE and not Path(COOKIES_FILE).exists():
        print(f"⚠️  Warning: Cookie file not found: {COOKIES_FILE}")
        print("   Twitter requires authentication - this will likely fail!")
        print()
        COOKIES_FILE = None

    if not COOKIES_FILE:
        print()
        print("="*80)
        print("⚠️  IMPORTANT: Twitter requires authentication!")
        print("="*80)
        print()
        print("Without cookies, yt-dlp cannot access Twitter search results.")
        print()
        print("To export cookies:")
        print("  1. Install browser extension: 'Get cookies.txt LOCALLY'")
        print("  2. Login to Twitter in your browser")
        print("  3. Click extension icon → Export")
        print("  4. Save as: cookies_twitter.txt")
        print("  5. Run: python3 collect_twitter_ytdlp.py twitter_videos cookies_twitter.txt")
        print()
        print("See TIKTOK_TWITTER_LOGIN_SOLUTION.md for detailed instructions")
        print("="*80)
        print()
        response = input("Try without cookies anyway? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled - please provide cookies file")
            sys.exit(0)

    print()
    print("Usage:")
    print(f"  python3 {sys.argv[0]} [output_dir] [cookies_file] [max_per_query]")
    print()
    print("Example:")
    print(f"  python3 {sys.argv[0]} twitter_videos cookies_twitter.txt 200")
    print()
    print("Current settings:")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Cookies: {COOKIES_FILE if COOKIES_FILE else 'None'}")
    print(f"  Max per query: {MAX_PER_QUERY}")
    print()

    # Run collector
    collector = TwitterYtdlpCollector(
        output_dir=OUTPUT_DIR,
        cookies_file=COOKIES_FILE
    )
    collector.collect_all_queries(max_per_query=MAX_PER_QUERY)


if __name__ == "__main__":
    main()
