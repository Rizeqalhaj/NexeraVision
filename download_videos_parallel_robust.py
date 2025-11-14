#!/usr/bin/env python3
"""
Parallel Video Downloader - Robust Version
Handles format errors, retries with different format options
Optimized for high-speed connections (10Gbps)
"""

import sys
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread-safe counters
class DownloadStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self.total = 0

    def increment_downloaded(self):
        with self.lock:
            self.downloaded += 1

    def increment_failed(self):
        with self.lock:
            self.failed += 1

    def increment_skipped(self):
        with self.lock:
            self.skipped += 1

    def get_stats(self):
        with self.lock:
            return self.downloaded, self.failed, self.skipped


def download_single_video(url, output_path, stats, success_log, failed_log, index, total):
    """
    Download a single video with format fallback (runs in thread)
    """

    # Multiple format strategies (try in order)
    format_strategies = [
        'best[height<=720]',           # Try 720p first
        'bestvideo[height<=720]+bestaudio/best[height<=720]',  # Separate video+audio
        'best[height<=480]',           # Fall back to 480p
        'bestvideo[height<=480]+bestaudio/best[height<=480]',
        'best',                        # Any quality
        'bestvideo+bestaudio/best',    # Separate streams
        '18',                          # YouTube format code (360p mp4)
    ]

    for attempt, format_str in enumerate(format_strategies, 1):
        try:
            # yt-dlp command with current format strategy
            cmd = [
                'yt-dlp',
                '--no-playlist',
                '--format', format_str,
                '--output', str(output_path / '%(title)s-%(id)s.%(ext)s'),
                '--no-warnings',
                '--quiet',
                '--no-check-certificate',
                '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                '--concurrent-fragments', '10',
                '--retries', '3',
                '--ignore-errors',  # Continue on errors
                '--no-abort-on-error',
                url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Check if successful
            if result.returncode == 0:
                stats.increment_downloaded()
                with open(success_log, 'a') as f:
                    f.write(f"{url}\n")

                downloaded, failed, skipped = stats.get_stats()
                completed = downloaded + failed + skipped
                print(f"[{completed}/{total}] ✓ Downloaded (format: {attempt}) | ✓{downloaded} ✗{failed} ⏭️{skipped}")
                return "success", "Downloaded"

            # Check for "already downloaded"
            elif 'has already been downloaded' in result.stdout or 'has already been downloaded' in result.stderr:
                stats.increment_skipped()
                downloaded, failed, skipped = stats.get_stats()
                completed = downloaded + failed + skipped
                print(f"[{completed}/{total}] ⏭️ Skipped (exists) | ✓{downloaded} ✗{failed} ⏭️{skipped}")
                return "skipped", "Already exists"

            # Check for format error - try next format
            elif 'Requested format is not available' in result.stderr or 'format is not available' in result.stderr:
                if attempt < len(format_strategies):
                    # Try next format
                    continue
                else:
                    # All formats failed
                    stats.increment_failed()
                    with open(failed_log, 'a') as f:
                        f.write(f"{url} (no compatible format)\n")

                    downloaded, failed, skipped = stats.get_stats()
                    completed = downloaded + failed + skipped
                    print(f"[{completed}/{total}] ✗ Failed (no format) | ✓{downloaded} ✗{failed} ⏭️{skipped}")
                    return "failed", "No compatible format"

            else:
                # Other error - try next format
                if attempt < len(format_strategies):
                    continue
                else:
                    # All attempts failed
                    stats.increment_failed()
                    error_msg = result.stderr[:100] if result.stderr else "Unknown error"
                    with open(failed_log, 'a') as f:
                        f.write(f"{url} ({error_msg})\n")

                    downloaded, failed, skipped = stats.get_stats()
                    completed = downloaded + failed + skipped
                    print(f"[{completed}/{total}] ✗ Failed | ✓{downloaded} ✗{failed} ⏭️{skipped}")
                    return "failed", error_msg

        except subprocess.TimeoutExpired:
            # Try next format on timeout (might be format issue)
            if attempt < len(format_strategies):
                continue
            else:
                stats.increment_failed()
                with open(failed_log, 'a') as f:
                    f.write(f"{url} (timeout)\n")

                downloaded, failed, skipped = stats.get_stats()
                completed = downloaded + failed + skipped
                print(f"[{completed}/{total}] ✗ Timeout | ✓{downloaded} ✗{failed} ⏭️{skipped}")
                return "failed", "Timeout"

        except Exception as e:
            # Try next format on exception
            if attempt < len(format_strategies):
                continue
            else:
                stats.increment_failed()
                with open(failed_log, 'a') as f:
                    f.write(f"{url} (error: {e})\n")

                downloaded, failed, skipped = stats.get_stats()
                completed = downloaded + failed + skipped
                print(f"[{completed}/{total}] ✗ Error | ✓{downloaded} ✗{failed} ⏭️{skipped}")
                return "failed", str(e)[:30]

    # Should never reach here
    return "failed", "Unknown"


def download_videos_parallel(url_file, output_dir, max_workers=20, max_downloads=None, start_from=0):
    """
    Download videos in parallel with robust format handling

    Args:
        url_file: Text file with URLs
        output_dir: Output directory
        max_workers: Number of parallel downloads (10-50 for 10Gbps)
        max_downloads: Max videos to download
        start_from: Skip first N URLs
    """

    # Check yt-dlp
    try:
        result = subprocess.run(['yt-dlp', '--version'],
                              capture_output=True, text=True)
        print(f"✓ yt-dlp version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ yt-dlp not found! Install: pip install yt-dlp")
        sys.exit(1)

    # Read URLs
    url_file_path = Path(url_file)
    if not url_file_path.exists():
        print(f"❌ File not found: {url_file}")
        sys.exit(1)

    with open(url_file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*70}")
    print(f"Parallel Video Downloader - Robust Format Handling")
    print(f"{'='*70}")
    print(f"URL file: {url_file}")
    print(f"Total URLs: {len(urls)}")
    print(f"Output: {output_dir}")
    print(f"Parallel downloads: {max_workers}")
    if start_from > 0:
        print(f"Starting from: URL #{start_from + 1}")
    if max_downloads:
        print(f"Max downloads: {max_downloads}")
    print(f"{'='*70}")
    print(f"\nFormat Strategy:")
    print(f"  1. Try 720p")
    print(f"  2. Try 480p")
    print(f"  3. Try any quality")
    print(f"  4. Try format code 18 (YouTube)")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare URLs
    urls_to_download = urls[start_from:]
    if max_downloads:
        urls_to_download = urls_to_download[:max_downloads]

    print(f"Will download: {len(urls_to_download)} videos")
    print(f"Speed optimization: {max_workers} simultaneous downloads\n")

    # Stats
    stats = DownloadStats()
    stats.total = len(urls_to_download)

    # Log files
    success_log = output_path / "downloaded_success.txt"
    failed_log = output_path / "downloaded_failed.txt"

    start_time = time.time()

    # Parallel download using ThreadPoolExecutor
    print(f"{'='*70}")
    print(f"DOWNLOADING...")
    print(f"{'='*70}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(
                download_single_video,
                url,
                output_path,
                stats,
                success_log,
                failed_log,
                i,
                len(urls_to_download)
            ): (i, url)
            for i, url in enumerate(urls_to_download, 1)
        }

        # Wait for completion
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread error: {e}")

    # Final summary
    total_time = time.time() - start_time
    downloaded, failed, skipped = stats.get_stats()

    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Downloaded: {downloaded} videos")
    print(f"✗ Failed: {failed} videos")
    print(f"⏭️  Skipped: {skipped} videos (already exists)")
    print(f"📁 Location: {output_path.absolute()}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    if downloaded > 0:
        print(f"⚡ Speed: {downloaded/(total_time/60):.1f} videos/minute")
    print(f"\nLogs:")
    print(f"  Success: {success_log}")
    print(f"  Failed: {failed_log}")
    print(f"{'='*70}")

    if failed > 0:
        print(f"\n⚠️  {failed} downloads failed")
        print(f"Common reasons:")
        print(f"  - Video removed/private")
        print(f"  - Geographic restrictions")
        print(f"  - Network issues")
        print(f"\nTo retry failed downloads:")
        print(f"  python {sys.argv[0]} {failed_log} {output_dir} {max_workers}")


def main():
    print("="*70)
    print("Parallel Video Downloader - Robust Format Handling")
    print("="*70)
    print("Handles format errors automatically\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python download_videos_parallel_robust.py <url_file> [output_dir] [workers] [max_downloads] [start_from]")
            print()
            print("Arguments:")
            print("  url_file      : Text file with URLs")
            print("  output_dir    : Output directory (default: downloaded_videos)")
            print("  workers       : Parallel downloads (default: 20)")
            print("  max_downloads : Max videos (default: all)")
            print("  start_from    : Skip first N URLs (default: 0)")
            print()
            print("Examples:")
            print("  # Download with 20 parallel threads")
            print("  python download_videos_parallel_robust.py video_urls.txt")
            print()
            print("  # Download with 50 parallel threads (10Gbps)")
            print("  python download_videos_parallel_robust.py video_urls.txt datasets/fights 50")
            print()
            print("  # Download first 1000 videos with 30 threads")
            print("  python download_videos_parallel_robust.py video_urls.txt datasets/fights 30 1000")
            print()
            print("Workers recommendation:")
            print("  - 1Gbps  : 10-15 workers")
            print("  - 10Gbps : 30-50 workers")
            print("  - Slower : 5-10 workers")
            print()
            print("Features:")
            print("  ✓ Auto format fallback (720p → 480p → any)")
            print("  ✓ Handles format errors gracefully")
            print("  ✓ Parallel downloads (10-50x faster)")
            print("  ✓ Auto-skip duplicates")
            print("  ✓ Thread-safe logging")
            print("  ✓ Progress tracking")
            sys.exit(0)

        url_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "downloaded_videos"
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 20

        max_downloads = None
        if len(sys.argv) > 4 and sys.argv[4].lower() != 'none':
            max_downloads = int(sys.argv[4])

        start_from = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    else:
        # Interactive
        url_file = input("URL file path: ").strip()

        output_dir = input("Output directory (default 'downloaded_videos'): ").strip()
        output_dir = output_dir if output_dir else "downloaded_videos"

        workers_str = input("Parallel downloads (default 20, recommend 30-50 for 10Gbps): ").strip()
        max_workers = int(workers_str) if workers_str else 20

        max_str = input("Max downloads (press Enter for all): ").strip()
        max_downloads = int(max_str) if max_str else None

        start_str = input("Start from URL # (default 0): ").strip()
        start_from = int(start_str) if start_str else 0

    # Confirm
    print(f"\n{'='*70}")
    print("CONFIRMATION")
    print(f"{'='*70}")
    print(f"URL file: {url_file}")
    print(f"Output: {output_dir}")
    print(f"Parallel workers: {max_workers}")
    print(f"Max downloads: {max_downloads or 'All'}")
    print(f"Start from: {start_from}")
    print(f"\n⚡ With {max_workers} parallel downloads, expect:")
    print(f"   ~{max_workers * 2}-{max_workers * 3} videos/minute")
    print(f"{'='*70}")

    confirm = input("\nStart downloading? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Download
    download_videos_parallel(url_file, output_dir, max_workers, max_downloads, start_from)


if __name__ == "__main__":
    main()
