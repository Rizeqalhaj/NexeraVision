#!/usr/bin/env python3
"""
Video Downloader from URL List
Downloads videos from scraped URL list using yt-dlp
"""

import sys
import os
from pathlib import Path
import subprocess
import time

def download_videos(url_file, output_dir, max_downloads=None, start_from=0):
    """
    Download videos from URL list

    Args:
        url_file: Text file with one URL per line
        output_dir: Directory to save videos
        max_downloads: Maximum number to download (None = all)
        start_from: Skip first N URLs (for resuming)
    """

    # Check if yt-dlp is installed
    try:
        result = subprocess.run(['yt-dlp', '--version'],
                              capture_output=True, text=True)
        print(f"✓ yt-dlp version: {result.stdout.strip()}\n")
    except FileNotFoundError:
        print("❌ yt-dlp not found!")
        print("\nInstall with:")
        print("  pip install yt-dlp")
        print("  # or")
        print("  sudo apt install yt-dlp")
        sys.exit(1)

    # Read URLs
    url_file_path = Path(url_file)
    if not url_file_path.exists():
        print(f"❌ File not found: {url_file}")
        sys.exit(1)

    with open(url_file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"{'='*70}")
    print(f"Video Downloader")
    print(f"{'='*70}")
    print(f"URL file: {url_file}")
    print(f"Total URLs: {len(urls)}")
    print(f"Output: {output_dir}")
    if start_from > 0:
        print(f"Starting from: URL #{start_from + 1}")
    if max_downloads:
        print(f"Max downloads: {max_downloads}")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare URLs to download
    urls_to_download = urls[start_from:]
    if max_downloads:
        urls_to_download = urls_to_download[:max_downloads]

    print(f"Will download: {len(urls_to_download)} videos\n")

    # Download settings
    downloaded = 0
    failed = 0
    skipped = 0

    # Create log files
    success_log = output_path / "downloaded_success.txt"
    failed_log = output_path / "downloaded_failed.txt"
    progress_log = output_path / "download_progress.txt"

    start_time = time.time()

    for i, url in enumerate(urls_to_download, 1):
        actual_index = start_from + i

        print(f"\n{'='*70}")
        print(f"[{i}/{len(urls_to_download)}] (URL #{actual_index}/{len(urls)})")
        print(f"{'='*70}")
        print(f"URL: {url}")

        # yt-dlp command
        cmd = [
            'yt-dlp',
            '--no-playlist',
            '--format', 'best[height<=720]',  # 720p max
            '--output', str(output_path / '%(title)s-%(id)s.%(ext)s'),
            '--no-warnings',
            '--no-check-certificate',  # Some sites have SSL issues
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per video
            )

            if result.returncode == 0:
                downloaded += 1
                print(f"✓ Downloaded successfully")

                # Log success
                with open(success_log, 'a') as f:
                    f.write(f"{url}\n")
            else:
                # Check if already downloaded
                if 'has already been downloaded' in result.stdout or 'has already been downloaded' in result.stderr:
                    skipped += 1
                    print(f"⏭️  Already downloaded (skipped)")
                else:
                    failed += 1
                    print(f"✗ Failed: {result.stderr[:200]}")

                    # Log failure
                    with open(failed_log, 'a') as f:
                        f.write(f"{url}\n")

        except subprocess.TimeoutExpired:
            failed += 1
            print(f"✗ Timeout (> 5 minutes)")
            with open(failed_log, 'a') as f:
                f.write(f"{url} (timeout)\n")

        except Exception as e:
            failed += 1
            print(f"✗ Error: {e}")
            with open(failed_log, 'a') as f:
                f.write(f"{url} (error: {e})\n")

        # Progress summary
        elapsed = time.time() - start_time
        avg_time = elapsed / i if i > 0 else 0
        remaining = (len(urls_to_download) - i) * avg_time

        print(f"\nProgress: {downloaded} ✓ | {failed} ✗ | {skipped} ⏭️")
        print(f"Time: {elapsed/60:.1f}m elapsed | {remaining/60:.1f}m remaining")

        # Save progress
        with open(progress_log, 'w') as f:
            f.write(f"Last processed: URL #{actual_index}/{len(urls)}\n")
            f.write(f"Downloaded: {downloaded}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Skipped: {skipped}\n")
            f.write(f"Elapsed: {elapsed/60:.1f} minutes\n")

        # Rate limiting
        time.sleep(2)

    # Final summary
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Downloaded: {downloaded} videos")
    print(f"✗ Failed: {failed} videos")
    print(f"⏭️  Skipped: {skipped} videos (already exists)")
    print(f"📁 Location: {output_path.absolute()}")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"\nLogs:")
    print(f"  Success: {success_log}")
    print(f"  Failed: {failed_log}")
    print(f"  Progress: {progress_log}")
    print(f"{'='*70}")

    if failed > 0:
        print(f"\n⚠️  {failed} downloads failed")
        print(f"To retry failed downloads:")
        print(f"  python {sys.argv[0]} {failed_log} {output_dir}")

def main():
    print("="*70)
    print("Video Downloader from URL List")
    print("="*70)
    print("Downloads videos using yt-dlp\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python download_videos_from_urls.py <url_file> [output_dir] [max_downloads] [start_from]")
            print()
            print("Arguments:")
            print("  url_file      : Text file with URLs (one per line)")
            print("  output_dir    : Directory to save videos (default: downloaded_videos)")
            print("  max_downloads : Maximum videos to download (default: all)")
            print("  start_from    : Skip first N URLs, for resuming (default: 0)")
            print()
            print("Examples:")
            print("  # Download all videos")
            print("  python download_videos_from_urls.py video_urls.txt")
            print()
            print("  # Download to specific folder")
            print("  python download_videos_from_urls.py video_urls.txt datasets/fights")
            print()
            print("  # Download only first 100 videos")
            print("  python download_videos_from_urls.py video_urls.txt datasets/fights 100")
            print()
            print("  # Resume from URL #500 (if interrupted)")
            print("  python download_videos_from_urls.py video_urls.txt datasets/fights None 500")
            print()
            print("Features:")
            print("  ✓ 720p max quality (faster downloads, good for training)")
            print("  ✓ Auto-skip already downloaded videos")
            print("  ✓ Progress tracking and resume support")
            print("  ✓ Success/failure logs")
            print("  ✓ 5-minute timeout per video")
            sys.exit(0)

        url_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "downloaded_videos"

        max_downloads = None
        if len(sys.argv) > 3 and sys.argv[3].lower() != 'none':
            max_downloads = int(sys.argv[3])

        start_from = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    else:
        # Interactive mode
        url_file = input("URL file path: ").strip()

        output_dir = input("Output directory (default 'downloaded_videos'): ").strip()
        output_dir = output_dir if output_dir else "downloaded_videos"

        max_str = input("Max downloads (press Enter for all): ").strip()
        max_downloads = int(max_str) if max_str else None

        start_str = input("Start from URL # (default 0): ").strip()
        start_from = int(start_str) if start_str else 0

    # Confirm before starting
    print(f"\n{'='*70}")
    print("CONFIRMATION")
    print(f"{'='*70}")
    print(f"URL file: {url_file}")
    print(f"Output: {output_dir}")
    print(f"Max downloads: {max_downloads or 'All'}")
    print(f"Start from: {start_from}")
    print(f"{'='*70}")

    confirm = input("\nStart downloading? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Download
    download_videos(url_file, output_dir, max_downloads, start_from)

if __name__ == "__main__":
    main()
