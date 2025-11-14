#!/usr/bin/env python3
"""
Video Downloader Script
LEGAL NOTICE: Only use this script to download videos you have permission to download.
Respect copyright laws and website terms of service.
"""

import sys
import os
from pathlib import Path

def download_video(url, output_path=None):
    """
    Download a video from a given URL using yt-dlp

    Args:
        url: The video URL
        output_path: Directory to save the video (default: datasets/downloaded_videos)
    """
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp is not installed.")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)

    # Set default output path if not provided
    if output_path is None:
        output_path = Path("datasets/downloaded_videos")
    else:
        output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Download location: {output_path.absolute()}")

    # Configure download options
    ydl_opts = {
        'format': 'best[height<=720]',  # Download best quality up to 720p
        'outtmpl': str(output_path / '%(title)s-%(id)s.%(ext)s'),  # Output template
        'progress_hooks': [progress_hook],
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        # Save metadata
        'writeinfojson': True,
        'writethumbnail': True,
        'writedescription': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"\nDownloading video from: {url}")
            print("="*60)
            ydl.download([url])
            print("\n" + "="*60)
            print(f"✓ Download completed!")
            print(f"✓ Saved to: {output_path.absolute()}")
    except Exception as e:
        print(f"\n✗ Error downloading video: {e}")
        sys.exit(1)

def progress_hook(d):
    """Display download progress"""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A')
        speed = d.get('_speed_str', 'N/A')
        eta = d.get('_eta_str', 'N/A')
        print(f"\rProgress: {percent} | Speed: {speed} | ETA: {eta}", end='')
    elif d['status'] == 'finished':
        print("\n\nProcessing video...")

def download_multiple(url_file, output_path=None):
    """
    Download multiple videos from a file containing URLs

    Args:
        url_file: Path to file containing URLs (one per line)
        output_path: Directory to save videos
    """
    url_file = Path(url_file)

    if not url_file.exists():
        print(f"Error: File not found: {url_file}")
        sys.exit(1)

    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Found {len(urls)} URLs to download")
    print("="*60)

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Downloading: {url}")
        try:
            download_video(url, output_path)
        except Exception as e:
            print(f"Skipping {url} due to error: {e}")
            continue

def main():
    print("=" * 60)
    print("Video Downloader")
    print("=" * 60)
    print("LEGAL NOTICE: Only download videos you have permission to use.")
    print()

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  Single video:")
            print("    python download_video.py <URL> [output_path]")
            print()
            print("  Multiple videos from file:")
            print("    python download_video.py -f <url_file> [output_path]")
            print()
            print("Examples:")
            print("  python download_video.py 'https://youtube.com/watch?v=...'")
            print("  python download_video.py 'https://youtube.com/watch?v=...' datasets/my_videos")
            print("  python download_video.py -f urls.txt datasets/youtube_videos")
            print()
            print("Default output path: datasets/downloaded_videos")
            sys.exit(0)

        # Check if downloading from file
        if sys.argv[1] in ['-f', '--file']:
            if len(sys.argv) < 3:
                print("Error: Please provide URL file")
                print("Usage: python download_video.py -f <url_file> [output_path]")
                sys.exit(1)

            url_file = sys.argv[2]
            output_path = sys.argv[3] if len(sys.argv) > 3 else None
            download_multiple(url_file, output_path)
        else:
            # Single URL download
            url = sys.argv[1]
            output_path = sys.argv[2] if len(sys.argv) > 2 else None
            download_video(url, output_path)
    else:
        # Interactive mode
        print("Mode:")
        print("  1. Download single video")
        print("  2. Download from URL file")
        choice = input("\nChoose mode (1 or 2): ").strip()

        if choice == '1':
            url = input("\nEnter video URL: ").strip()
            if not url:
                print("Error: No URL provided")
                sys.exit(1)

            output_path = input("Output path (press Enter for default): ").strip()
            output_path = output_path if output_path else None

            download_video(url, output_path)

        elif choice == '2':
            url_file = input("\nEnter path to URL file: ").strip()
            if not url_file:
                print("Error: No file provided")
                sys.exit(1)

            output_path = input("Output path (press Enter for default): ").strip()
            output_path = output_path if output_path else None

            download_multiple(url_file, output_path)
        else:
            print("Invalid choice")
            sys.exit(1)

if __name__ == "__main__":
    main()
