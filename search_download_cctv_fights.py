#!/usr/bin/env python3
"""
CCTV Fight Video Search & Download
Automatically searches and downloads CCTV fight footage from YouTube
LEGAL NOTICE: For research purposes only. Respect copyright and terms of service.
"""

import sys
import os
from pathlib import Path
import time

def search_and_download(search_query, max_results=20, output_path=None):
    """
    Search for videos and download them

    Args:
        search_query: Search term
        max_results: Maximum number of videos to download
        output_path: Directory to save videos
    """
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp is not installed.")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)

    # Set default output path
    if output_path is None:
        output_path = Path("datasets/cctv_fights")
    else:
        output_path = Path(output_path)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Searching: {search_query}")
    print(f"Max results: {max_results}")
    print(f"Download location: {output_path.absolute()}")
    print(f"{'='*70}\n")

    # Configure download options
    ydl_opts = {
        'format': 'best[height<=720]',  # 720p max for faster downloads
        'outtmpl': str(output_path / '%(title)s-%(id)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': True,  # Skip errors and continue
        'max_downloads': max_results,
        # Save metadata
        'writeinfojson': True,
        'writethumbnail': True,
        'writedescription': True,
        'restrictfilenames': True,  # Remove special characters
    }

    # Search query format for yt-dlp
    search_url = f"ytsearch{max_results}:{search_query}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading {max_results} videos for: '{search_query}'")
            print("-" * 70)
            ydl.download([search_url])
            print("\n" + "="*70)
            print(f"✓ Completed downloads for: {search_query}")
            print(f"✓ Saved to: {output_path.absolute()}")
            print("="*70)
    except Exception as e:
        print(f"\n✗ Error: {e}")

def download_all_keywords(max_per_keyword=20, output_path=None):
    """
    Download videos for all predefined CCTV fight keywords

    Args:
        max_per_keyword: Max videos per search term
        output_path: Base directory to save videos
    """

    # Comprehensive CCTV fight keywords
    KEYWORDS = [
        # Generic CCTV fights
        "cctv fight",
        "security camera fight",
        "surveillance camera fight",
        "caught on camera fight",

        # Location-specific
        "gas station fight cctv",
        "convenience store fight camera",
        "parking lot fight surveillance",
        "street fight security camera",
        "bar fight cctv",
        "restaurant fight security camera",
        "mall fight surveillance",
        "subway fight cctv",

        # Action-specific
        "cctv brawl",
        "security footage fight",
        "surveillance assault",
        "cctv attack",
        "security camera violence",

        # Quality indicators
        "cctv fight footage",
        "security camera fight compilation",
        "real cctv fights",
    ]

    print("="*70)
    print("CCTV FIGHT VIDEO BULK DOWNLOADER")
    print("="*70)
    print(f"\nTotal search keywords: {len(KEYWORDS)}")
    print(f"Max videos per keyword: {max_per_keyword}")
    print(f"Expected total: ~{len(KEYWORDS) * max_per_keyword} videos")
    print(f"\nThis will take approximately {len(KEYWORDS) * 2} minutes")
    print("="*70)

    total_downloaded = 0

    for i, keyword in enumerate(KEYWORDS, 1):
        print(f"\n[{i}/{len(KEYWORDS)}] Processing: {keyword}")

        try:
            search_and_download(keyword, max_per_keyword, output_path)
            time.sleep(2)  # Rate limiting
            total_downloaded += max_per_keyword
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user")
            break
        except Exception as e:
            print(f"Error with '{keyword}': {e}")
            continue

    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Total keywords processed: {i}")
    print(f"Expected downloads: ~{total_downloaded}")
    print(f"Location: {output_path or 'datasets/cctv_fights'}")
    print("="*70)

def main():
    print("="*70)
    print("CCTV Fight Video Downloader")
    print("="*70)
    print("LEGAL NOTICE: For research purposes only.")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  Download with all keywords (recommended):")
            print("    python search_download_cctv_fights.py")
            print("    python search_download_cctv_fights.py [output_path]")
            print()
            print("  Custom search:")
            print("    python search_download_cctv_fights.py --search 'your query' [max_results] [output_path]")
            print()
            print("  Bulk download:")
            print("    python search_download_cctv_fights.py --bulk [max_per_keyword] [output_path]")
            print()
            print("Examples:")
            print("  python search_download_cctv_fights.py")
            print("  python search_download_cctv_fights.py datasets/my_cctv_videos")
            print("  python search_download_cctv_fights.py --search 'gas station fight cctv' 50")
            print("  python search_download_cctv_fights.py --bulk 30 datasets/bulk_cctv")
            print()
            sys.exit(0)

        if sys.argv[1] == '--search':
            # Custom search
            if len(sys.argv) < 3:
                print("Error: Please provide search query")
                print("Usage: python search_download_cctv_fights.py --search 'query' [max_results] [output_path]")
                sys.exit(1)

            query = sys.argv[2]
            max_results = int(sys.argv[3]) if len(sys.argv) > 3 else 20
            output_path = sys.argv[4] if len(sys.argv) > 4 else None

            search_and_download(query, max_results, output_path)

        elif sys.argv[1] == '--bulk':
            # Bulk download with all keywords
            max_per_keyword = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            output_path = sys.argv[3] if len(sys.argv) > 3 else None

            download_all_keywords(max_per_keyword, output_path)

        else:
            # Assume it's an output path
            output_path = sys.argv[1]
            download_all_keywords(max_per_keyword=20, output_path=output_path)

    else:
        # Interactive mode
        print("Select download mode:")
        print("  1. Quick download (20 videos per keyword, ~400 total)")
        print("  2. Medium download (50 videos per keyword, ~1000 total)")
        print("  3. Large download (100 videos per keyword, ~2000 total)")
        print("  4. Custom search")
        print()

        choice = input("Choose mode (1-4): ").strip()

        if choice == '1':
            max_per_keyword = 20
        elif choice == '2':
            max_per_keyword = 50
        elif choice == '3':
            max_per_keyword = 100
        elif choice == '4':
            query = input("\nEnter search query: ").strip()
            max_results = input("Max results (default 20): ").strip()
            max_results = int(max_results) if max_results else 20

            output_path = input("Output path (press Enter for default): ").strip()
            output_path = output_path if output_path else None

            search_and_download(query, max_results, output_path)
            sys.exit(0)
        else:
            print("Invalid choice")
            sys.exit(1)

        output_path = input("\nOutput path (press Enter for default 'datasets/cctv_fights'): ").strip()
        output_path = output_path if output_path else None

        print(f"\nStarting download with {max_per_keyword} videos per keyword...")
        print("Press Ctrl+C to stop at any time\n")

        time.sleep(2)  # Give user time to read

        download_all_keywords(max_per_keyword, output_path)

if __name__ == "__main__":
    main()
