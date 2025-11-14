#!/usr/bin/env python3
"""
Universal Video Search & Download
Search and download videos from any website that yt-dlp supports
LEGAL NOTICE: For research purposes only. Respect copyright and terms of service.
"""

import sys
import os
from pathlib import Path
import time

def search_and_download(website, search_query, max_results=20, output_path=None):
    """
    Search for videos on specified website and download them

    Args:
        website: Website/platform to search (e.g., 'youtube', 'vimeo', 'dailymotion')
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
        output_path = Path(f"datasets/{website}_downloads")
    else:
        output_path = Path(output_path)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Website: {website}")
    print(f"Search query: {search_query}")
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

    # Build search URL based on website
    website_lower = website.lower()

    # Map common website names to search formats
    if website_lower in ['youtube', 'yt']:
        search_url = f"ytsearch{max_results}:{search_query}"
    elif website_lower in ['vimeo']:
        search_url = f"vimeosearch{max_results}:{search_query}"
    elif website_lower in ['dailymotion', 'dm']:
        search_url = f"dmsearch{max_results}:{search_query}"
    elif website_lower in ['soundcloud', 'sc']:
        search_url = f"scsearch{max_results}:{search_query}"
    else:
        # Try generic search format
        search_url = f"{website_lower}search{max_results}:{search_query}"
        print(f"⚠️  Using generic search format for '{website}'")
        print(f"   If this doesn't work, provide direct URL instead")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Searching and downloading from {website}...")
            print("-" * 70)
            ydl.download([search_url])
            print("\n" + "="*70)
            print(f"✓ Completed downloads")
            print(f"✓ Saved to: {output_path.absolute()}")
            print("="*70)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if yt-dlp supports this website")
        print("2. Try providing a direct URL instead")
        print("3. Update yt-dlp: pip install -U yt-dlp")

def download_from_url(url, output_path=None):
    """
    Download video from direct URL

    Args:
        url: Direct video URL
        output_path: Directory to save video
    """
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp is not installed.")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)

    # Set default output path
    if output_path is None:
        output_path = Path("datasets/direct_downloads")
    else:
        output_path = Path(output_path)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"URL: {url}")
    print(f"Download location: {output_path.absolute()}")
    print(f"{'='*70}\n")

    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': str(output_path / '%(title)s-%(id)s.%(ext)s'),
        'quiet': False,
        'ignoreerrors': True,
        'writeinfojson': True,
        'writethumbnail': True,
        'writedescription': True,
        'restrictfilenames': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading...")
            ydl.download([url])
            print("\n✓ Download completed")
            print(f"✓ Saved to: {output_path.absolute()}")
    except Exception as e:
        print(f"\n✗ Error: {e}")

def main():
    print("="*70)
    print("Universal Video Search & Download")
    print("="*70)
    print("LEGAL NOTICE: For research purposes only.")
    print("Supports: YouTube, Vimeo, Dailymotion, and 1000+ other sites")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print()
            print("  Search mode:")
            print("    python search_download_from_any_site.py --search <website> '<query>' [max_results] [output_path]")
            print()
            print("  Direct URL mode:")
            print("    python search_download_from_any_site.py --url <video_url> [output_path]")
            print()
            print("Examples:")
            print("  python search_download_from_any_site.py --search youtube 'cctv fight' 50")
            print("  python search_download_from_any_site.py --search vimeo 'security camera' 20 datasets/vimeo")
            print("  python search_download_from_any_site.py --url 'https://youtube.com/watch?v=...'")
            print()
            print("Supported websites: YouTube, Vimeo, Dailymotion, Facebook, Twitter, and 1000+ more")
            print("Full list: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md")
            print()
            sys.exit(0)

        if sys.argv[1] == '--search':
            # Search mode
            if len(sys.argv) < 4:
                print("Error: Please provide website and search query")
                print("Usage: python search_download_from_any_site.py --search <website> '<query>' [max_results] [output_path]")
                sys.exit(1)

            website = sys.argv[2]
            query = sys.argv[3]
            max_results = int(sys.argv[4]) if len(sys.argv) > 4 else 20
            output_path = sys.argv[5] if len(sys.argv) > 5 else None

            search_and_download(website, query, max_results, output_path)

        elif sys.argv[1] == '--url':
            # Direct URL mode
            if len(sys.argv) < 3:
                print("Error: Please provide video URL")
                print("Usage: python search_download_from_any_site.py --url <video_url> [output_path]")
                sys.exit(1)

            url = sys.argv[2]
            output_path = sys.argv[3] if len(sys.argv) > 3 else None

            download_from_url(url, output_path)

        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)

    else:
        # Interactive mode
        print("Select mode:")
        print("  1. Search and download from website")
        print("  2. Download from direct URL")
        print()

        choice = input("Choose mode (1 or 2): ").strip()

        if choice == '1':
            # Search mode
            print("\n" + "-"*70)
            print("SEARCH MODE")
            print("-"*70)

            print("\nSupported websites:")
            print("  - youtube (or yt)")
            print("  - vimeo")
            print("  - dailymotion (or dm)")
            print("  - soundcloud (or sc)")
            print("  - facebook")
            print("  - twitter")
            print("  - instagram")
            print("  - tiktok")
            print("  - reddit")
            print("  - and 1000+ more")
            print()

            website = input("Enter website name: ").strip()
            if not website:
                print("Error: No website provided")
                sys.exit(1)

            query = input("Enter search query: ").strip()
            if not query:
                print("Error: No search query provided")
                sys.exit(1)

            max_results = input("Max results (default 20): ").strip()
            max_results = int(max_results) if max_results else 20

            output_path = input("Output path (press Enter for default): ").strip()
            output_path = output_path if output_path else None

            search_and_download(website, query, max_results, output_path)

        elif choice == '2':
            # Direct URL mode
            print("\n" + "-"*70)
            print("DIRECT URL MODE")
            print("-"*70)

            url = input("\nEnter video URL: ").strip()
            if not url:
                print("Error: No URL provided")
                sys.exit(1)

            output_path = input("Output path (press Enter for default): ").strip()
            output_path = output_path if output_path else None

            download_from_url(url, output_path)

        else:
            print("Invalid choice")
            sys.exit(1)

if __name__ == "__main__":
    main()
