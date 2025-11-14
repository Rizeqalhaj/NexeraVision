#!/usr/bin/env python3
"""
Web Video Scraper - For websites with search URLs
Example: website.com/search?search=keyword
LEGAL NOTICE: For research purposes only. Respect robots.txt and terms of service.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import re
import os
from pathlib import Path
import time
import json

def scrape_video_links(base_url, search_query, max_pages=5):
    """
    Scrape video links from a website's search results

    Args:
        base_url: Base URL of the website (e.g., 'https://example.com')
        search_query: Search term
        max_pages: Maximum number of result pages to scrape

    Returns:
        List of video URLs found
    """

    video_urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"\n{'='*70}")
    print(f"Scraping: {base_url}")
    print(f"Search query: {search_query}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*70}\n")

    for page in range(1, max_pages + 1):
        # Common search URL patterns - try multiple formats
        search_urls = [
            f"{base_url}/search?search={search_query}&page={page}",
            f"{base_url}/search?q={search_query}&page={page}",
            f"{base_url}/search?query={search_query}&page={page}",
            f"{base_url}/videos/search?search={search_query}&page={page}",
            f"{base_url}?s={search_query}&page={page}",
        ]

        success = False
        for search_url in search_urls:
            try:
                print(f"Trying: {search_url}")
                response = requests.get(search_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    success = True
                    print(f"✓ Success! Processing page {page}...")

                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Find video links - common patterns
                    video_links = []

                    # Pattern 1: <a> tags with 'video' in href or class
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if any(keyword in href.lower() for keyword in ['video', 'watch', 'v/', '/v?']):
                            video_links.append(urljoin(base_url, href))

                    # Pattern 2: <video> tags with source
                    for video in soup.find_all('video'):
                        if video.get('src'):
                            video_links.append(urljoin(base_url, video['src']))
                        for source in video.find_all('source'):
                            if source.get('src'):
                                video_links.append(urljoin(base_url, source['src']))

                    # Pattern 3: Links with video file extensions
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if any(ext in href.lower() for ext in ['.mp4', '.webm', '.avi', '.mov', '.flv']):
                            video_links.append(urljoin(base_url, href))

                    print(f"  Found {len(video_links)} potential video links on page {page}")
                    video_urls.extend(video_links)

                    break  # Success, no need to try other URL formats

            except Exception as e:
                continue

        if not success:
            print(f"✗ Could not access page {page} with any URL format")

        time.sleep(2)  # Rate limiting

    # Remove duplicates
    video_urls = list(set(video_urls))
    print(f"\n✓ Total unique video links found: {len(video_urls)}")

    return video_urls

def download_video_from_url(url, output_path, filename=None):
    """Download video from direct URL"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        # Get filename from URL if not provided
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]
            if not any(filename.endswith(ext) for ext in ['.mp4', '.webm', '.avi', '.mov']):
                filename = f"{filename}.mp4"

        filepath = output_path / filename

        # Skip if already exists
        if filepath.exists():
            print(f"  ⏭️  Skip: {filename} (already exists)")
            return True

        print(f"  📥 Downloading: {filename}")

        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            percent = (downloaded / total_size) * 100
                            print(f"\r    Progress: {percent:.1f}%", end='')

            print(f"\r    ✓ Downloaded: {filename}                    ")
            return True
        else:
            print(f"    ✗ Failed: Status {response.status_code}")
            return False

    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

def scrape_and_download(base_url, search_query, max_pages=5, output_path=None):
    """
    Complete workflow: scrape links and download videos

    Args:
        base_url: Website base URL
        search_query: Search term
        max_pages: Max result pages to scrape
        output_path: Where to save videos
    """

    # Set output path
    if output_path is None:
        domain = urlparse(base_url).netloc.replace('www.', '')
        output_path = Path(f"datasets/{domain}_videos")
    else:
        output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("WEB VIDEO SCRAPER")
    print("="*70)

    # Step 1: Scrape video links
    video_urls = scrape_video_links(base_url, search_query, max_pages)

    if not video_urls:
        print("\n✗ No video links found!")
        print("\nTroubleshooting:")
        print("1. Check if the website blocks scraping")
        print("2. Verify the search URL format manually")
        print("3. The website may require JavaScript (use Selenium instead)")
        return

    # Save found URLs
    urls_file = output_path / 'found_urls.txt'
    with open(urls_file, 'w') as f:
        for url in video_urls:
            f.write(f"{url}\n")

    print(f"\n✓ Saved URLs to: {urls_file}")

    # Step 2: Try downloading with yt-dlp first (works for many sites)
    print(f"\n{'='*70}")
    print("DOWNLOADING VIDEOS")
    print(f"{'='*70}\n")

    try:
        import yt_dlp
        use_ytdlp = True
    except ImportError:
        print("⚠️  yt-dlp not installed, will try direct download")
        use_ytdlp = False

    downloaded = 0

    for i, url in enumerate(video_urls, 1):
        print(f"\n[{i}/{len(video_urls)}] {url}")

        if use_ytdlp:
            # Try yt-dlp first
            try:
                ydl_opts = {
                    'format': 'best[height<=720]',
                    'outtmpl': str(output_path / '%(title)s-%(id)s.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    downloaded += 1
                    print(f"  ✓ Downloaded with yt-dlp")
                    continue
            except:
                pass

        # Fallback: direct download
        if download_video_from_url(url, output_path):
            downloaded += 1

        time.sleep(1)  # Rate limiting

    # Summary
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Downloaded: {downloaded}/{len(video_urls)} videos")
    print(f"✓ Saved to: {output_path.absolute()}")
    print(f"✓ URL list: {urls_file}")
    print("\n" + "="*70)

def main():
    print("="*70)
    print("Web Video Scraper")
    print("="*70)
    print("For websites with search URLs like: site.com/search?search=keyword")
    print()

    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage:")
        print("  python scrape_videos_from_search.py <base_url> <search_query> [max_pages] [output_path]")
        print()
        print("Examples:")
        print("  python scrape_videos_from_search.py 'https://example.com' 'cctv fight' 10")
        print("  python scrape_videos_from_search.py 'https://example.com' 'violence' 5 datasets/example_videos")
        print()
        print("Interactive mode:")
        print("  python scrape_videos_from_search.py")
        print()
        sys.exit(0)

    if len(sys.argv) > 2:
        # Command line mode
        base_url = sys.argv[1]
        search_query = sys.argv[2]
        max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        output_path = sys.argv[4] if len(sys.argv) > 4 else None

        scrape_and_download(base_url, search_query, max_pages, output_path)

    else:
        # Interactive mode
        print("INTERACTIVE MODE")
        print("-"*70)

        base_url = input("\nEnter website URL (e.g., https://example.com): ").strip()
        if not base_url:
            print("Error: No URL provided")
            sys.exit(1)

        # Add https:// if missing
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        search_query = input("Enter search query (e.g., 'cctv fight'): ").strip()
        if not search_query:
            print("Error: No search query provided")
            sys.exit(1)

        max_pages = input("Max result pages to scrape (default 5): ").strip()
        max_pages = int(max_pages) if max_pages else 5

        output_path = input("Output path (press Enter for default): ").strip()
        output_path = output_path if output_path else None

        scrape_and_download(base_url, search_query, max_pages, output_path)

if __name__ == "__main__":
    main()
