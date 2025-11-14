#!/usr/bin/env python3
"""
URL Scraper Only - No Downloads
Just extracts video URLs and saves them to a text file
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import sys
from pathlib import Path

def scrape_video_urls(base_url, search_query, max_pages=5):
    """
    Scrape video URLs from search results - NO DOWNLOADING

    Args:
        base_url: Website base URL (e.g., 'https://example.com')
        search_query: Search keywords
        max_pages: Number of result pages to scrape

    Returns:
        List of video URLs found
    """

    video_urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"\n{'='*70}")
    print(f"Scraping URLs from: {base_url}")
    print(f"Search query: {search_query}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*70}\n")

    for page in range(1, max_pages + 1):
        # Try multiple common search URL patterns
        search_urls = [
            f"{base_url}/search?search={search_query}&page={page}",
            f"{base_url}/search?q={search_query}&page={page}",
            f"{base_url}/search?query={search_query}&page={page}",
            f"{base_url}/videos/search?search={search_query}&page={page}",
            f"{base_url}?s={search_query}&page={page}",
            f"{base_url}/browse?search={search_query}&page={page}",
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
                    page_urls = []

                    # Pattern 1: <a> tags with 'video' or 'watch' in href
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if any(keyword in href.lower() for keyword in ['video', 'watch', 'v/', '/v?']):
                            full_url = urljoin(base_url, href)
                            page_urls.append(full_url)

                    # Pattern 2: <video> tags with src
                    for video in soup.find_all('video'):
                        if video.get('src'):
                            page_urls.append(urljoin(base_url, video['src']))
                        for source in video.find_all('source'):
                            if source.get('src'):
                                page_urls.append(urljoin(base_url, source['src']))

                    # Pattern 3: Direct video file links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if any(ext in href.lower() for ext in ['.mp4', '.webm', '.avi', '.mov', '.flv', '.mkv']):
                            page_urls.append(urljoin(base_url, href))

                    print(f"  Found {len(page_urls)} URLs on page {page}")
                    video_urls.extend(page_urls)
                    break

            except Exception as e:
                continue

        if not success:
            print(f"✗ Could not access page {page}")

        time.sleep(2)  # Rate limiting

    # Remove duplicates
    video_urls = list(set(video_urls))
    print(f"\n✓ Total unique URLs found: {len(video_urls)}")

    return video_urls

def save_urls(urls, output_file):
    """Save URLs to text file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for url in urls:
            f.write(f"{url}\n")

    print(f"\n✓ URLs saved to: {output_path.absolute()}")
    print(f"✓ Total URLs: {len(urls)}")

def main():
    print("="*70)
    print("URL SCRAPER (No Downloads)")
    print("="*70)
    print("Extracts video URLs only - saves to text file\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_urls_only.py <base_url> <search_query> [max_pages] [output_file]")
            print()
            print("Examples:")
            print("  python scrape_urls_only.py 'https://example.com' 'cctv fight' 10")
            print("  python scrape_urls_only.py 'https://example.com' 'violence' 5 urls.txt")
            print()
            sys.exit(0)

        # Command line mode
        base_url = sys.argv[1]
        search_query = sys.argv[2]
        max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        output_file = sys.argv[4] if len(sys.argv) > 4 else "video_urls.txt"

    else:
        # Interactive mode
        print("INTERACTIVE MODE")
        print("-"*70)

        base_url = input("\nEnter website URL (e.g., https://example.com): ").strip()
        if not base_url:
            print("Error: No URL provided")
            sys.exit(1)

        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        search_query = input("Enter search query (e.g., 'cctv fight'): ").strip()
        if not search_query:
            print("Error: No search query provided")
            sys.exit(1)

        max_pages = input("Max pages to scrape (default 5): ").strip()
        max_pages = int(max_pages) if max_pages else 5

        output_file = input("Output filename (default 'video_urls.txt'): ").strip()
        output_file = output_file if output_file else "video_urls.txt"

    # Scrape URLs (no downloading)
    urls = scrape_video_urls(base_url, search_query, max_pages)

    if not urls:
        print("\n✗ No URLs found!")
        print("\nTroubleshooting:")
        print("1. Check if the website blocks scraping")
        print("2. Verify the search URL format manually")
        print("3. The website may require JavaScript (check in browser)")
        return

    # Save to file
    save_urls(urls, output_file)

    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"\n✓ {len(urls)} URLs extracted")
    print(f"✓ Saved to: {output_file}")
    print("\nNext steps:")
    print("1. Review the URLs in the text file")
    print("2. Filter/select which videos you want")
    print("3. Use download script on selected URLs only")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
