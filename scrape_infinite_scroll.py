#!/usr/bin/env python3
"""
Infinite Scroll Web Scraper
For websites that load content as you scroll (no page 1, 2, 3...)
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin

def scrape_infinite_scroll_playwright(base_url, search_query, num_scrolls=10):
    """
    Scrape URLs from infinite scroll page using Playwright
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        print("\nInstall with:")
        print("  pip install playwright")
        print("  playwright install chromium")
        sys.exit(1)

    video_urls = set()

    print(f"\n{'='*70}")
    print(f"Infinite Scroll Scraper")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"Scrolls: {num_scrolls}")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        print("🌐 Starting browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = context.new_page()

        # Build search URL
        search_url = f"{base_url}/search?search={search_query}"

        # Try alternate formats if first doesn't work
        search_url_formats = [
            f"{base_url}/search?search={search_query}",
            f"{base_url}/search?q={search_query}",
            f"{base_url}/browse?search={search_query}",
            f"{base_url}/videos?search={search_query}",
            f"{base_url}?s={search_query}",
        ]

        page_loaded = False
        for url_format in search_url_formats:
            try:
                print(f"Trying: {url_format}")
                page.goto(url_format, timeout=15000)
                page.wait_for_timeout(2000)

                if "404" not in page.title().lower() and "not found" not in page.content().lower():
                    search_url = url_format
                    page_loaded = True
                    print(f"✓ Page loaded successfully")
                    break
            except:
                continue

        if not page_loaded:
            print("❌ Could not load search page")
            browser.close()
            return []

        # Scroll and collect URLs
        for scroll_num in range(1, num_scrolls + 1):
            print(f"\n🔄 Scroll {scroll_num}/{num_scrolls}")

            # Extract URLs before scrolling
            current_count = len(video_urls)

            # Method 1: <a> tags with video keywords
            links = page.query_selector_all('a[href]')
            for link in links:
                try:
                    href = link.get_attribute('href')
                    if href and any(kw in href.lower() for kw in ['video', 'watch', '/v/', '/v?', 'play']):
                        full_url = urljoin(base_url, href)
                        video_urls.add(full_url)
                except:
                    pass

            # Method 2: <video> tags
            videos = page.query_selector_all('video')
            for video in videos:
                try:
                    src = video.get_attribute('src')
                    if src:
                        video_urls.add(urljoin(base_url, src))

                    # Check source tags inside video
                    sources = video.query_selector_all('source')
                    for source in sources:
                        src = source.get_attribute('src')
                        if src:
                            video_urls.add(urljoin(base_url, src))
                except:
                    pass

            # Method 3: data-src attributes
            data_elements = page.query_selector_all('[data-src]')
            for elem in data_elements:
                try:
                    data_src = elem.get_attribute('data-src')
                    if data_src and any(ext in data_src.lower() for ext in ['.mp4', '.webm', 'video']):
                        video_urls.add(urljoin(base_url, data_src))
                except:
                    pass

            # Method 4: Direct video file links
            all_links = page.query_selector_all('a[href]')
            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href and any(ext in href.lower() for ext in ['.mp4', '.webm', '.avi', '.mov', '.flv', '.mkv']):
                        video_urls.add(urljoin(base_url, href))
                except:
                    pass

            new_found = len(video_urls) - current_count
            print(f"   Found {new_found} new URLs (Total: {len(video_urls)})")

            # Scroll down to load more content
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Wait for new content to load
            page.wait_for_timeout(3000)  # 3 seconds for content to load

            # Optional: Check if we've reached the end
            # If no new URLs found for 2 consecutive scrolls, stop
            if scroll_num > 2 and new_found == 0:
                prev_scroll_urls = page.evaluate("document.querySelectorAll('a').length")
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(2000)
                current_scroll_urls = page.evaluate("document.querySelectorAll('a').length")

                if prev_scroll_urls == current_scroll_urls:
                    print("\n✓ Reached end of page (no more content loading)")
                    break

        browser.close()
        print("\n🔒 Browser closed")

    video_urls = list(video_urls)
    print(f"\n{'='*70}")
    print(f"✓ Total unique URLs: {len(video_urls)}")
    print(f"{'='*70}")

    return video_urls

def scrape_infinite_scroll_simple(base_url, search_query):
    """
    Simple method for infinite scroll (limited - can't scroll without browser)
    This just gets whatever is in the initial page load
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("❌ requests or beautifulsoup4 not installed!")
        sys.exit(1)

    video_urls = set()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"\n{'='*70}")
    print(f"Simple Scraper (Initial Load Only)")
    print(f"⚠️  Warning: Can't scroll, only gets first batch of videos")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"{'='*70}\n")

    search_urls = [
        f"{base_url}/search?search={search_query}",
        f"{base_url}/search?q={search_query}",
        f"{base_url}/browse?search={search_query}",
        f"{base_url}/videos?search={search_query}",
    ]

    for search_url in search_urls:
        try:
            print(f"Trying: {search_url}")
            response = requests.get(search_url, headers=headers, timeout=10)

            if response.status_code == 200:
                print("✓ Page loaded")
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract video links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(kw in href.lower() for kw in ['video', 'watch', '/v/', 'play']):
                        video_urls.add(urljoin(base_url, href))

                for video in soup.find_all('video'):
                    if video.get('src'):
                        video_urls.add(urljoin(base_url, video['src']))

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(ext in href.lower() for ext in ['.mp4', '.webm', '.avi', '.mov']):
                        video_urls.add(urljoin(base_url, href))

                print(f"✓ Found {len(video_urls)} URLs")
                break

        except Exception as e:
            continue

    return list(video_urls)

def save_urls(urls, output_file):
    """Save URLs to file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for url in urls:
            f.write(f"{url}\n")

    print(f"\n✓ Saved to: {output_path.absolute()}")

def main():
    print("="*70)
    print("Infinite Scroll Scraper")
    print("="*70)
    print("For websites that load content as you scroll\n")
    print("Methods:")
    print("  1. Playwright (Recommended - can scroll and load more content)")
    print("  2. Simple (Limited - only initial page load)")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_infinite_scroll.py <method> <url> <query> [scrolls] [output]")
            print()
            print("Methods: playwright, simple")
            print()
            print("Examples:")
            print("  python scrape_infinite_scroll.py playwright 'https://example.com' 'fight' 20")
            print("  python scrape_infinite_scroll.py simple 'https://example.com' 'fight'")
            print()
            print("Scrolls: Number of times to scroll down (more scrolls = more videos)")
            print("         Recommended: 10-50 scrolls depending on how much content you want")
            sys.exit(0)

        method = sys.argv[1].lower()
        base_url = sys.argv[2]
        search_query = sys.argv[3]
        num_scrolls = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        output_file = sys.argv[5] if len(sys.argv) > 5 else "video_urls.txt"

    else:
        # Interactive
        print("Choose method:")
        method = input("  1=Playwright (scroll), 2=Simple (no scroll): ").strip()
        method = 'playwright' if method == '1' else 'simple'

        base_url = input("\nWebsite URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        search_query = input("Search query: ").strip()

        if method == 'playwright':
            num_scrolls = int(input("Number of scrolls (default 10, more=more videos): ").strip() or "10")
        else:
            num_scrolls = 0

        output_file = input("Output file (default 'video_urls.txt'): ").strip() or "video_urls.txt"

    # Scrape
    if method == 'playwright':
        urls = scrape_infinite_scroll_playwright(base_url, search_query, num_scrolls)
    else:
        urls = scrape_infinite_scroll_simple(base_url, search_query)

    if not urls:
        print("\n❌ No URLs found!")
        print("\nTips:")
        print("  - Try more scrolls if using Playwright")
        print("  - Check if the search URL format is correct")
        print("  - Use Playwright method for infinite scroll sites")
        return

    save_urls(urls, output_file)

    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ {len(urls)} URLs extracted and saved")
    print(f"✓ File: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
