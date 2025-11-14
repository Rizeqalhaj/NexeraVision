#!/usr/bin/env python3
"""
Playwright-based URL Scraper (Alternative to Selenium)
Lighter and works better on cloud instances like vast.ai
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin

def scrape_with_playwright(base_url, search_query, max_pages=5):
    """
    Scrape URLs using Playwright (lighter than Selenium)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        print("\nInstall with:")
        print("  pip install playwright")
        print("  playwright install chromium")
        sys.exit(1)

    video_urls = []

    print(f"\n{'='*70}")
    print(f"Playwright Scraper (JavaScript-enabled)")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        # Launch browser
        print("🌐 Starting Chromium browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = context.new_page()

        for page_num in range(1, max_pages + 1):
            # Try multiple search URL patterns
            search_urls = [
                f"{base_url}/search?search={search_query}&page={page_num}",
                f"{base_url}/search?q={search_query}&page={page_num}",
                f"{base_url}/browse?search={search_query}&page={page_num}",
                f"{base_url}/videos?search={search_query}&page={page_num}",
            ]

            page_urls = []
            success = False

            for search_url in search_urls:
                try:
                    print(f"\n📄 Page {page_num}: {search_url}")
                    page.goto(search_url, timeout=15000)
                    page.wait_for_timeout(3000)  # Wait for JS to render

                    # Check if page loaded
                    if "404" in page.title().lower():
                        continue

                    success = True
                    print(f"   ✓ Page loaded, extracting URLs...")

                    # Extract all links
                    links = page.query_selector_all('a[href]')
                    for link in links:
                        try:
                            href = link.get_attribute('href')
                            if href and any(kw in href.lower() for kw in ['video', 'watch', '/v/', '/v?']):
                                page_urls.append(urljoin(base_url, href))
                        except:
                            pass

                    # Extract video sources
                    videos = page.query_selector_all('video')
                    for video in videos:
                        try:
                            src = video.get_attribute('src')
                            if src:
                                page_urls.append(urljoin(base_url, src))
                        except:
                            pass

                    # Extract data-src attributes
                    data_elements = page.query_selector_all('[data-src]')
                    for elem in data_elements:
                        try:
                            data_src = elem.get_attribute('data-src')
                            if data_src and any(ext in data_src.lower() for ext in ['.mp4', '.webm', 'video']):
                                page_urls.append(urljoin(base_url, data_src))
                        except:
                            pass

                    print(f"   ✓ Found {len(page_urls)} URLs")
                    break

                except Exception as e:
                    print(f"   ✗ Error: {e}")
                    continue

            if success:
                video_urls.extend(page_urls)

            time.sleep(2)

        browser.close()
        print("\n🔒 Browser closed")

    # Remove duplicates
    video_urls = list(set(video_urls))
    print(f"\n{'='*70}")
    print(f"✓ Total unique URLs: {len(video_urls)}")
    print(f"{'='*70}")

    return video_urls

def scrape_simple_requests(base_url, search_query, max_pages=5):
    """
    Simple scraper using requests (no JavaScript support)
    Use this if Playwright doesn't work
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("❌ requests or beautifulsoup4 not installed!")
        print("Install with: pip install requests beautifulsoup4")
        sys.exit(1)

    video_urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"\n{'='*70}")
    print(f"Simple HTTP Scraper (No JavaScript)")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*70}\n")

    for page_num in range(1, max_pages + 1):
        search_urls = [
            f"{base_url}/search?search={search_query}&page={page_num}",
            f"{base_url}/search?q={search_query}&page={page_num}",
            f"{base_url}/browse?search={search_query}&page={page_num}",
        ]

        for search_url in search_urls:
            try:
                print(f"📄 Page {page_num}: {search_url}")
                response = requests.get(search_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    page_urls = []

                    # Find video links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if any(kw in href.lower() for kw in ['video', 'watch', '/v/']):
                            page_urls.append(urljoin(base_url, href))

                    # Find video tags
                    for video in soup.find_all('video'):
                        if video.get('src'):
                            page_urls.append(urljoin(base_url, video['src']))

                    # Find direct video files
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if any(ext in href.lower() for ext in ['.mp4', '.webm', '.avi', '.mov']):
                            page_urls.append(urljoin(base_url, href))

                    print(f"   ✓ Found {len(page_urls)} URLs")
                    video_urls.extend(page_urls)
                    break

            except Exception as e:
                continue

        time.sleep(2)

    video_urls = list(set(video_urls))
    print(f"\n✓ Total: {len(video_urls)} URLs")
    return video_urls

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
    print("Multi-Method Web Scraper")
    print("="*70)
    print("Choose scraping method:\n")
    print("1. Playwright (JavaScript support, lighter than Selenium)")
    print("2. Simple Requests (No JavaScript, works everywhere)")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_urls_playwright.py <method> <url> <query> [pages] [output]")
            print()
            print("Methods: playwright, simple")
            print()
            print("Examples:")
            print("  python scrape_urls_playwright.py playwright 'https://example.com' 'fight' 5")
            print("  python scrape_urls_playwright.py simple 'https://example.com' 'fight' 10")
            sys.exit(0)

        method = sys.argv[1].lower()
        base_url = sys.argv[2]
        search_query = sys.argv[3]
        max_pages = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        output_file = sys.argv[5] if len(sys.argv) > 5 else "video_urls.txt"

    else:
        # Interactive
        method = input("Choose method (1=playwright, 2=simple): ").strip()
        method = 'playwright' if method == '1' else 'simple'

        base_url = input("\nWebsite URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        search_query = input("Search query: ").strip()
        max_pages = int(input("Max pages (default 5): ").strip() or "5")
        output_file = input("Output file (default 'video_urls.txt'): ").strip() or "video_urls.txt"

    # Scrape based on method
    if method == 'playwright':
        urls = scrape_with_playwright(base_url, search_query, max_pages)
    else:
        urls = scrape_simple_requests(base_url, search_query, max_pages)

    if not urls:
        print("\n❌ No URLs found!")
        return

    save_urls(urls, output_file)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"✓ {len(urls)} URLs extracted")
    print("="*70)

if __name__ == "__main__":
    main()
