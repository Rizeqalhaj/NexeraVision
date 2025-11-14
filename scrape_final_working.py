#!/usr/bin/env python3
"""
Final Working Scraper
Uses correct URL pattern: /w/[id]
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin, quote
import re

def scrape_videos(base_url, search_query, num_scrolls=100):
    """
    Scrape using correct /w/ pattern
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    video_urls = set()

    print(f"\n{'='*70}")
    print(f"Working Scraper - Pattern: /w/[id]")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"Max scrolls: {num_scrolls}")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        print("🌐 Starting browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        # Build search URL
        encoded_query = quote(search_query)
        search_url = f"{base_url}/search?search={encoded_query}&searchTarget=local"

        print(f"Loading: {search_url}")

        try:
            page.goto(search_url, timeout=30000, wait_until='networkidle')
            page.wait_for_timeout(5000)
            print(f"✓ Loaded: {page.title()}\n")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            browser.close()
            return []

        consecutive_empty = 0

        for scroll_num in range(1, num_scrolls + 1):
            before_count = len(video_urls)

            # Get ALL links
            all_links = page.query_selector_all('a[href]')

            # Filter for /w/ pattern (video pages)
            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        # Check if it matches /w/[id] pattern
                        if re.search(r'/w/[a-zA-Z0-9]+', full_url):
                            video_urls.add(full_url)
                except:
                    pass

            after_count = len(video_urls)
            new_found = after_count - before_count

            if new_found == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            print(f"🔄 Scroll {scroll_num}/{num_scrolls} | +{new_found} new | Total: {len(video_urls)} | Empty: {consecutive_empty}/5")

            if consecutive_empty >= 5:
                print(f"\n✓ Stopping - no new content after 5 scrolls")
                break

            # Aggressive scrolling
            for step in range(5):
                page.evaluate(f"window.scrollBy(0, {page.viewport_size['height'] / 5})")
                page.wait_for_timeout(400)

            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(6000)  # 6 second wait

            # Scroll up and back down
            page.evaluate("window.scrollBy(0, -300)")
            page.wait_for_timeout(1000)
            page.evaluate("window.scrollBy(0, 300)")
            page.wait_for_timeout(2000)

        browser.close()
        print("\n🔒 Browser closed")

    video_urls = list(video_urls)
    print(f"\n{'='*70}")
    print(f"✓ Total unique video URLs: {len(video_urls)}")
    print(f"{'='*70}")

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
    print("Final Working Scraper")
    print("="*70)
    print("Uses correct pattern: /w/[id]")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_final_working.py <url> <query> [max_scrolls] [output]")
            print()
            print("Examples:")
            print("  python scrape_final_working.py 'https://example.com' 'fighting' 100")
            print("  python scrape_final_working.py 'https://example.com' 'cctv' 150 videos.txt")
            sys.exit(0)

        base_url = sys.argv[1]
        search_query = sys.argv[2]
        num_scrolls = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        output_file = sys.argv[4] if len(sys.argv) > 4 else "video_urls.txt"

    else:
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        search_query = input("Search query: ").strip()
        num_scrolls = int(input("Max scrolls (default 100): ").strip() or "100")
        output_file = input("Output file (default 'video_urls.txt'): ").strip() or "video_urls.txt"

    print(f"\n⏱️  Estimated time: {num_scrolls * 8 / 60:.1f} minutes")
    time.sleep(2)

    urls = scrape_videos(base_url, search_query, num_scrolls)

    if not urls:
        print("\n❌ No URLs found!")
        return

    save_urls(urls, output_file)

    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ {len(urls)} video URLs collected")
    print(f"✓ Saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
