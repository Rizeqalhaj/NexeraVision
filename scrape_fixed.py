#!/usr/bin/env python3
"""
Fixed Infinite Scroll Scraper
Based on debug results - uses longer waits and proper selectors
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin, quote

def scrape_with_long_waits(base_url, search_query, num_scrolls=100):
    """
    Scraper with longer wait times for slow-loading infinite scroll
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    video_urls = set()

    print(f"\n{'='*70}")
    print(f"Fixed Infinite Scroll Scraper")
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

        # Build search URL with proper encoding
        encoded_query = quote(search_query)
        search_url = f"{base_url}/search?search={encoded_query}&searchTarget=local"

        print(f"Loading: {search_url}")

        try:
            page.goto(search_url, timeout=30000, wait_until='networkidle')
            page.wait_for_timeout(5000)  # Initial 5 second wait
            print(f"✓ Loaded: {page.title()}\n")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            browser.close()
            return []

        consecutive_empty = 0

        for scroll_num in range(1, num_scrolls + 1):
            # Get current video links using the selector from debug
            current_links = page.query_selector_all('a[href*="video"]')

            before_count = len(video_urls)

            # Extract URLs
            for link in current_links:
                try:
                    href = link.get_attribute('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        # Filter out non-video pages
                        if not any(skip in full_url.lower() for skip in ['login', 'register', 'account', 'profile', 'search', 'categories', 'tags']):
                            video_urls.add(full_url)
                except:
                    pass

            after_count = len(video_urls)
            new_found = after_count - before_count

            # Track consecutive empty scrolls
            if new_found == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            print(f"🔄 Scroll {scroll_num}/{num_scrolls} | +{new_found} new | Total: {len(video_urls)} | Empty streak: {consecutive_empty}")

            # Stop if 5 consecutive empty scrolls
            if consecutive_empty >= 5:
                print(f"\n✓ Stopping - no new content after 5 scrolls")
                break

            # AGGRESSIVE SCROLLING with multiple techniques

            # 1. Scroll in smooth steps (triggers lazy loading)
            for step in range(5):
                page.evaluate(f"window.scrollBy(0, {page.viewport_size['height'] / 5})")
                page.wait_for_timeout(400)

            # 2. Scroll to absolute bottom
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # 3. LONGER WAIT - this is key!
            page.wait_for_timeout(6000)  # 6 seconds for content to load

            # 4. Scroll up a bit then back down (triggers some lazy loaders)
            page.evaluate("window.scrollBy(0, -300)")
            page.wait_for_timeout(1000)
            page.evaluate("window.scrollBy(0, 300)")
            page.wait_for_timeout(2000)

            # 5. Check if page height increased (means new content loaded)
            new_height = page.evaluate("document.body.scrollHeight")
            if scroll_num > 1:
                # If height didn't increase much, wait longer
                if new_height == page.evaluate("document.body.scrollHeight"):
                    page.wait_for_timeout(3000)  # Extra 3 second wait

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
    print("Fixed Infinite Scroll Scraper")
    print("="*70)
    print("Based on debug analysis - uses:")
    print("  ✓ Correct URL format with searchTarget=local")
    print("  ✓ Proper selector: a[href*=\"video\"]")
    print("  ✓ 6-8 second wait times between scrolls")
    print("  ✓ Aggressive multi-step scrolling")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_fixed.py <url> <query> [max_scrolls] [output]")
            print()
            print("Examples:")
            print("  python scrape_fixed.py 'https://example.com' 'fighting' 100")
            print("  python scrape_fixed.py 'https://example.com' 'cctv fight' 150 videos.txt")
            sys.exit(0)

        base_url = sys.argv[1]
        search_query = sys.argv[2]
        num_scrolls = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        output_file = sys.argv[4] if len(sys.argv) > 4 else "video_urls.txt"

    else:
        # Interactive
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        search_query = input("Search query: ").strip()
        num_scrolls = int(input("Max scrolls (default 100): ").strip() or "100")
        output_file = input("Output file (default 'video_urls.txt'): ").strip() or "video_urls.txt"

    # Scrape
    print(f"\n⏱️  This will take approximately {num_scrolls * 8 / 60:.1f} minutes")
    print("Press Ctrl+C to stop early if needed\n")

    time.sleep(2)

    urls = scrape_with_long_waits(base_url, search_query, num_scrolls)

    if not urls:
        print("\n❌ No URLs found!")
        return

    save_urls(urls, output_file)

    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ {len(urls)} video URLs collected")
    print(f"✓ File: {output_file}")
    print("\nEstimated videos if downloaded:")
    print(f"  - With 90% success rate: ~{int(len(urls) * 0.9)} videos")
    print("="*70)

if __name__ == "__main__":
    main()
