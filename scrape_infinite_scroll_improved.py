#!/usr/bin/env python3
"""
Improved Infinite Scroll Scraper
Handles slow loading, "Load More" buttons, and early termination
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin

def scrape_infinite_scroll_smart(base_url, search_query, num_scrolls=60, stop_after_empty=3):
    """
    Smart infinite scroll with:
    - Longer wait times for slow loading
    - Auto-stop if no new content
    - "Load More" button detection
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        print("Install: pip install playwright && playwright install chromium")
        sys.exit(1)

    video_urls = set()
    empty_scroll_count = 0

    print(f"\n{'='*70}")
    print(f"Smart Infinite Scroll Scraper")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"Max scrolls: {num_scrolls}")
    print(f"Auto-stop after {stop_after_empty} empty scrolls")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        print("🌐 Starting browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        # Try different search URL formats
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
                page.goto(url_format, timeout=20000, wait_until='networkidle')
                page.wait_for_timeout(3000)

                if "404" not in page.title().lower():
                    search_url = url_format
                    page_loaded = True
                    print(f"✓ Page loaded: {page.title()}\n")
                    break
            except:
                continue

        if not page_loaded:
            print("❌ Could not load search page")
            browser.close()
            return []

        # Scroll and collect
        for scroll_num in range(1, num_scrolls + 1):
            current_count = len(video_urls)

            # Extract URLs
            # Method 1: Video page links
            links = page.query_selector_all('a[href]')
            for link in links:
                try:
                    href = link.get_attribute('href')
                    if href and any(kw in href.lower() for kw in ['video', 'watch', '/v/', '/v?', 'play']):
                        full_url = urljoin(base_url, href)
                        # Only add if looks like actual video page
                        if not any(skip in full_url.lower() for skip in ['login', 'register', 'account', 'profile']):
                            video_urls.add(full_url)
                except:
                    pass

            # Method 2: Video elements
            videos = page.query_selector_all('video, source')
            for video in videos:
                try:
                    for attr in ['src', 'data-src']:
                        src = video.get_attribute(attr)
                        if src:
                            video_urls.add(urljoin(base_url, src))
                except:
                    pass

            # Method 3: All data attributes that might contain video URLs
            for attr in ['data-src', 'data-video', 'data-url', 'data-video-src']:
                elements = page.query_selector_all(f'[{attr}]')
                for elem in elements:
                    try:
                        url = elem.get_attribute(attr)
                        if url and ('video' in url.lower() or any(ext in url.lower() for ext in ['.mp4', '.webm'])):
                            video_urls.add(urljoin(base_url, url))
                    except:
                        pass

            # Method 4: Direct video file links
            for link in page.query_selector_all('a[href]'):
                try:
                    href = link.get_attribute('href')
                    if href and any(ext in href.lower() for ext in ['.mp4', '.webm', '.avi', '.mov', '.flv', '.mkv']):
                        video_urls.add(urljoin(base_url, href))
                except:
                    pass

            new_found = len(video_urls) - current_count

            # Update empty scroll counter
            if new_found == 0:
                empty_scroll_count += 1
            else:
                empty_scroll_count = 0

            print(f"🔄 Scroll {scroll_num}/{num_scrolls} | Found {new_found} new | Total: {len(video_urls)} | Empty: {empty_scroll_count}/{stop_after_empty}")

            # Stop if too many empty scrolls
            if empty_scroll_count >= stop_after_empty:
                print(f"\n✓ Stopping early - no new content after {stop_after_empty} scrolls")
                break

            # Try to click "Load More" button if exists
            try:
                load_more_selectors = [
                    'button:has-text("Load More")',
                    'button:has-text("Show More")',
                    'a:has-text("Load More")',
                    'a:has-text("Show More")',
                    '.load-more',
                    '#load-more',
                ]
                for selector in load_more_selectors:
                    try:
                        btn = page.query_selector(selector)
                        if btn and btn.is_visible():
                            print("   🖱️  Clicking 'Load More' button...")
                            btn.click()
                            page.wait_for_timeout(2000)
                            break
                    except:
                        pass
            except:
                pass

            # Scroll down in multiple steps for smoother loading
            for _ in range(3):
                page.evaluate("window.scrollBy(0, window.innerHeight / 3)")
                page.wait_for_timeout(500)

            # Final scroll to bottom
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Wait for content to load (longer wait)
            page.wait_for_timeout(4000)

            # Scroll up slightly and back down (triggers lazy loading)
            page.evaluate("window.scrollBy(0, -200)")
            page.wait_for_timeout(500)
            page.evaluate("window.scrollBy(0, 200)")
            page.wait_for_timeout(1000)

        browser.close()
        print("\n🔒 Browser closed")

    video_urls = list(video_urls)
    print(f"\n{'='*70}")
    print(f"✓ Total unique URLs: {len(video_urls)}")
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
    print(f"✓ Total URLs: {len(urls)}")

def main():
    print("="*70)
    print("Smart Infinite Scroll Scraper")
    print("="*70)
    print("Features:")
    print("  ✓ Auto-stops when no more content")
    print("  ✓ Clicks 'Load More' buttons")
    print("  ✓ Longer wait times for slow sites")
    print("  ✓ Multiple scroll techniques")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_infinite_scroll_improved.py <url> <query> [max_scrolls] [stop_after] [output]")
            print()
            print("Arguments:")
            print("  url         : Website URL")
            print("  query       : Search query")
            print("  max_scrolls : Maximum scrolls (default: 60)")
            print("  stop_after  : Stop after N empty scrolls (default: 3)")
            print("  output      : Output filename (default: video_urls.txt)")
            print()
            print("Examples:")
            print("  python scrape_infinite_scroll_improved.py 'https://example.com' 'fight' 100")
            print("  python scrape_infinite_scroll_improved.py 'https://example.com' 'cctv' 50 5 urls.txt")
            sys.exit(0)

        base_url = sys.argv[1]
        search_query = sys.argv[2]
        num_scrolls = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        stop_after = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        output_file = sys.argv[5] if len(sys.argv) > 5 else "video_urls.txt"

    else:
        # Interactive
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        search_query = input("Search query: ").strip()
        num_scrolls = int(input("Max scrolls (default 60): ").strip() or "60")
        stop_after = int(input("Stop after empty scrolls (default 3): ").strip() or "3")
        output_file = input("Output file (default 'video_urls.txt'): ").strip() or "video_urls.txt"

    # Scrape
    urls = scrape_infinite_scroll_smart(base_url, search_query, num_scrolls, stop_after)

    if not urls:
        print("\n❌ No URLs found!")
        print("\nTroubleshooting:")
        print("  1. Check if search query returns results on the website")
        print("  2. Try different search query")
        print("  3. Verify the website URL is correct")
        return

    save_urls(urls, output_file)

    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ {len(urls)} video URLs extracted")
    print(f"✓ Saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Review the URLs in the text file")
    print("  2. Use yt-dlp or other downloader to get the videos")
    print("="*70)

if __name__ == "__main__":
    main()
