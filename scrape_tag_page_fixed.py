#!/usr/bin/env python3
"""
Tag Page Scraper - Fixed for slug-based video URLs
For sites where videos are like: example.com/video-title-slug/
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

def scrape_tag_page(base_url, tag_path, max_clicks=150):
    """
    Scrape tag page with slug-based video URLs
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    video_urls = set()

    print(f"\n{'='*70}")
    print(f"Tag Page Scraper (Slug-based URLs)")
    print(f"{'='*70}")
    print(f"URL: {base_url}{tag_path}")
    print(f"Max Load More clicks: {max_clicks}")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        print("🌐 Starting browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        page.set_default_timeout(45000)

        full_url = f"{base_url}{tag_path}"
        print(f"Loading: {full_url}")

        try:
            page.goto(full_url, timeout=45000, wait_until='networkidle')
            page.wait_for_timeout(3000)
            print(f"✓ Loaded: {page.title()}\n")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            browser.close()
            return []

        consecutive_no_new = 0

        for click_num in range(1, max_clicks + 1):
            before_count = len(video_urls)

            # Get all links
            all_links = page.query_selector_all('a[href]')

            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href:
                        full_link = urljoin(base_url, href)
                        parsed = urlparse(full_link)

                        # Filter for video post URLs:
                        # - Must be from same domain
                        # - Path should have content (not just /)
                        # - Exclude: tag/, category/, page/, author/, etc.
                        # - Should look like a post slug

                        if (parsed.netloc in base_url and
                            parsed.path and
                            parsed.path != '/' and
                            not any(skip in parsed.path.lower() for skip in [
                                '/tag/', '/category/', '/page/', '/author/',
                                '/search/', '/wp-', '/feed/', '/login', '/register',
                                '.css', '.js', '.jpg', '.png', '.gif'
                            ]) and
                            parsed.path.count('/') >= 2):  # At least /something/

                            video_urls.add(full_link)
                except:
                    pass

            after_count = len(video_urls)
            new_found = after_count - before_count

            if new_found == 0:
                consecutive_no_new += 1
            else:
                consecutive_no_new = 0

            print(f"🔄 Click {click_num}/{max_clicks} | +{new_found} new | Total: {len(video_urls)} | No new: {consecutive_no_new}/3")

            # Stop if no new videos for 3 consecutive clicks
            if consecutive_no_new >= 3:
                print(f"\n✓ Stopping - no new videos after 3 Load More clicks")
                break

            # Scroll down
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

            # Try to find and click Load More
            load_more_clicked = False
            load_more_selectors = [
                'a:has-text("Load More")',
                'button:has-text("Load More")',
                'a:has-text("Show More")',
                'button:has-text("Show More")',
                '.load-more',
                '#load-more',
            ]

            for selector in load_more_selectors:
                try:
                    button = page.query_selector(selector)
                    if button and button.is_visible():
                        button.click()
                        load_more_clicked = True
                        page.wait_for_timeout(5000)
                        break
                except:
                    continue

            if not load_more_clicked:
                # Try scrolling as fallback
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(5000)

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
        for url in sorted(urls):
            f.write(f"{url}\n")

    print(f"\n✓ Saved to: {output_path.absolute()}")

def main():
    print("="*70)
    print("Tag Page Scraper (Slug-based URLs)")
    print("="*70)
    print("For video posts like: example.com/video-title-slug/\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_tag_page_fixed.py <base_url> <tag_path> [max_clicks] [output]")
            print()
            print("Examples:")
            print("  python scrape_tag_page_fixed.py 'https://example.com' '/tag/street-fights/' 150")
            print("  python scrape_tag_page_fixed.py 'https://example.com' '/tag/violence/' 100 violence.txt")
            sys.exit(0)

        base_url = sys.argv[1]
        tag_path = sys.argv[2]
        max_clicks = int(sys.argv[3]) if len(sys.argv) > 3 else 150
        output_file = sys.argv[4] if len(sys.argv) > 4 else "tag_videos.txt"

    else:
        base_url = input("Base URL (e.g., 'https://example.com'): ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        tag_path = input("Tag path (e.g., '/tag/street-fights/'): ").strip()
        if not tag_path.startswith('/'):
            tag_path = '/' + tag_path

        max_clicks = int(input("Max Load More clicks (default 150): ").strip() or "150")
        output_file = input("Output file (default 'tag_videos.txt'): ").strip() or "tag_videos.txt"

    print(f"\nConfiguration:")
    print(f"  URL: {base_url}{tag_path}")
    print(f"  Max clicks: {max_clicks}")

    confirm = input("\nStart scraping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Scrape
    urls = scrape_tag_page(base_url, tag_path, max_clicks)

    if not urls:
        print("\n❌ No URLs found!")
        return

    # Save
    save_urls(urls, output_file)

    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ {len(urls)} video URLs collected")
    print(f"✓ Saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
