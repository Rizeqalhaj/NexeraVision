#!/usr/bin/env python3
"""
Tag Page Scraper with Load More Button
For pages like example.com/tag/street-fights/ with load more functionality
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin
import re

def scrape_tag_page(base_url, tag_path, max_clicks=50):
    """
    Scrape tag page by clicking Load More button

    Args:
        base_url: Base website URL (e.g., 'https://example.com')
        tag_path: Tag path (e.g., '/tag/street-fights/')
        max_clicks: Maximum "Load More" clicks (default: 50)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    video_urls = set()

    print(f"\n{'='*70}")
    print(f"Tag Page Scraper with Load More")
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

        # Build full URL
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
            # Extract video URLs before clicking
            before_count = len(video_urls)

            # Get all links
            all_links = page.query_selector_all('a[href]')

            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href:
                        full_link = urljoin(base_url, href)
                        # Match /w/[id] pattern (adjust if different)
                        if re.search(r'/w/[a-zA-Z0-9]+', full_link):
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

            # Scroll down to make button visible
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

            # Try to find and click "Load More" button
            load_more_clicked = False

            # Common selectors for Load More buttons
            load_more_selectors = [
                'button:has-text("Load More")',
                'a:has-text("Load More")',
                'button:has-text("Show More")',
                'a:has-text("Show More")',
                'button:has-text("Load more")',
                'a:has-text("Load more")',
                '.load-more',
                '#load-more',
                'button.load-more',
                'a.load-more',
                '[data-action="load-more"]',
            ]

            for selector in load_more_selectors:
                try:
                    button = page.query_selector(selector)
                    if button and button.is_visible():
                        print(f"   🖱️  Found Load More button: {selector}")
                        button.click()
                        load_more_clicked = True
                        print(f"   ✓ Clicked! Waiting for content to load...")
                        page.wait_for_timeout(5000)  # Wait 5 seconds for new content
                        break
                except Exception as e:
                    continue

            if not load_more_clicked:
                print(f"   ⚠️  No Load More button found")
                # If no button, try scrolling more to trigger infinite scroll
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(5000)

                # Check if any new content loaded
                new_check_links = page.query_selector_all('a[href]')
                if len(new_check_links) == len(all_links):
                    consecutive_no_new += 1
                    if consecutive_no_new >= 3:
                        print(f"\n✓ Stopping - reached end of content")
                        break

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
    print("Tag Page Scraper with Load More Button")
    print("="*70)
    print("For tag pages with Load More functionality\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_tag_page_loadmore.py <base_url> <tag_path> [max_clicks] [output]")
            print()
            print("Arguments:")
            print("  base_url   : Base website URL (e.g., 'https://example.com')")
            print("  tag_path   : Tag path (e.g., '/tag/street-fights/')")
            print("  max_clicks : Max Load More clicks (default: 50)")
            print("  output     : Output file (default: tag_videos.txt)")
            print()
            print("Examples:")
            print("  python scrape_tag_page_loadmore.py 'https://example.com' '/tag/street-fights/'")
            print("  python scrape_tag_page_loadmore.py 'https://example.com' '/tag/fights/' 100")
            print("  python scrape_tag_page_loadmore.py 'https://example.com' '/tag/violence/' 50 violence.txt")
            sys.exit(0)

        base_url = sys.argv[1]
        tag_path = sys.argv[2]
        max_clicks = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        output_file = sys.argv[4] if len(sys.argv) > 4 else "tag_videos.txt"

    else:
        # Interactive
        base_url = input("Base URL (e.g., 'https://example.com'): ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        tag_path = input("Tag path (e.g., '/tag/street-fights/'): ").strip()
        if not tag_path.startswith('/'):
            tag_path = '/' + tag_path

        max_clicks = int(input("Max Load More clicks (default 50): ").strip() or "50")
        output_file = input("Output file (default 'tag_videos.txt'): ").strip() or "tag_videos.txt"

    print(f"\nConfiguration:")
    print(f"  URL: {base_url}{tag_path}")
    print(f"  Max clicks: {max_clicks}")
    print(f"  Output: {output_file}")

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
