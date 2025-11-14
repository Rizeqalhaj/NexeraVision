#!/usr/bin/env python3
"""
Debug Tag Page - Show all links to find the correct pattern
"""

import sys
from urllib.parse import urljoin

def debug_tag_page(base_url, tag_path):
    """
    Show all links on the tag page to find video URL pattern
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Tag Page Debugger")
    print(f"{'='*70}")
    print(f"URL: {base_url}{tag_path}")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        full_url = f"{base_url}{tag_path}"
        print(f"Loading: {full_url}")

        page.goto(full_url, timeout=45000, wait_until='networkidle')
        page.wait_for_timeout(3000)
        print(f"✓ Loaded: {page.title()}\n")

        # Click Load More once
        print("Clicking Load More button...")
        try:
            button = page.query_selector('a:has-text("Load More")')
            if button:
                button.click()
                page.wait_for_timeout(5000)
                print("✓ Clicked Load More\n")
        except:
            print("⚠️  Could not click Load More\n")

        # Get ALL links
        all_links = page.query_selector_all('a[href]')
        print(f"{'='*70}")
        print(f"Found {len(all_links)} total links")
        print(f"{'='*70}\n")

        # Analyze link patterns
        from collections import Counter
        link_patterns = Counter()

        print("All unique link patterns:")
        print("-" * 70)

        unique_links = set()
        for link in all_links:
            href = link.get_attribute('href')
            if href:
                full_link = urljoin(base_url, href)
                unique_links.add(full_link)

        # Categorize links
        for url in sorted(unique_links)[:100]:  # Show first 100
            # Categorize
            if '/w/' in url:
                category = "VIDEO-W"
            elif '/v/' in url:
                category = "VIDEO-V"
            elif '/video/' in url.lower():
                category = "VIDEO"
            elif '/watch' in url.lower():
                category = "WATCH"
            elif '/tag/' in url:
                category = "TAG"
            elif '/category/' in url:
                category = "CATEGORY"
            elif base_url in url and len(url.split('/')[-1]) > 10:
                category = "POSSIBLE-VIDEO"
            else:
                category = "OTHER"

            link_patterns[category] += 1
            print(f"[{category:15}] {url}")

        print(f"\n{'='*70}")
        print("Summary by Category:")
        print(f"{'='*70}")
        for category, count in link_patterns.most_common():
            print(f"{category:20} : {count} links")

        # Save all links
        output_file = "all_tag_links.txt"
        with open(output_file, 'w') as f:
            for url in sorted(unique_links):
                f.write(f"{url}\n")

        print(f"\n✓ Saved all links to: {output_file}")

        # Take screenshot
        screenshot = "tag_page_screenshot.png"
        page.screenshot(path=screenshot, full_page=True)
        print(f"✓ Screenshot saved to: {screenshot}")

        browser.close()

    print(f"\n{'='*70}")
    print("ANALYSIS:")
    print(f"{'='*70}")
    print("\nLook at the categories above to find the video pattern.")
    print("\nCommon patterns:")
    print("  - /w/[id]       → Videos on some sites")
    print("  - /v/[id]       → Videos on other sites")
    print("  - /video/[id]   → Classic video URLs")
    print("  - /watch?id=    → YouTube-style")
    print("\nCheck 'all_tag_links.txt' for the full list!")

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_tag_page.py <base_url> <tag_path>")
        print("\nExample:")
        print("  python debug_tag_page.py 'https://example.com' '/tag/street-fights/'")
        sys.exit(1)

    base_url = sys.argv[1]
    tag_path = sys.argv[2]

    debug_tag_page(base_url, tag_path)

if __name__ == "__main__":
    main()
