#!/usr/bin/env python3
"""
Live Page Debugger
Shows exactly what's on the page so we can find the right selectors
"""

import sys
from urllib.parse import urljoin

def debug_page_structure(base_url, search_query):
    """
    Debug what's actually on the page
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Live Page Debugger")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # NOT headless - you can see it
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        # Try search URLs
        search_urls = [
            f"{base_url}/search?search={search_query}",
            f"{base_url}/search?q={search_query}",
            f"{base_url}/browse?search={search_query}",
        ]

        for search_url in search_urls:
            try:
                print(f"Loading: {search_url}")
                page.goto(search_url, timeout=20000, wait_until='networkidle')
                page.wait_for_timeout(3000)
                print(f"✓ Loaded: {page.title()}\n")
                break
            except:
                continue

        # Now analyze what's on the page
        print("="*70)
        print("ANALYZING PAGE CONTENT")
        print("="*70)

        # 1. Count all links
        all_links = page.query_selector_all('a')
        print(f"\n1. Total <a> links on page: {len(all_links)}")

        # Show first 20 links
        print("\nFirst 20 links:")
        for i, link in enumerate(all_links[:20], 1):
            href = link.get_attribute('href')
            text = link.inner_text()[:50]
            print(f"   {i}. href={href} | text={text}")

        # 2. Look for video containers
        print(f"\n{'='*70}")
        print("2. Looking for common video container patterns:")
        print(f"{'='*70}")

        patterns = [
            ('div.video', 'div with class containing "video"'),
            ('div.item', 'div with class containing "item"'),
            ('div.card', 'div with class containing "card"'),
            ('div.post', 'div with class containing "post"'),
            ('article', 'article tags'),
            ('.thumb', 'elements with class "thumb"'),
            ('.thumbnail', 'elements with class "thumbnail"'),
        ]

        for selector, description in patterns:
            try:
                elements = page.query_selector_all(selector)
                if elements:
                    print(f"\n✓ Found {len(elements)} {description}")
                    # Show first element's HTML
                    if len(elements) > 0:
                        html = elements[0].inner_html()[:300]
                        print(f"   First element HTML: {html}...")
            except:
                pass

        # 3. Check what happens when scrolling
        print(f"\n{'='*70}")
        print("3. Testing scroll behavior:")
        print(f"{'='*70}")

        initial_html_length = len(page.content())
        initial_link_count = len(page.query_selector_all('a'))

        print(f"Before scroll: {initial_link_count} links, {initial_html_length} bytes HTML")

        # Scroll
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(5000)  # Wait 5 seconds

        after_html_length = len(page.content())
        after_link_count = len(page.query_selector_all('a'))

        print(f"After scroll:  {after_link_count} links, {after_html_length} bytes HTML")
        print(f"New content loaded: {after_link_count - initial_link_count} new links")

        if after_link_count == initial_link_count:
            print("\n⚠️  WARNING: No new content after scroll!")
            print("This means:")
            print("  - All content is already loaded (no infinite scroll)")
            print("  - OR: Site needs button click or different technique")

        # 4. Look for "Load More" buttons
        print(f"\n{'='*70}")
        print("4. Looking for 'Load More' buttons:")
        print(f"{'='*70}")

        button_texts = ['load more', 'show more', 'see more', 'load', 'more']
        found_buttons = []

        for text in button_texts:
            buttons = page.query_selector_all(f'button:has-text("{text}"), a:has-text("{text}")')
            if buttons:
                for btn in buttons:
                    btn_text = btn.inner_text()
                    found_buttons.append(btn_text)

        if found_buttons:
            print(f"✓ Found buttons: {found_buttons}")
        else:
            print("✗ No 'Load More' buttons found")

        # 5. Save HTML for manual inspection
        html_file = "page_debug.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(page.content())
        print(f"\n✓ Saved full HTML to: {html_file}")

        # 6. Take screenshot
        screenshot_file = "page_debug.png"
        page.screenshot(path=screenshot_file, full_page=True)
        print(f"✓ Saved screenshot to: {screenshot_file}")

        print(f"\n{'='*70}")
        print("KEEP BROWSER OPEN TO INSPECT")
        print(f"{'='*70}")
        print("\nThe browser will stay open so you can:")
        print("  1. Scroll manually and see what happens")
        print("  2. Right-click video items → Inspect")
        print("  3. Find the actual HTML structure")
        print("\nPress ENTER when done inspecting...")
        input()

        browser.close()

    print("\n✓ Debug complete!")
    print(f"\nCheck these files:")
    print(f"  - {html_file} (full HTML)")
    print(f"  - {screenshot_file} (screenshot)")

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_page_live.py <url> <search_query>")
        print("\nExample:")
        print("  python debug_page_live.py 'https://example.com' 'fight'")
        sys.exit(1)

    base_url = sys.argv[1]
    search_query = sys.argv[2]

    debug_page_structure(base_url, search_query)

if __name__ == "__main__":
    main()
