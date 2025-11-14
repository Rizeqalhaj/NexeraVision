#!/usr/bin/env python3
"""
Headless Page Debugger
Works on vast.ai without display
"""

import sys
from urllib.parse import urljoin
import json

def debug_page_structure(base_url, search_query):
    """
    Debug page structure in headless mode
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Headless Page Debugger")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"{'='*70}\n")

    debug_info = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
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
            f"{base_url}/videos?search={search_query}",
        ]

        loaded_url = None
        for search_url in search_urls:
            try:
                print(f"Trying: {search_url}")
                page.goto(search_url, timeout=20000, wait_until='networkidle')
                page.wait_for_timeout(3000)
                if "404" not in page.title().lower():
                    loaded_url = search_url
                    print(f"✓ Loaded: {page.title()}\n")
                    break
            except Exception as e:
                print(f"  Failed: {e}")
                continue

        if not loaded_url:
            print("❌ Could not load any search URL")
            browser.close()
            return

        debug_info['url'] = loaded_url
        debug_info['title'] = page.title()

        # 1. Analyze all links
        print("="*70)
        print("1. ANALYZING LINKS")
        print("="*70)

        all_links = page.query_selector_all('a[href]')
        print(f"\nTotal <a> links: {len(all_links)}")

        links_data = []
        for i, link in enumerate(all_links[:30], 1):
            href = link.get_attribute('href')
            text = link.inner_text().strip()[:50]
            classes = link.get_attribute('class') or ''
            links_data.append({
                'href': href,
                'text': text,
                'classes': classes
            })
            print(f"  {i}. {href}")
            if text:
                print(f"      Text: {text}")
            if classes:
                print(f"      Classes: {classes}")

        debug_info['sample_links'] = links_data

        # 2. Check for video-related patterns
        print(f"\n{'='*70}")
        print("2. VIDEO PATTERNS")
        print("="*70)

        patterns = {
            'links_with_video': 'a[href*="video"]',
            'links_with_watch': 'a[href*="watch"]',
            'links_with_v': 'a[href*="/v/"]',
            'video_tags': 'video',
            'divs_with_video_class': 'div[class*="video"]',
            'divs_with_item_class': 'div[class*="item"]',
            'divs_with_card_class': 'div[class*="card"]',
            'articles': 'article',
        }

        pattern_results = {}
        for name, selector in patterns.items():
            try:
                elements = page.query_selector_all(selector)
                count = len(elements)
                pattern_results[name] = count
                print(f"\n{name}: {count}")

                if count > 0 and count < 50:
                    # Show first few
                    for i, elem in enumerate(elements[:3], 1):
                        try:
                            if name.startswith('links'):
                                href = elem.get_attribute('href')
                                print(f"  {i}. {href}")
                            else:
                                html = elem.inner_html()[:200]
                                print(f"  {i}. {html}...")
                        except:
                            pass
            except Exception as e:
                pattern_results[name] = 0

        debug_info['patterns'] = pattern_results

        # 3. Test scrolling
        print(f"\n{'='*70}")
        print("3. SCROLL TEST")
        print("="*70)

        # Get page height before scroll
        height_before = page.evaluate("document.body.scrollHeight")
        links_before = len(page.query_selector_all('a'))

        print(f"\nBefore scroll:")
        print(f"  Page height: {height_before}px")
        print(f"  Total links: {links_before}")

        # Scroll to bottom
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(5000)

        height_after = page.evaluate("document.body.scrollHeight")
        links_after = len(page.query_selector_all('a'))

        print(f"\nAfter scroll:")
        print(f"  Page height: {height_after}px")
        print(f"  Total links: {links_after}")
        print(f"\nChange:")
        print(f"  Height: +{height_after - height_before}px")
        print(f"  Links: +{links_after - links_before} new")

        debug_info['scroll_test'] = {
            'height_before': height_before,
            'height_after': height_after,
            'links_before': links_before,
            'links_after': links_after,
            'new_content_loaded': links_after > links_before
        }

        # 4. Look for load more buttons
        print(f"\n{'='*70}")
        print("4. LOAD MORE BUTTONS")
        print("="*70)

        button_selectors = [
            'button',
            'a.button',
            'a.btn',
            '[role="button"]',
        ]

        found_buttons = []
        for selector in button_selectors:
            buttons = page.query_selector_all(selector)
            for btn in buttons:
                text = btn.inner_text().lower()
                if any(word in text for word in ['load', 'more', 'show', 'next']):
                    found_buttons.append({
                        'text': btn.inner_text(),
                        'selector': selector,
                        'visible': btn.is_visible()
                    })

        if found_buttons:
            print(f"\n✓ Found {len(found_buttons)} potential buttons:")
            for btn in found_buttons:
                print(f"  - '{btn['text']}' (visible: {btn['visible']})")
        else:
            print("\n✗ No load more buttons found")

        debug_info['buttons'] = found_buttons

        # 5. Check for pagination
        print(f"\n{'='*70}")
        print("5. PAGINATION")
        print("="*70)

        pagination_selectors = [
            '.pagination',
            '.pager',
            'nav[aria-label*="pagination"]',
            'a[href*="page="]',
        ]

        has_pagination = False
        for selector in pagination_selectors:
            elements = page.query_selector_all(selector)
            if elements:
                print(f"\n✓ Found pagination: {selector} ({len(elements)} elements)")
                has_pagination = True

        if not has_pagination:
            print("\n✗ No pagination found")

        debug_info['has_pagination'] = has_pagination

        # 6. Save HTML and screenshot
        html_file = "debug_page.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(page.content())

        screenshot_file = "debug_page.png"
        page.screenshot(path=screenshot_file, full_page=True)

        print(f"\n✓ Saved HTML to: {html_file}")
        print(f"✓ Saved screenshot to: {screenshot_file}")

        # Save debug info as JSON
        json_file = "debug_info.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2)

        print(f"✓ Saved debug info to: {json_file}")

        browser.close()

    # Analysis summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print("="*70)

    if debug_info['scroll_test']['new_content_loaded']:
        print("\n✓ INFINITE SCROLL DETECTED")
        print("  - New content loads when scrolling")
        print("  - Current scraper should work but might need longer wait times")
    else:
        print("\n✗ NO INFINITE SCROLL")
        print("  - All content loads at once")
        print("  - Only 9 videos might be all that exist for this search")

    # Suggest best selector
    print(f"\n{'='*70}")
    print("RECOMMENDED APPROACH")
    print("="*70)

    if pattern_results.get('links_with_video', 0) > 0:
        print(f"\n✓ Use selector: a[href*=\"video\"]")
        print(f"  Found {pattern_results['links_with_video']} video links")
    elif pattern_results.get('links_with_watch', 0) > 0:
        print(f"\n✓ Use selector: a[href*=\"watch\"]")
        print(f"  Found {pattern_results['links_with_watch']} watch links")
    else:
        print("\n⚠️  Custom approach needed - check debug_page.html manually")

    print("\n" + "="*70)

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_page_headless.py <url> <search_query>")
        print("\nExample:")
        print("  python debug_page_headless.py 'https://example.com' 'fight'")
        sys.exit(1)

    base_url = sys.argv[1]
    search_query = sys.argv[2]

    debug_page_structure(base_url, search_query)

if __name__ == "__main__":
    main()
