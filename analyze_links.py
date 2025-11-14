#!/usr/bin/env python3
"""
Analyze actual link patterns to find the correct video URL format
"""

import sys
from urllib.parse import urljoin, quote
from collections import Counter

def analyze_links(base_url, search_query):
    """
    Show ALL links and their patterns so we can identify video URLs
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Link Pattern Analyzer")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        # Load page
        encoded_query = quote(search_query)
        search_url = f"{base_url}/search?search={encoded_query}&searchTarget=local"

        print(f"Loading: {search_url}")
        page.goto(search_url, timeout=30000, wait_until='networkidle')
        page.wait_for_timeout(5000)
        print(f"✓ Loaded\n")

        # Scroll once to get more content
        print("Scrolling to load more content...")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(6000)
        print("✓ Scrolled\n")

        # Get ALL links
        all_links = page.query_selector_all('a[href]')
        print(f"Total links found: {len(all_links)}\n")

        # Analyze patterns
        print("="*70)
        print("ANALYZING ALL LINK PATTERNS")
        print("="*70)

        link_patterns = []
        for link in all_links:
            href = link.get_attribute('href')
            if href:
                full_url = urljoin(base_url, href)
                link_patterns.append(full_url)

        # Remove duplicates
        unique_links = list(set(link_patterns))
        print(f"\nUnique links: {len(unique_links)}\n")

        # Group by pattern
        from urllib.parse import urlparse
        path_patterns = Counter()

        print("Link patterns by path:")
        print("-" * 70)

        for url in unique_links[:50]:  # Show first 50
            parsed = urlparse(url)
            path = parsed.path

            # Categorize
            if '/video' in path.lower():
                category = "VIDEO"
            elif '/watch' in path.lower():
                category = "WATCH"
            elif '/v/' in path.lower():
                category = "V/"
            elif '/search' in path.lower():
                category = "SEARCH"
            elif '/user' in path.lower() or '/profile' in path.lower():
                category = "USER"
            elif '/tag' in path.lower() or '/category' in path.lower():
                category = "TAG/CAT"
            elif path == '/' or path == '':
                category = "HOME"
            else:
                category = "OTHER"

            path_patterns[category] += 1

            print(f"[{category:10}] {url}")

        print("\n" + "="*70)
        print("SUMMARY BY CATEGORY")
        print("="*70)
        for category, count in path_patterns.most_common():
            print(f"{category:15} : {count} links")

        # Save all links to file for manual inspection
        output_file = "all_links.txt"
        with open(output_file, 'w') as f:
            for url in sorted(unique_links):
                f.write(f"{url}\n")

        print(f"\n✓ Saved all links to: {output_file}")

        browser.close()

    print("\n" + "="*70)
    print("WHAT TO DO NEXT")
    print("="*70)
    print("\n1. Look at the categories above")
    print("2. Check 'all_links.txt' file")
    print("3. Find the pattern for actual video pages")
    print("4. Tell me what pattern the video URLs follow")
    print("\nExamples:")
    print("  - If videos are like: example.com/video/12345")
    print("    → Pattern is: /video/")
    print("  - If videos are like: example.com/watch?id=abc")
    print("    → Pattern is: /watch?")
    print("  - If videos are like: example.com/v/abc123")
    print("    → Pattern is: /v/")

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_links.py <url> <search_query>")
        print("\nExample:")
        print("  python analyze_links.py 'https://example.com' 'fighting'")
        sys.exit(1)

    base_url = sys.argv[1]
    search_query = sys.argv[2]

    analyze_links(base_url, search_query)

if __name__ == "__main__":
    main()
