#!/usr/bin/env python3
"""
Comprehensive Fight/Violence Video Scraper
Searches 50+ fight-related keywords to build training dataset
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin, quote
import re

# Comprehensive fight/violence keywords for training data
FIGHT_KEYWORDS = [
    # CCTV specific
    'cctv fight',
    'security camera fight',
    'surveillance fight',
    'caught on camera fight',
    'security footage fight',
    'camera caught fight',

    # General fights
    'fight',
    'street fight',
    'fist fight',
    'brawl',
    'physical fight',
    'real fight',
    'fight caught on camera',
    'public fight',

    # Location-specific fights
    'bar fight',
    'parking lot fight',
    'gas station fight',
    'store fight',
    'convenience store fight',
    'restaurant fight',
    'subway fight',
    'train fight',
    'bus fight',
    'mall fight',
    'school fight',
    'street brawl',

    # Violence keywords
    'violence',
    'assault',
    'attack',
    'physical assault',
    'beating',
    'altercation',
    'confrontation',

    # Road incidents
    'road rage',
    'road rage fight',
    'traffic fight',

    # Multiple people
    'group fight',
    'mass brawl',
    'gang fight',
    'mob fight',
    'riot',

    # Sports/organized
    'street fighting',
    'backyard fight',
    'underground fight',

    # Specific scenarios
    'knockout',
    'sucker punch',
    'cheap shot',
    'jumped',

    # CCTV violence
    'cctv violence',
    'cctv assault',
    'cctv attack',
    'security camera violence',
    'surveillance violence',
    'surveillance assault',
]

def scrape_videos(base_url, search_query, num_scrolls=100):
    """
    Scrape using /w/ pattern
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        sys.exit(1)

    video_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        # Build search URL
        encoded_query = quote(search_query)
        search_url = f"{base_url}/search?search={encoded_query}&searchTarget=local"

        try:
            page.goto(search_url, timeout=30000, wait_until='networkidle')
            page.wait_for_timeout(5000)
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            browser.close()
            return []

        consecutive_empty = 0

        for scroll_num in range(1, num_scrolls + 1):
            before_count = len(video_urls)

            # Get ALL links and filter for /w/ pattern
            all_links = page.query_selector_all('a[href]')
            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href and re.search(r'/w/[a-zA-Z0-9]+', href):
                        video_urls.add(urljoin(base_url, href))
                except:
                    pass

            after_count = len(video_urls)
            new_found = after_count - before_count

            if new_found == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            # Show progress every 10 scrolls to reduce clutter
            if scroll_num % 10 == 0 or new_found > 0:
                print(f"  Scroll {scroll_num}/{num_scrolls} | +{new_found} | Total: {len(video_urls)}")

            if consecutive_empty >= 5:
                break

            # Aggressive scrolling
            for step in range(5):
                page.evaluate(f"window.scrollBy(0, {page.viewport_size['height'] / 5})")
                page.wait_for_timeout(400)

            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(6000)

            page.evaluate("window.scrollBy(0, -300)")
            page.wait_for_timeout(1000)
            page.evaluate("window.scrollBy(0, 300)")
            page.wait_for_timeout(2000)

        browser.close()

    return list(video_urls)

def scrape_all_keywords(base_url, keywords, scrolls_per_query=100):
    """
    Scrape multiple keywords and combine results
    """
    all_urls = set()

    print(f"\n{'='*70}")
    print(f"Multi-Keyword Violence Detection Dataset Scraper")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Keywords: {len(keywords)}")
    print(f"Scrolls per keyword: {scrolls_per_query}")
    print(f"Estimated time: {len(keywords) * scrolls_per_query * 8 / 3600:.1f} hours")
    print(f"{'='*70}\n")

    for i, keyword in enumerate(keywords, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(keywords)}] Keyword: '{keyword}'")
        print(f"{'='*70}")

        urls = scrape_videos(base_url, keyword, scrolls_per_query)

        before = len(all_urls)
        all_urls.update(urls)
        after = len(all_urls)
        new_unique = after - before

        print(f"\n✓ Query '{keyword}':")
        print(f"  Found: {len(urls)} URLs")
        print(f"  New unique: {new_unique}")
        print(f"  Total unique: {len(all_urls)}")

        # Save checkpoint every 10 queries
        if i % 10 == 0:
            checkpoint_file = f"checkpoint_{i}_keywords.txt"
            with open(checkpoint_file, 'w') as f:
                for url in sorted(all_urls):
                    f.write(f"{url}\n")
            print(f"  💾 Checkpoint saved: {checkpoint_file}")

    return list(all_urls)

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
    print("Comprehensive Fight/Violence Dataset Scraper")
    print("="*70)
    print("For training violence detection models")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_all_fight_keywords.py <url> [scrolls_per_query] [output]")
            print()
            print("Examples:")
            print("  python scrape_all_fight_keywords.py 'https://example.com' 100")
            print("  python scrape_all_fight_keywords.py 'https://example.com' 50 dataset_urls.txt")
            print()
            print(f"Default keywords: {len(FIGHT_KEYWORDS)} fight/violence terms")
            print("\nKeyword categories:")
            print("  - CCTV fights (6 keywords)")
            print("  - General fights (8 keywords)")
            print("  - Location-specific (12 keywords)")
            print("  - Violence keywords (7 keywords)")
            print("  - Road incidents (3 keywords)")
            print("  - Multiple people (5 keywords)")
            print("  - Specific scenarios (4 keywords)")
            print("  - CCTV violence (6 keywords)")
            sys.exit(0)

        base_url = sys.argv[1]
        scrolls_per_query = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        output_file = sys.argv[3] if len(sys.argv) > 3 else "all_fight_videos.txt"

    else:
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        print(f"\nDefault: {len(FIGHT_KEYWORDS)} fight/violence keywords")
        print("Use defaults or custom?")
        print("  1. Use all default keywords (recommended)")
        print("  2. Use fewer keywords (faster test)")
        choice = input("Choose (1 or 2): ").strip()

        if choice == '2':
            num_keywords = int(input("How many keywords to use (1-51): ").strip())
            keywords = FIGHT_KEYWORDS[:num_keywords]
        else:
            keywords = FIGHT_KEYWORDS

        scrolls_per_query = int(input(f"\nScrolls per query (default 100): ").strip() or "100")
        output_file = input("Output file (default 'all_fight_videos.txt'): ").strip() or "all_fight_videos.txt"

    if 'keywords' not in locals():
        keywords = FIGHT_KEYWORDS

    # Show keyword list
    print(f"\n{'='*70}")
    print(f"Keywords to search ({len(keywords)} total):")
    print(f"{'='*70}")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i:2}. {kw}")
    print(f"{'='*70}")

    confirmation = input("\nStart scraping? (y/n): ").strip().lower()
    if confirmation != 'y':
        print("Cancelled")
        sys.exit(0)

    # Scrape
    start_time = time.time()
    all_urls = scrape_all_keywords(base_url, keywords, scrolls_per_query)

    if not all_urls:
        print("\n❌ No URLs found!")
        return

    # Save
    save_urls(all_urls, output_file)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ Searched: {len(keywords)} keywords")
    print(f"✓ Found: {len(all_urls)} unique video URLs")
    print(f"✓ Time: {elapsed / 3600:.1f} hours")
    print(f"✓ Saved to: {output_file}")
    print("\nDataset potential:")
    print(f"  - If 90% downloadable: ~{int(len(all_urls) * 0.9)} videos")
    print(f"  - Current training data: ~15K videos")
    print(f"  - After adding this: ~{15000 + int(len(all_urls) * 0.9)} videos")
    print(f"  - Expected accuracy boost: +1-2%")
    print("="*70)

if __name__ == "__main__":
    main()
