#!/usr/bin/env python3
"""
Non-Violence Video Scraper
Scrapes normal/peaceful videos to balance violence detection dataset
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin, quote
import re

# Non-violence keywords for balanced dataset
NONVIOLENCE_KEYWORDS = [
    # CCTV normal activities
    'cctv normal',
    'security camera normal',
    'surveillance normal activity',
    'cctv people walking',
    'security footage normal',
    'cctv shopping',
    'cctv store normal',

    # Public places - normal
    'people walking street',
    'pedestrians crossing',
    'shopping mall normal',
    'store customers',
    'restaurant dining',
    'cafe normal',
    'park people',
    'subway commuters',
    'train station normal',
    'bus station',
    'airport normal',

    # Social interactions - peaceful
    'people talking',
    'conversation',
    'handshake',
    'greeting',
    'hugging',
    'friends meeting',
    'family gathering',

    # Activities - non-violent
    'people working',
    'office workers',
    'construction workers',
    'delivery',
    'shopping',
    'walking dog',
    'jogging',
    'exercising',
    'playing sports',
    'basketball game',
    'soccer game',
    'baseball game',

    # Crowds - peaceful
    'crowd gathering',
    'concert crowd',
    'festival',
    'parade',
    'celebration',
    'wedding',
    'graduation',
    'protest peaceful',

    # Traffic - normal
    'traffic normal',
    'cars driving',
    'parking lot normal',
    'gas station normal',
    'highway traffic',
    'intersection',

    # Daily life
    'daily routine',
    'morning commute',
    'lunch break',
    'queue line',
    'waiting room',
    'lobby',
    'hallway',
    'elevator',

    # Public transport
    'bus passengers',
    'metro riders',
    'train passengers',
    'taxi ride',

    # Recreation
    'playground',
    'beach',
    'swimming pool',
    'gym workout',
    'yoga class',
    'dance class',

    # Events
    'conference',
    'meeting',
    'presentation',
    'seminar',
    'classroom',
    'library',
]

def scrape_videos(base_url, search_query, num_scrolls=100):
    """
    Scrape using /w/ pattern (adjust if needed)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        return []

    video_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        page.set_default_timeout(45000)

        # Build search URL
        encoded_query = quote(search_query)
        search_url = f"{base_url}/search?search={encoded_query}&searchTarget=local"

        try:
            page.goto(search_url, wait_until='domcontentloaded', timeout=45000)
            page.wait_for_timeout(3000)
        except:
            browser.close()
            return []

        consecutive_empty = 0

        for scroll_num in range(1, num_scrolls + 1):
            before_count = len(video_urls)

            # Get all links and filter for video pattern
            all_links = page.query_selector_all('a[href]')
            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        # Adjust pattern based on your site
                        if re.search(r'/w/[a-zA-Z0-9]+', full_url):
                            video_urls.add(full_url)
                except:
                    pass

            new_found = len(video_urls) - before_count

            if new_found == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            if consecutive_empty >= 5:
                break

            # Scroll
            for step in range(3):
                page.evaluate(f"window.scrollBy(0, {page.viewport_size['height'] / 3})")
                page.wait_for_timeout(500)

            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(5000)

        browser.close()

    return list(video_urls)

def scrape_all_keywords(base_url, keywords, scrolls_per_query=100):
    """
    Scrape all non-violence keywords
    """
    all_urls = set()

    print(f"\n{'='*70}")
    print(f"Non-Violence Video Scraper")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Keywords: {len(keywords)}")
    print(f"Scrolls per keyword: {scrolls_per_query}")
    print(f"{'='*70}\n")

    estimated_time = len(keywords) * scrolls_per_query * 8 / 3600
    print(f"Estimated time: {estimated_time:.1f} hours ({estimated_time*60:.0f} minutes)\n")

    start_time = time.time()

    for i, keyword in enumerate(keywords, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(keywords)}] Keyword: '{keyword}'")
        print(f"{'='*70}")

        urls = scrape_videos(base_url, keyword, scrolls_per_query)

        before = len(all_urls)
        all_urls.update(urls)
        after = len(all_urls)
        new_unique = after - before

        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (len(keywords) - i) * avg_time

        print(f"\n✓ Results:")
        print(f"  Found: {len(urls)} URLs")
        print(f"  New unique: {new_unique}")
        print(f"  Total unique: {len(all_urls)}")
        print(f"\nProgress: {i}/{len(keywords)} | {elapsed/60:.1f}m elapsed | ~{remaining/60:.1f}m remaining")

        # Checkpoint every 10
        if i % 10 == 0:
            checkpoint = f"checkpoint_nonviolence_{i}.txt"
            with open(checkpoint, 'w') as f:
                for url in sorted(all_urls):
                    f.write(f"{url}\n")
            print(f"💾 Checkpoint: {checkpoint}")

        time.sleep(2)

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Keywords: {len(keywords)}")
    print(f"✓ Total URLs: {len(all_urls)}")
    print(f"⏱️  Time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")
    print(f"{'='*70}")

    return list(all_urls)

def save_urls(urls, output_file):
    """Save URLs"""
    with open(output_file, 'w') as f:
        for url in sorted(urls):
            f.write(f"{url}\n")
    print(f"\n✓ Saved to: {output_file}")

def main():
    print("="*70)
    print("Non-Violence Video Scraper")
    print("="*70)
    print("For balanced violence detection dataset\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_nonviolence_keywords.py <url> [scrolls] [output]")
            print()
            print("Examples:")
            print("  python scrape_nonviolence_keywords.py 'https://example.com'")
            print("  python scrape_nonviolence_keywords.py 'https://example.com' 100 nonviolence_urls.txt")
            print()
            print(f"Keywords: {len(NONVIOLENCE_KEYWORDS)} non-violence terms")
            sys.exit(0)

        base_url = sys.argv[1]
        scrolls = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        output = sys.argv[3] if len(sys.argv) > 3 else "nonviolence_urls.txt"
    else:
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url
        scrolls = int(input("Scrolls per keyword (default 100): ").strip() or "100")
        output = input("Output file (default 'nonviolence_urls.txt'): ").strip() or "nonviolence_urls.txt"

    print(f"\nConfiguration:")
    print(f"  {len(NONVIOLENCE_KEYWORDS)} keywords")
    print(f"  {scrolls} scrolls per keyword")

    confirm = input("\nStart scraping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    urls = scrape_all_keywords(base_url, NONVIOLENCE_KEYWORDS, scrolls)

    if urls:
        save_urls(urls, output)
        print(f"\nDataset balance:")
        print(f"  Violence: 14,000 videos")
        print(f"  Non-violence: 8,000 + {len(urls)} = ~{8000 + int(len(urls)*0.9)} videos")
        if (8000 + int(len(urls)*0.9)) >= 14000:
            print(f"  ✓ BALANCED! Ready for training")
        else:
            print(f"  ⚠️  Need {14000 - (8000 + int(len(urls)*0.9))} more non-violence videos")
    else:
        print("\n❌ No URLs found!")

if __name__ == "__main__":
    main()
