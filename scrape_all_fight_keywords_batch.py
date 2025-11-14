#!/usr/bin/env python3
"""
Batch Sequential Multi-Keyword Scraper
Processes keywords in small batches to avoid resource limits
More reliable than parallel for systems with process limits
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin, quote
import re

# Comprehensive fight/violence keywords
FIGHT_KEYWORDS = [
    'cctv fight', 'security camera fight', 'surveillance fight', 'caught on camera fight',
    'security footage fight', 'camera caught fight', 'fight', 'street fight', 'fist fight',
    'brawl', 'physical fight', 'real fight', 'fight caught on camera', 'public fight',
    'bar fight', 'parking lot fight', 'gas station fight', 'store fight',
    'convenience store fight', 'restaurant fight', 'subway fight', 'train fight',
    'bus fight', 'mall fight', 'school fight', 'street brawl', 'violence',
    'assault', 'attack', 'physical assault', 'beating', 'altercation', 'confrontation',
    'road rage', 'road rage fight', 'traffic fight', 'group fight', 'mass brawl',
    'gang fight', 'mob fight', 'riot', 'street fighting', 'backyard fight',
    'underground fight', 'knockout', 'sucker punch', 'cheap shot', 'jumped',
    'cctv violence', 'cctv assault', 'cctv attack', 'security camera violence',
    'surveillance violence', 'surveillance assault',
]

def scrape_single_keyword(base_url, keyword, num_scrolls):
    """
    Scrape a single keyword
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"❌ Playwright not installed!")
        return []

    video_urls = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--disable-dev-shm-usage', '--disable-gpu', '--no-sandbox']
            )
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            page.set_default_timeout(45000)

            # Build and load URL
            encoded_query = quote(keyword)
            search_url = f"{base_url}/search?search={encoded_query}&searchTarget=local"

            page.goto(search_url, wait_until='domcontentloaded', timeout=45000)
            page.wait_for_timeout(3000)

            consecutive_empty = 0

            for scroll_num in range(1, num_scrolls + 1):
                before_count = len(video_urls)

                # Get all links
                all_links = page.query_selector_all('a[href]')
                for link in all_links:
                    try:
                        href = link.get_attribute('href')
                        if href and re.search(r'/w/[a-zA-Z0-9]+', href):
                            video_urls.add(urljoin(base_url, href))
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

    except Exception as e:
        print(f"    ✗ Error: {str(e)[:50]}")
        return []

    return list(video_urls)

def scrape_all_keywords(base_url, keywords, scrolls_per_query=100, checkpoint_every=10):
    """
    Scrape all keywords sequentially with checkpoints
    """
    all_urls = set()
    results = {}

    print(f"\n{'='*70}")
    print(f"Sequential Multi-Keyword Scraper")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Keywords: {len(keywords)}")
    print(f"Scrolls per keyword: {scrolls_per_query}")
    print(f"Checkpoint every: {checkpoint_every} keywords")
    print(f"{'='*70}\n")

    estimated_time = len(keywords) * scrolls_per_query * 8 / 3600
    print(f"Estimated time: {estimated_time:.1f} hours ({estimated_time*60:.0f} minutes)\n")

    start_time = time.time()

    for i, keyword in enumerate(keywords, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(keywords)}] Keyword: '{keyword}'")
        print(f"{'='*70}")

        keyword_start = time.time()

        # Scrape this keyword
        urls = scrape_single_keyword(base_url, keyword, scrolls_per_query)

        keyword_time = time.time() - keyword_start

        # Update stats
        before = len(all_urls)
        all_urls.update(urls)
        after = len(all_urls)
        new_unique = after - before

        results[keyword] = {
            'found': len(urls),
            'new_unique': new_unique,
            'time': keyword_time
        }

        # Calculate progress
        elapsed = time.time() - start_time
        avg_time_per_keyword = elapsed / i
        remaining_keywords = len(keywords) - i
        estimated_remaining = remaining_keywords * avg_time_per_keyword

        print(f"\n✓ Results:")
        print(f"  Found: {len(urls)} URLs")
        print(f"  New unique: {new_unique}")
        print(f"  Total unique: {len(all_urls)}")
        print(f"  Time: {keyword_time:.1f}s")
        print(f"\nProgress:")
        print(f"  Completed: {i}/{len(keywords)} ({i/len(keywords)*100:.1f}%)")
        print(f"  Elapsed: {elapsed/60:.1f} min")
        print(f"  Remaining: ~{estimated_remaining/60:.1f} min")
        print(f"  ETA: {(elapsed + estimated_remaining)/60:.1f} min total")

        # Checkpoint save
        if i % checkpoint_every == 0:
            checkpoint_file = f"checkpoint_{i}_keywords.txt"
            with open(checkpoint_file, 'w') as f:
                for url in sorted(all_urls):
                    f.write(f"{url}\n")
            print(f"\n💾 Checkpoint saved: {checkpoint_file} ({len(all_urls)} URLs)")

        # Brief pause between keywords
        time.sleep(2)

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'='*70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Keywords processed: {len(keywords)}")
    print(f"✓ Total unique URLs: {len(all_urls)}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"⚡ Average: {len(all_urls)/len(keywords):.0f} URLs per keyword")
    print(f"⚡ Speed: {len(keywords)/(elapsed/60):.2f} keywords/minute")
    print(f"{'='*70}")

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
    print("Sequential Multi-Keyword Scraper")
    print("="*70)
    print("Reliable scraping without resource limits\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_all_fight_keywords_batch.py <url> [scrolls] [output]")
            print()
            print("Examples:")
            print("  python scrape_all_fight_keywords_batch.py 'https://example.com'")
            print("  python scrape_all_fight_keywords_batch.py 'https://example.com' 100 videos.txt")
            print()
            print(f"Keywords: {len(FIGHT_KEYWORDS)} fight/violence terms")
            print()
            print("Features:")
            print("  ✓ No resource limits (sequential)")
            print("  ✓ Auto-checkpoints every 10 keywords")
            print("  ✓ Progress tracking with ETA")
            print("  ✓ Reliable and stable")
            sys.exit(0)

        base_url = sys.argv[1]
        scrolls_per_query = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        output_file = sys.argv[3] if len(sys.argv) > 3 else "all_fight_videos.txt"

    else:
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        scrolls_str = input("Scrolls per keyword (default 100): ").strip()
        scrolls_per_query = int(scrolls_str) if scrolls_str else 100

        output_file = input("Output file (default 'all_fight_videos.txt'): ").strip() or "all_fight_videos.txt"

    keywords = FIGHT_KEYWORDS

    print(f"\nConfiguration:")
    print(f"  {len(keywords)} keywords")
    print(f"  {scrolls_per_query} scrolls per keyword")
    print(f"  Sequential processing (no parallel)")

    estimated_time = len(keywords) * scrolls_per_query * 8 / 3600
    print(f"  Estimated: {estimated_time:.1f} hours")

    confirm = input("\nStart scraping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Scrape
    all_urls = scrape_all_keywords(base_url, keywords, scrolls_per_query)

    if not all_urls:
        print("\n❌ No URLs found!")
        return

    # Save
    save_urls(all_urls, output_file)

    print("\nDataset potential:")
    print(f"  - URLs found: {len(all_urls)}")
    print(f"  - If 90% downloadable: ~{int(len(all_urls) * 0.9)} videos")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
