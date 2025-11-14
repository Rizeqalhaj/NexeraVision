#!/usr/bin/env python3
"""
Parallel Multi-Keyword Scraper
Scrapes multiple keywords simultaneously using multiple browser instances
Optimized for high-spec systems (260GB RAM, powerful CPU)
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin, quote
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Comprehensive fight/violence keywords
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

class ScrapeStats:
    """Thread-safe statistics tracker"""
    def __init__(self):
        self.lock = threading.Lock()
        self.all_urls = set()
        self.keyword_results = {}

    def add_urls(self, keyword, urls):
        with self.lock:
            before = len(self.all_urls)
            self.all_urls.update(urls)
            after = len(self.all_urls)
            new_unique = after - before
            self.keyword_results[keyword] = {
                'found': len(urls),
                'new_unique': new_unique,
                'total_after': after
            }
            return new_unique, after

    def get_all_urls(self):
        with self.lock:
            return list(self.all_urls)

    def get_stats(self):
        with self.lock:
            return len(self.all_urls), len(self.keyword_results)

def scrape_single_keyword(base_url, keyword, num_scrolls, stats, index, total):
    """
    Scrape a single keyword (runs in thread with own browser instance)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"❌ [{index}/{total}] Playwright not installed!")
        return []

    video_urls = set()

    try:
        with sync_playwright() as p:
            # Each thread gets its own browser instance
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()

            # Build search URL
            encoded_query = quote(keyword)
            search_url = f"{base_url}/search?search={encoded_query}&searchTarget=local"

            # Load page
            page.goto(search_url, timeout=30000, wait_until='networkidle')
            page.wait_for_timeout(3000)

            consecutive_empty = 0

            for scroll_num in range(1, num_scrolls + 1):
                before_count = len(video_urls)

                # Get all links and filter for /w/ pattern
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

                # Stop early if no new content
                if consecutive_empty >= 5:
                    break

                # Aggressive scrolling
                for step in range(5):
                    page.evaluate(f"window.scrollBy(0, {page.viewport_size['height'] / 5})")
                    page.wait_for_timeout(300)

                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(4000)

                page.evaluate("window.scrollBy(0, -300)")
                page.wait_for_timeout(800)
                page.evaluate("window.scrollBy(0, 300)")
                page.wait_for_timeout(1500)

            browser.close()

    except Exception as e:
        print(f"✗ [{index}/{total}] '{keyword}' - Error: {str(e)[:50]}")
        return []

    # Add to global stats
    urls_list = list(video_urls)
    new_unique, total_unique = stats.add_urls(keyword, urls_list)

    # Print result
    print(f"✓ [{index}/{total}] '{keyword}': {len(urls_list)} found, {new_unique} new unique | Total: {total_unique}")

    return urls_list

def scrape_parallel(base_url, keywords, scrolls_per_query=100, max_workers=10):
    """
    Scrape multiple keywords in parallel

    Args:
        base_url: Website URL
        keywords: List of keywords to scrape
        scrolls_per_query: Scrolls per keyword
        max_workers: Number of parallel browser instances (5-20 depending on RAM)
    """

    print(f"\n{'='*70}")
    print(f"Parallel Multi-Keyword Scraper")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Keywords: {len(keywords)}")
    print(f"Scrolls per keyword: {scrolls_per_query}")
    print(f"Parallel browsers: {max_workers}")
    print(f"{'='*70}\n")

    # Calculate time estimate
    # Serial time: 51 keywords × 100 scrolls × 8 sec/scroll = ~11 hours
    # Parallel time: 11 hours / workers = much faster
    serial_time = len(keywords) * scrolls_per_query * 8 / 3600
    parallel_time = serial_time / max_workers
    print(f"Estimated time:")
    print(f"  Serial: {serial_time:.1f} hours")
    print(f"  Parallel ({max_workers} workers): {parallel_time:.1f} hours")
    print(f"  Speed boost: {max_workers}x faster!\n")

    stats = ScrapeStats()
    start_time = time.time()

    print(f"{'='*70}")
    print(f"SCRAPING IN PROGRESS...")
    print(f"{'='*70}\n")

    # Parallel scraping using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all keyword scraping tasks
        futures = {
            executor.submit(
                scrape_single_keyword,
                base_url,
                keyword,
                scrolls_per_query,
                stats,
                i,
                len(keywords)
            ): (i, keyword)
            for i, keyword in enumerate(keywords, 1)
        }

        # Wait for all to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                idx, kw = futures[future]
                print(f"✗ Thread error for '{kw}': {e}")

    elapsed = time.time() - start_time
    all_urls = stats.get_all_urls()

    # Final summary
    print(f"\n{'='*70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Keywords scraped: {len(keywords)}")
    print(f"✓ Total unique URLs: {len(all_urls)}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"⚡ Speed: {len(keywords)/(elapsed/60):.1f} keywords/minute")
    print(f"⚡ Average: {len(all_urls)/len(keywords):.0f} URLs per keyword")
    print(f"{'='*70}")

    return all_urls

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
    print("Parallel Multi-Keyword Scraper (High-Performance)")
    print("="*70)
    print("Optimized for high-spec systems (260GB RAM, powerful CPU)\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_all_fight_keywords_parallel.py <url> [scrolls] [workers] [output]")
            print()
            print("Arguments:")
            print("  url      : Website URL")
            print("  scrolls  : Scrolls per keyword (default: 100)")
            print("  workers  : Parallel browsers (default: 10)")
            print("  output   : Output file (default: all_fight_videos.txt)")
            print()
            print("Examples:")
            print("  # Default: 10 parallel browsers")
            print("  python scrape_all_fight_keywords_parallel.py 'https://example.com'")
            print()
            print("  # 20 parallel browsers (260GB RAM)")
            print("  python scrape_all_fight_keywords_parallel.py 'https://example.com' 100 20")
            print()
            print("  # Fast test: 5 workers, 50 scrolls")
            print("  python scrape_all_fight_keywords_parallel.py 'https://example.com' 50 5")
            print()
            print("Workers recommendation (based on RAM):")
            print("  - 16GB RAM  : 3-5 workers")
            print("  - 32GB RAM  : 5-8 workers")
            print("  - 64GB RAM  : 8-12 workers")
            print("  - 128GB RAM : 12-16 workers")
            print("  - 256GB RAM : 15-20 workers (your system!)")
            print()
            print(f"Default keywords: {len(FIGHT_KEYWORDS)} fight/violence terms")
            sys.exit(0)

        base_url = sys.argv[1]
        scrolls_per_query = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        output_file = sys.argv[4] if len(sys.argv) > 4 else "all_fight_videos.txt"

    else:
        # Interactive
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        scrolls_str = input("Scrolls per keyword (default 100): ").strip()
        scrolls_per_query = int(scrolls_str) if scrolls_str else 100

        workers_str = input("Parallel browsers (default 10, recommend 15-20 for 260GB RAM): ").strip()
        max_workers = int(workers_str) if workers_str else 10

        output_file = input("Output file (default 'all_fight_videos.txt'): ").strip() or "all_fight_videos.txt"

    keywords = FIGHT_KEYWORDS

    # Show keyword list
    print(f"\n{'='*70}")
    print(f"Keywords ({len(keywords)} total):")
    print(f"{'='*70}")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i:2}. {kw}")
    print(f"{'='*70}")

    # Confirmation
    serial_time = len(keywords) * scrolls_per_query * 8 / 3600
    parallel_time = serial_time / max_workers

    print(f"\nConfiguration:")
    print(f"  {len(keywords)} keywords × {scrolls_per_query} scrolls × {max_workers} parallel")
    print(f"  Estimated: {parallel_time:.1f} hours ({parallel_time*60:.0f} minutes)")
    print(f"  vs Serial: {serial_time:.1f} hours")
    print(f"  Speedup: {max_workers}x faster!")

    confirm = input("\nStart scraping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Scrape
    all_urls = scrape_parallel(base_url, keywords, scrolls_per_query, max_workers)

    if not all_urls:
        print("\n❌ No URLs found!")
        return

    # Save
    save_urls(all_urls, output_file)

    # Dataset summary
    print("\nDataset potential:")
    print(f"  - URLs found: {len(all_urls)}")
    print(f"  - If 90% downloadable: ~{int(len(all_urls) * 0.9)} videos")
    print(f"  - Current dataset: ~15K videos")
    print(f"  - New total: ~{15000 + int(len(all_urls) * 0.9)} videos")
    print(f"  - Expected accuracy boost: +2-3%")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
