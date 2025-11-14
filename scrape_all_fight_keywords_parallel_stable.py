#!/usr/bin/env python3
"""
Stable Parallel Multi-Keyword Scraper
Better resource management to prevent browser crashes
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

class ScrapeStats:
    """Thread-safe statistics tracker"""
    def __init__(self):
        self.lock = threading.Lock()
        self.all_urls = set()
        self.keyword_results = {}
        self.errors = []

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

    def add_error(self, keyword, error):
        with self.lock:
            self.errors.append((keyword, error))

    def get_all_urls(self):
        with self.lock:
            return list(self.all_urls)

def scrape_single_keyword(base_url, keyword, num_scrolls, stats, index, total):
    """
    Scrape a single keyword with better error handling
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        print(f"❌ [{index}/{total}] Playwright not installed!")
        return []

    video_urls = set()
    max_retries = 2
    retry_count = 0

    while retry_count < max_retries:
        try:
            with sync_playwright() as p:
                # Launch with more conservative settings
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-dev-shm-usage',  # Prevent crashes on low /dev/shm
                        '--disable-gpu',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                    ]
                )

                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    viewport={'width': 1920, 'height': 1080}
                )
                page = context.new_page()

                # Set longer timeout for page load
                page.set_default_timeout(45000)  # 45 seconds

                # Build search URL
                encoded_query = quote(keyword)
                search_url = f"{base_url}/search?search={encoded_query}&searchTarget=local"

                # Load page with retry
                try:
                    page.goto(search_url, wait_until='domcontentloaded', timeout=45000)
                    page.wait_for_timeout(3000)
                except PlaywrightTimeout:
                    print(f"⚠️  [{index}/{total}] '{keyword}' - Page load timeout, retrying...")
                    browser.close()
                    retry_count += 1
                    time.sleep(5)
                    continue

                consecutive_empty = 0

                for scroll_num in range(1, num_scrolls + 1):
                    before_count = len(video_urls)

                    # Get all links
                    try:
                        all_links = page.query_selector_all('a[href]')
                        for link in all_links:
                            try:
                                href = link.get_attribute('href')
                                if href and re.search(r'/w/[a-zA-Z0-9]+', href):
                                    video_urls.add(urljoin(base_url, href))
                            except:
                                pass
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

                    # Gentler scrolling to prevent crashes
                    try:
                        for step in range(3):
                            page.evaluate(f"window.scrollBy(0, {page.viewport_size['height'] / 3})")
                            page.wait_for_timeout(500)

                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(5000)  # Longer wait between scrolls
                    except:
                        pass

                browser.close()
                break  # Success, exit retry loop

        except Exception as e:
            retry_count += 1
            error_msg = str(e)[:100]
            if retry_count < max_retries:
                print(f"⚠️  [{index}/{total}] '{keyword}' - Error: {error_msg}, retrying...")
                time.sleep(5)
            else:
                print(f"✗ [{index}/{total}] '{keyword}' - Failed after {max_retries} retries")
                stats.add_error(keyword, error_msg)
                return []

    # Add to global stats
    urls_list = list(video_urls)
    if urls_list:
        new_unique, total_unique = stats.add_urls(keyword, urls_list)
        print(f"✓ [{index}/{total}] '{keyword}': {len(urls_list)} found, {new_unique} new | Total: {total_unique}")
    else:
        print(f"⚠️  [{index}/{total}] '{keyword}': 0 URLs found")

    return urls_list

def scrape_parallel(base_url, keywords, scrolls_per_query=100, max_workers=5):
    """
    Scrape with conservative worker count to prevent crashes
    """

    print(f"\n{'='*70}")
    print(f"Stable Parallel Multi-Keyword Scraper")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Keywords: {len(keywords)}")
    print(f"Scrolls per keyword: {scrolls_per_query}")
    print(f"Parallel browsers: {max_workers} (conservative for stability)")
    print(f"{'='*70}\n")

    stats = ScrapeStats()
    start_time = time.time()

    print(f"{'='*70}")
    print(f"SCRAPING IN PROGRESS...")
    print(f"{'='*70}\n")

    # Parallel scraping with conservative settings
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    print(f"✓ Successful: {len(stats.keyword_results)}")
    print(f"✗ Failed: {len(stats.errors)}")
    print(f"✓ Total unique URLs: {len(all_urls)}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"⚡ Speed: {len(keywords)/(elapsed/60):.1f} keywords/minute")
    print(f"⚡ Average: {len(all_urls)/len(stats.keyword_results):.0f} URLs per successful keyword")

    if stats.errors:
        print(f"\n⚠️  Failed keywords ({len(stats.errors)}):")
        for kw, err in stats.errors[:10]:
            print(f"  - {kw}: {err[:50]}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")

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
    print("Stable Parallel Multi-Keyword Scraper")
    print("="*70)
    print("Optimized for stability with automatic retry\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_all_fight_keywords_parallel_stable.py <url> [scrolls] [workers] [output]")
            print()
            print("Arguments:")
            print("  url      : Website URL")
            print("  scrolls  : Scrolls per keyword (default: 100)")
            print("  workers  : Parallel browsers (default: 5)")
            print("  output   : Output file (default: all_fight_videos.txt)")
            print()
            print("Examples:")
            print("  # Conservative: 5 workers (most stable)")
            print("  python scrape_all_fight_keywords_parallel_stable.py 'https://example.com'")
            print()
            print("  # Moderate: 8 workers (good balance)")
            print("  python scrape_all_fight_keywords_parallel_stable.py 'https://example.com' 100 8")
            print()
            print("  # Fast: 10 workers (if system handles it)")
            print("  python scrape_all_fight_keywords_parallel_stable.py 'https://example.com' 100 10")
            print()
            print("Workers recommendation (for stability):")
            print("  - Start with 5 workers")
            print("  - If stable, increase to 8")
            print("  - Maximum 10-12 workers to prevent crashes")
            print()
            print(f"Keywords: {len(FIGHT_KEYWORDS)} fight/violence terms")
            sys.exit(0)

        base_url = sys.argv[1]
        scrolls_per_query = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        output_file = sys.argv[4] if len(sys.argv) > 4 else "all_fight_videos.txt"

    else:
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        scrolls_str = input("Scrolls per keyword (default 100): ").strip()
        scrolls_per_query = int(scrolls_str) if scrolls_str else 100

        workers_str = input("Parallel browsers (default 5 for stability): ").strip()
        max_workers = int(workers_str) if workers_str else 5

        output_file = input("Output file (default 'all_fight_videos.txt'): ").strip() or "all_fight_videos.txt"

    keywords = FIGHT_KEYWORDS

    print(f"\nConfiguration:")
    print(f"  {len(keywords)} keywords")
    print(f"  {scrolls_per_query} scrolls per keyword")
    print(f"  {max_workers} parallel browsers")

    estimated_time = len(keywords) * scrolls_per_query * 8 / 3600 / max_workers
    print(f"  Estimated: {estimated_time:.1f} hours ({estimated_time*60:.0f} minutes)")

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

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
