#!/usr/bin/env python3
"""
Browse Page Scraper - No Keywords
Scrapes the main /videos/browse page for all available videos
Useful for collecting general video content without search filters
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin
import re

def scrape_browse_page(base_url, browse_path="/videos/browse", num_scrolls=200):
    """
    Scrape the main browse page without any keyword filtering

    Args:
        base_url: Base website URL (e.g., 'https://example.com')
        browse_path: Path to browse page (default: '/videos/browse')
        num_scrolls: Number of scrolls to perform (default: 200)

    Returns:
        List of video URLs found
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        print("Install with: pip install playwright && playwright install chromium")
        sys.exit(1)

    video_urls = set()

    print(f"\n{'='*70}")
    print(f"Browse Page Scraper")
    print(f"{'='*70}")
    print(f"Target: {base_url}{browse_path}")
    print(f"Scrolls: {num_scrolls}")
    print(f"Expected time: {num_scrolls * 8 / 3600:.1f} hours")
    print(f"{'='*70}\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        # Navigate to browse page
        browse_url = f"{base_url}{browse_path}"

        try:
            print(f"🌐 Loading: {browse_url}")
            page.goto(browse_url, timeout=30000, wait_until='networkidle')
            page.wait_for_timeout(5000)
            print(f"✓ Page loaded successfully\n")
        except Exception as e:
            print(f"❌ Failed to load page: {e}")
            browser.close()
            return []

        consecutive_empty = 0
        checkpoint_interval = 50  # Save checkpoint every 50 scrolls

        for scroll_num in range(1, num_scrolls + 1):
            before_count = len(video_urls)

            # Get ALL links and filter for video patterns
            # Pattern 1: /w/[id] format (common video ID pattern)
            # Pattern 2: /videos/watch/[id] or similar
            # Pattern 3: /v/[id] format
            all_links = page.query_selector_all('a[href]')

            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href:
                        # Match video URL patterns
                        if re.search(r'/w/[a-zA-Z0-9]+', href) or \
                           re.search(r'/videos?/watch/[a-zA-Z0-9]+', href) or \
                           re.search(r'/v/[a-zA-Z0-9]+', href):
                            full_url = urljoin(base_url, href)
                            video_urls.add(full_url)
                except:
                    pass

            after_count = len(video_urls)
            new_found = after_count - before_count

            # Track consecutive empty scrolls
            if new_found == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            # Progress reporting
            print(f"  Scroll {scroll_num}/{num_scrolls} | +{new_found} new | Total: {len(video_urls)} | Empty: {consecutive_empty}/10")

            # Stop if no new videos found for 10 consecutive scrolls
            if consecutive_empty >= 10:
                print(f"\n⚠️  No new videos found for {consecutive_empty} scrolls - stopping early")
                break

            # Save checkpoint
            if scroll_num % checkpoint_interval == 0:
                checkpoint_file = f"browse_checkpoint_{scroll_num}.txt"
                with open(checkpoint_file, 'w') as f:
                    for url in sorted(video_urls):
                        f.write(f"{url}\n")
                print(f"  💾 Checkpoint saved: {checkpoint_file} ({len(video_urls)} URLs)")

            # Aggressive scrolling strategy
            # Step 1: Scroll in increments (helps trigger lazy loading)
            for step in range(5):
                page.evaluate(f"window.scrollBy(0, {page.viewport_size['height'] / 5})")
                page.wait_for_timeout(400)

            # Step 2: Scroll to absolute bottom
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(6000)  # Wait for content to load

            # Step 3: Scroll up slightly then back down (triggers lazy load detection)
            page.evaluate("window.scrollBy(0, -300)")
            page.wait_for_timeout(1000)
            page.evaluate("window.scrollBy(0, 300)")
            page.wait_for_timeout(2000)

        browser.close()

    return list(video_urls)

def save_urls(urls, output_file):
    """Save URLs to file with deduplication"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove duplicates and sort
    unique_urls = sorted(set(urls))

    with open(output_path, 'w') as f:
        for url in unique_urls:
            f.write(f"{url}\n")

    print(f"\n✓ Saved {len(unique_urls)} unique URLs to: {output_path.absolute()}")
    return len(unique_urls)

def main():
    print("="*70)
    print("Browse Page Video Scraper (No Keywords)")
    print("="*70)
    print("Scrapes main browse page for all available videos")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_browse_page.py <base_url> [browse_path] [num_scrolls] [output]")
            print()
            print("Arguments:")
            print("  base_url      - Base website URL (required)")
            print("  browse_path   - Path to browse page (default: '/videos/browse')")
            print("  num_scrolls   - Number of scrolls (default: 200)")
            print("  output        - Output filename (default: 'browse_videos.txt')")
            print()
            print("Examples:")
            print("  python scrape_browse_page.py 'https://example.com'")
            print("  python scrape_browse_page.py 'https://example.com' '/videos/browse' 300")
            print("  python scrape_browse_page.py 'https://example.com' '/videos' 150 output.txt")
            print()
            print("Notes:")
            print("  - Automatically detects video URL patterns (/w/, /v/, /videos/watch/)")
            print("  - Saves checkpoints every 50 scrolls")
            print("  - Stops after 10 consecutive empty scrolls")
            sys.exit(0)

        base_url = sys.argv[1]
        browse_path = sys.argv[2] if len(sys.argv) > 2 else "/videos/browse"
        num_scrolls = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        output_file = sys.argv[4] if len(sys.argv) > 4 else "browse_videos.txt"

    else:
        # Interactive mode
        base_url = input("Website URL (e.g., https://example.com): ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        browse_path = input("Browse page path (default '/videos/browse'): ").strip() or "/videos/browse"
        num_scrolls_input = input("Number of scrolls (default 200): ").strip()
        num_scrolls = int(num_scrolls_input) if num_scrolls_input else 200
        output_file = input("Output file (default 'browse_videos.txt'): ").strip() or "browse_videos.txt"

    # Normalize base URL (remove trailing slash)
    base_url = base_url.rstrip('/')

    # Ensure browse path starts with /
    if not browse_path.startswith('/'):
        browse_path = '/' + browse_path

    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"{'='*70}")
    print(f"Base URL:     {base_url}")
    print(f"Browse Path:  {browse_path}")
    print(f"Full URL:     {base_url}{browse_path}")
    print(f"Scrolls:      {num_scrolls}")
    print(f"Output:       {output_file}")
    print(f"{'='*70}")

    confirmation = input("\nStart scraping? (y/n): ").strip().lower()
    if confirmation != 'y':
        print("Cancelled")
        sys.exit(0)

    # Scrape
    start_time = time.time()
    video_urls = scrape_browse_page(base_url, browse_path, num_scrolls)

    if not video_urls:
        print("\n❌ No video URLs found!")
        print("\nTroubleshooting:")
        print("  1. Check that the browse path is correct")
        print("  2. Verify the site uses /w/, /v/, or /videos/watch/ URL patterns")
        print("  3. Try increasing num_scrolls if page loads slowly")
        print("  4. Check network connectivity")
        return

    # Save
    unique_count = save_urls(video_urls, output_file)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ Target URL: {base_url}{browse_path}")
    print(f"✓ Scrolls performed: {num_scrolls}")
    print(f"✓ Unique videos found: {unique_count}")
    print(f"✓ Time elapsed: {elapsed / 60:.1f} minutes ({elapsed / 3600:.1f} hours)")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Rate: {unique_count / (elapsed / 60):.1f} videos/minute")
    print("\nNext steps:")
    print(f"  1. Review URLs in {output_file}")
    print(f"  2. Download videos using download script")
    print(f"  3. Filter for relevant content (fights/violence)")
    print("="*70)

if __name__ == "__main__":
    main()
