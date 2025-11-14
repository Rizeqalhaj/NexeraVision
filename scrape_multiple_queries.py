#!/usr/bin/env python3
"""
Multi-Query Scraper
Runs multiple search queries to collect more videos
"""

import sys
from pathlib import Path
from scrape_infinite_scroll_improved import scrape_infinite_scroll_smart, save_urls

# Common violence/fight keywords
DEFAULT_QUERIES = [
    'fight',
    'street fight',
    'brawl',
    'violence',
    'assault',
    'attack',
    'cctv fight',
    'security camera fight',
    'surveillance fight',
    'caught on camera fight',
    'real fight',
    'fight compilation',
    'fight caught on camera',
    'public fight',
    'bar fight',
    'parking lot fight',
    'gas station fight',
    'store fight',
    'road rage',
    'altercation',
]

def scrape_multiple_queries(base_url, queries, max_scrolls=60, stop_after=5):
    """
    Scrape multiple search queries and combine results
    """
    all_urls = set()

    print(f"\n{'='*70}")
    print(f"Multi-Query Scraper")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Queries: {len(queries)}")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(queries)}: '{query}'")
        print(f"{'='*70}")

        urls = scrape_infinite_scroll_smart(base_url, query, max_scrolls, stop_after)

        before = len(all_urls)
        all_urls.update(urls)
        after = len(all_urls)
        new_unique = after - before

        print(f"\n✓ Query '{query}': Found {len(urls)} URLs ({new_unique} new unique)")
        print(f"✓ Total unique so far: {len(all_urls)}")

    return list(all_urls)

def main():
    print("="*70)
    print("Multi-Query Video Scraper")
    print("="*70)
    print("Automatically tries multiple search queries to find more videos\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_multiple_queries.py <url> [output] [max_scrolls] [stop_after]")
            print()
            print("Examples:")
            print("  python scrape_multiple_queries.py 'https://example.com'")
            print("  python scrape_multiple_queries.py 'https://example.com' all_videos.txt 100 5")
            print()
            print("Uses 20 predefined fight/violence search queries")
            sys.exit(0)

        base_url = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "all_video_urls.txt"
        max_scrolls = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        stop_after = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    else:
        # Interactive
        base_url = input("Website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        print("\nUse default queries or custom?")
        print("  1. Use 20 default fight/violence queries (recommended)")
        print("  2. Enter custom queries")
        choice = input("Choose (1 or 2): ").strip()

        if choice == '2':
            print("\nEnter queries (one per line, empty line to finish):")
            queries = []
            while True:
                q = input(f"Query {len(queries)+1}: ").strip()
                if not q:
                    break
                queries.append(q)
        else:
            queries = DEFAULT_QUERIES
            print(f"\n✓ Using {len(queries)} default queries")

        output_file = input("\nOutput file (default 'all_video_urls.txt'): ").strip() or "all_video_urls.txt"
        max_scrolls = int(input("Max scrolls per query (default 60): ").strip() or "60")
        stop_after = int(input("Stop after empty scrolls (default 5): ").strip() or "5")

    # Show queries that will be used
    if 'queries' not in locals():
        queries = DEFAULT_QUERIES

    print(f"\n{'='*70}")
    print("Queries to search:")
    print(f"{'='*70}")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print(f"{'='*70}\n")

    # Scrape
    all_urls = scrape_multiple_queries(base_url, queries, max_scrolls, stop_after)

    if not all_urls:
        print("\n❌ No URLs found across all queries!")
        return

    # Save
    save_urls(all_urls, output_file)

    print("\n" + "="*70)
    print("MULTI-QUERY SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ Searched {len(queries)} different queries")
    print(f"✓ Found {len(all_urls)} unique video URLs")
    print(f"✓ Saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
