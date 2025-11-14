#!/usr/bin/env python3
"""
Pixabay Video Scraper - FREE Stock Videos
Official API - No scraping, fast and reliable

Get FREE API key: https://pixabay.com/api/docs/
Expected yield: 3,000-8,000 non-violence videos
"""

import sys
import time
import requests
from pathlib import Path

# Non-violence categories and keywords
PIXABAY_QUERIES = [
    # Nature
    'nature', 'ocean', 'sea', 'beach', 'mountain', 'forest', 'tree',
    'flower', 'garden', 'sunset', 'sunrise', 'sky', 'cloud', 'rain',
    'snow', 'river', 'lake', 'waterfall', 'landscape',

    # Urban
    'city', 'street', 'building', 'architecture', 'traffic', 'road',
    'bridge', 'night city', 'downtown', 'skyline',

    # People and activities
    'people', 'walking', 'running', 'cycling', 'family', 'children',
    'baby', 'friends', 'happy', 'smile', 'celebration', 'party',

    # Sports
    'sports', 'football', 'basketball', 'soccer', 'tennis', 'golf',
    'swimming', 'surfing', 'skiing', 'skateboard', 'yoga', 'exercise',

    # Work
    'business', 'office', 'computer', 'typing', 'meeting', 'work',
    'handshake', 'team', 'conference',

    # Transportation
    'car', 'train', 'airplane', 'boat', 'bus', 'bicycle', 'highway',
    'airport', 'subway',

    # Food
    'food', 'cooking', 'restaurant', 'coffee', 'tea', 'fruit',
    'vegetables', 'bread', 'cake',

    # Animals
    'dog', 'cat', 'bird', 'fish', 'horse', 'butterfly', 'wildlife',

    # Abstract and backgrounds
    'abstract', 'background', 'texture', 'pattern', 'light', 'bokeh',
    'particles', 'smoke', 'fire', 'water',

    # Technology
    'technology', 'digital', 'data', 'network', 'code', 'screen',

    # Art and culture
    'art', 'music', 'dance', 'painting', 'sculpture', 'performance',
]


def scrape_pixabay(api_key, queries, videos_per_query=200, output_file='pixabay_urls.txt'):
    """
    Scrape video URLs from Pixabay using official API

    Args:
        api_key: Pixabay API key (get from https://pixabay.com/api/docs/)
        queries: List of search queries
        videos_per_query: Max videos per query (max 200)
        output_file: Output file for URLs
    """

    print(f"\n{'='*70}")
    print(f"Pixabay Video Scraper")
    print(f"{'='*70}")
    print(f"Queries: {len(queries)}")
    print(f"Videos per query: {videos_per_query}")
    print(f"Expected total: {len(queries) * videos_per_query * 0.6:.0f} (with deduplication)")
    print(f"{'='*70}\n")

    base_url = "https://pixabay.com/api/videos/"

    all_urls = set()
    all_video_info = {}

    start_time = time.time()

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Searching: '{query}'")

        # Pixabay API allows max 200 results, paginated
        page = 1
        query_urls = 0

        while True:
            try:
                params = {
                    'key': api_key,
                    'q': query,
                    'per_page': min(200, videos_per_query),
                    'page': page
                }

                response = requests.get(base_url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    videos = data.get('hits', [])

                    if not videos:
                        break  # No more results

                    for video in videos:
                        video_id = video.get('id')

                        # Get download URL (medium quality - good for training)
                        video_files = video.get('videos', {})

                        # Try to get medium or small quality (not huge files)
                        download_url = None
                        quality_preference = ['medium', 'small', 'large', 'tiny']

                        for quality in quality_preference:
                            if quality in video_files:
                                download_url = video_files[quality].get('url')
                                break

                        if download_url:
                            all_urls.add(download_url)
                            all_video_info[video_id] = {
                                'download_url': download_url,
                                'page_url': video.get('pageURL'),
                                'duration': video.get('duration', 0),
                                'tags': video.get('tags', ''),
                                'user': video.get('user', '')
                            }
                            query_urls += 1

                    # Check if we got all results
                    total_hits = data.get('totalHits', 0)
                    if page * 200 >= total_hits or page * 200 >= videos_per_query:
                        break

                    page += 1
                    time.sleep(0.5)  # Small delay between pages

                elif response.status_code == 429:
                    print(f"  ⚠️  Rate limited, waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"  ✗ HTTP {response.status_code}")
                    break

            except Exception as e:
                print(f"  ✗ Error: {str(e)[:50]}")
                break

        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (len(queries) - i) * avg_time

        print(f"  ✓ Found: {query_urls} | Total unique: {len(all_urls)}")
        print(f"  ⏱️  {elapsed/60:.1f}m elapsed | ~{remaining/60:.1f}m remaining")

        time.sleep(1)  # Delay between queries

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Total unique videos: {len(all_urls)}")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # Save URLs
    with open(output_file, 'w') as f:
        for url in sorted(all_urls):
            f.write(f"{url}\n")

    print(f"\n✓ Saved URLs to: {output_file}")

    # Save metadata
    import json
    metadata_file = output_file.replace('.txt', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(all_video_info, f, indent=2)

    print(f"✓ Saved metadata to: {metadata_file}")

    return list(all_urls)


def main():
    print("="*70)
    print("Pixabay Video Scraper - FREE Stock Videos")
    print("="*70)
    print("Get API key: https://pixabay.com/api/docs/")
    print("Expected yield: 3,000-8,000 videos\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_pixabay_videos.py <api_key> [output]")
            print()
            print("Example:")
            print("  python scrape_pixabay_videos.py YOUR_API_KEY pixabay_urls.txt")
            print()
            print("Get API key:")
            print("  1. Go to https://pixabay.com/api/docs/")
            print("  2. Sign up (free)")
            print("  3. Copy your API key")
            sys.exit(0)

        api_key = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "pixabay_urls.txt"
    else:
        api_key = input("Enter your Pixabay API key: ").strip()
        if not api_key:
            print("\n❌ API key required!")
            print("Get one at: https://pixabay.com/api/docs/")
            sys.exit(1)

        output = input("Output file (default 'pixabay_urls.txt'): ").strip() or "pixabay_urls.txt"

    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  API key: {api_key[:10]}...")
    print(f"  Queries: {len(PIXABAY_QUERIES)}")
    print(f"  Output: {output}")
    print(f"{'='*70}")

    confirm = input("\nStart scraping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Scrape
    urls = scrape_pixabay(api_key, PIXABAY_QUERIES, videos_per_query=200, output_file=output)

    if urls:
        print(f"\nDataset Balance:")
        print(f"  Violence:     14,000")
        print(f"  Non-violence: 10,454 + {len(urls)} = {10454 + len(urls):,}")

        if (10454 + len(urls)) >= 14000:
            print(f"  ✓ BALANCED! Ready for training")
            print(f"  📊 Expected accuracy: 93-95%")
        else:
            needed = 14000 - (10454 + len(urls))
            print(f"  ⚠️  Need {needed:,} more non-violence videos")

        print(f"\n{'='*70}")
        print(f"Next step: Download videos")
        print(f"{'='*70}")
        print(f"python download_videos_parallel_robust.py {output} pixabay_videos/ 30")
    else:
        print("\n❌ No URLs found!")


if __name__ == "__main__":
    main()
