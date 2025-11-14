#!/usr/bin/env python3
"""
Pexels Video Scraper - FREE Stock Videos
Official API - No scraping, no rate limits

Get FREE API key: https://www.pexels.com/api/
Expected yield: 5,000-10,000 non-violence videos
"""

import sys
import time
import requests
from pathlib import Path

# Non-violence search queries for Pexels
PEXELS_QUERIES = [
    # Nature and landscapes
    'nature', 'ocean', 'mountains', 'forest', 'beach', 'sunset', 'sunrise',
    'river', 'waterfall', 'lake', 'sky', 'clouds', 'rain', 'snow',

    # Urban and daily life
    'city', 'street', 'traffic', 'people walking', 'shopping', 'cafe',
    'restaurant', 'park', 'garden', 'building', 'architecture',

    # Activities
    'dancing', 'running', 'cycling', 'swimming', 'yoga', 'exercise',
    'cooking', 'eating', 'working', 'studying', 'reading', 'writing',

    # Sports
    'basketball', 'soccer', 'football', 'tennis', 'golf', 'baseball',
    'volleyball', 'skateboard', 'surfing', 'skiing', 'skating',

    # Social
    'family', 'friends', 'children playing', 'celebration', 'party',
    'wedding', 'graduation', 'meeting', 'conference', 'teamwork',

    # Animals
    'dog', 'cat', 'bird', 'fish', 'horse', 'wildlife', 'pets',

    # Transportation
    'car', 'bus', 'train', 'airplane', 'boat', 'bicycle', 'subway',

    # Work and business
    'office', 'business', 'computer', 'typing', 'phone call', 'handshake',

    # Entertainment
    'music', 'concert', 'festival', 'art', 'painting', 'performance',

    # Food
    'food', 'cooking', 'baking', 'vegetables', 'fruit', 'coffee', 'tea',
]


def scrape_pexels(api_key, queries, videos_per_query=80, output_file='pexels_urls.txt'):
    """
    Scrape video URLs from Pexels using official API

    Args:
        api_key: Pexels API key (get from https://www.pexels.com/api/)
        queries: List of search queries
        videos_per_query: Max videos per query (max 80 per request)
        output_file: Output file for URLs
    """

    print(f"\n{'='*70}")
    print(f"Pexels Video Scraper")
    print(f"{'='*70}")
    print(f"Queries: {len(queries)}")
    print(f"Videos per query: {videos_per_query}")
    print(f"Expected total: {len(queries) * videos_per_query * 0.7:.0f} (with deduplication)")
    print(f"{'='*70}\n")

    base_url = "https://api.pexels.com/videos/search"
    headers = {'Authorization': api_key}

    all_urls = set()
    all_download_urls = {}  # video_id -> download_url

    start_time = time.time()

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Searching: '{query}'")

        try:
            params = {
                'query': query,
                'per_page': min(videos_per_query, 80),  # Max 80 per request
                'page': 1
            }

            response = requests.get(base_url, headers=headers, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                videos = data.get('videos', [])

                before = len(all_urls)

                for video in videos:
                    video_id = video.get('id')
                    video_url = video.get('url')  # Pexels page URL

                    # Get download URL (highest quality)
                    video_files = video.get('video_files', [])
                    if video_files:
                        # Sort by quality (highest first)
                        video_files_sorted = sorted(
                            video_files,
                            key=lambda x: x.get('width', 0) * x.get('height', 0),
                            reverse=True
                        )

                        # Get HD or lower (not 4K to save bandwidth)
                        for vf in video_files_sorted:
                            if vf.get('width', 0) <= 1920:  # Max 1080p
                                download_url = vf.get('link')
                                all_download_urls[video_id] = {
                                    'page_url': video_url,
                                    'download_url': download_url,
                                    'width': vf.get('width'),
                                    'height': vf.get('height'),
                                    'quality': vf.get('quality'),
                                    'duration': video.get('duration', 0)
                                }
                                all_urls.add(download_url)
                                break

                after = len(all_urls)
                new_unique = after - before

                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(queries) - i) * avg_time

                print(f"  ✓ Found: {len(videos)} | New unique: {new_unique} | Total: {len(all_urls)}")
                print(f"  ⏱️  {elapsed/60:.1f}m elapsed | ~{remaining/60:.1f}m remaining")

            elif response.status_code == 429:
                print(f"  ⚠️  Rate limited, waiting 60s...")
                time.sleep(60)
            else:
                print(f"  ✗ HTTP {response.status_code}")

        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")

        # Small delay to be nice to API
        time.sleep(1)

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
        json.dump(all_download_urls, f, indent=2)

    print(f"✓ Saved metadata to: {metadata_file}")

    return list(all_urls)


def main():
    print("="*70)
    print("Pexels Video Scraper - FREE Stock Videos")
    print("="*70)
    print("Get API key: https://www.pexels.com/api/")
    print("Expected yield: 5,000-10,000 videos\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_pexels_videos.py <api_key> [output]")
            print()
            print("Example:")
            print("  python scrape_pexels_videos.py YOUR_API_KEY pexels_urls.txt")
            print()
            print("Get API key:")
            print("  1. Go to https://www.pexels.com/api/")
            print("  2. Sign up (free)")
            print("  3. Copy your API key")
            sys.exit(0)

        api_key = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "pexels_urls.txt"
    else:
        api_key = input("Enter your Pexels API key: ").strip()
        if not api_key:
            print("\n❌ API key required!")
            print("Get one at: https://www.pexels.com/api/")
            sys.exit(1)

        output = input("Output file (default 'pexels_urls.txt'): ").strip() or "pexels_urls.txt"

    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  API key: {api_key[:10]}...")
    print(f"  Queries: {len(PEXELS_QUERIES)}")
    print(f"  Output: {output}")
    print(f"{'='*70}")

    confirm = input("\nStart scraping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Scrape
    urls = scrape_pexels(api_key, PEXELS_QUERIES, videos_per_query=80, output_file=output)

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
        print(f"python download_videos_parallel_robust.py {output} pexels_videos/ 30")
    else:
        print("\n❌ No URLs found!")


if __name__ == "__main__":
    main()
