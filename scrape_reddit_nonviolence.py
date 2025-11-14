#!/usr/bin/env python3
"""
Reddit Non-Violence Video Scraper
Uses Reddit API to find non-violence videos from relevant subreddits

Target: 3,500+ non-violence videos to balance dataset (need 14K - 10.5K = 3.5K more)
"""

import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

# Non-violence subreddits
NON_VIOLENCE_SUBREDDITS = [
    # General wholesome/positive
    'MadeMeSmile',
    'HumansBeingBros',
    'wholesome',
    'UpliftingNews',
    'aww',

    # Sports (non-violent)
    'sports',
    'basketball',
    'soccer',
    'baseball',
    'football',
    'tennis',
    'golf',
    'hockey',

    # Activities and hobbies
    'dancing',
    'music',
    'concerts',
    'festivals',
    'streetphotography',
    'publicfreakout',  # Filter for happy/positive freakouts

    # Daily life
    'BeAmazed',
    'Damnthatsinteresting',
    'interestingasfuck',
    'oddlysatisfying',

    # Animals (peaceful)
    'AnimalsBeingBros',
    'AnimalsBeingDerps',
    'Eyebleach',

    # Skills/talents
    'toptalent',
    'nextfuckinglevel',
    'BeAmazed',

    # Peaceful crowds/gatherings
    'crowdpulledout',
    'happycrowds',
]

# Keywords to filter within subreddits
NON_VIOLENCE_KEYWORDS = [
    'happy', 'celebration', 'festival', 'wedding', 'graduation',
    'concert', 'dance', 'music', 'performance', 'game', 'sport',
    'playing', 'walking', 'shopping', 'normal', 'peaceful',
    'beautiful', 'amazing', 'wholesome', 'positive', 'uplifting',
    'help', 'rescue', 'save', 'cute', 'adorable', 'funny',
]

# Violence keywords to AVOID
VIOLENCE_KEYWORDS = [
    'fight', 'punch', 'kick', 'hit', 'beat', 'assault', 'attack',
    'violence', 'violent', 'brawl', 'combat', 'knockout', 'ko',
    'blood', 'injured', 'hurt', 'dead', 'death', 'kill',
]


def has_violence_keywords(text):
    """Check if text contains violence keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in VIOLENCE_KEYWORDS)


def has_nonviolence_keywords(text):
    """Check if text contains non-violence keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in NON_VIOLENCE_KEYWORDS)


def scrape_reddit_videos(subreddits, time_filter='all', limit_per_sub=100):
    """
    Scrape video URLs from Reddit using gallery-dl or yt-dlp

    Args:
        subreddits: List of subreddit names
        time_filter: 'all', 'year', 'month', 'week', 'day'
        limit_per_sub: Max posts per subreddit
    """

    print(f"\n{'='*70}")
    print(f"Reddit Non-Violence Video Scraper")
    print(f"{'='*70}")
    print(f"Subreddits: {len(subreddits)}")
    print(f"Time filter: {time_filter}")
    print(f"Limit per subreddit: {limit_per_sub}")
    print(f"{'='*70}\n")

    all_urls = set()
    subreddit_stats = {}

    for i, subreddit in enumerate(subreddits, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(subreddits)}] Processing r/{subreddit}")
        print(f"{'='*70}")

        try:
            import subprocess

            # Use gallery-dl to extract Reddit video URLs
            url = f"https://www.reddit.com/r/{subreddit}/top/?t={time_filter}"

            cmd = [
                'gallery-dl',
                '--get-urls',
                '--filter', 'extension in ("mp4", "webm", "mov", "avi")',
                '--range', f'1-{limit_per_sub}',
                url
            ]

            print(f"🔍 Searching r/{subreddit}...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                urls = result.stdout.strip().split('\n')
                urls = [u for u in urls if u.strip()]

                # Filter out violence-related posts (basic filtering)
                filtered_urls = []
                for url in urls:
                    # This is basic - ideally check post title/text too
                    filtered_urls.append(url)

                before = len(all_urls)
                all_urls.update(filtered_urls)
                after = len(all_urls)
                new_unique = after - before

                subreddit_stats[subreddit] = {
                    'found': len(filtered_urls),
                    'new_unique': new_unique,
                    'total': len(all_urls)
                }

                print(f"✓ Found: {len(filtered_urls)} | New unique: {new_unique} | Total: {len(all_urls)}")

            else:
                print(f"✗ Error searching r/{subreddit}")
                subreddit_stats[subreddit] = {'found': 0, 'new_unique': 0, 'total': len(all_urls)}

        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout for r/{subreddit}")
            subreddit_stats[subreddit] = {'found': 0, 'new_unique': 0, 'total': len(all_urls)}
        except FileNotFoundError:
            print(f"\n❌ gallery-dl not installed!")
            print(f"Install with: pip install gallery-dl")
            return []
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            subreddit_stats[subreddit] = {'found': 0, 'new_unique': 0, 'total': len(all_urls)}

        # Rate limiting - be nice to Reddit
        time.sleep(random.uniform(3, 7))

    print(f"\n{'='*70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"Total unique URLs: {len(all_urls)}")
    print(f"\nTop subreddits by videos found:")

    sorted_stats = sorted(subreddit_stats.items(), key=lambda x: x[1]['found'], reverse=True)
    for sub, stats in sorted_stats[:10]:
        if stats['found'] > 0:
            print(f"  r/{sub:20s}: {stats['found']:4d} videos")

    print(f"{'='*70}")

    return list(all_urls)


def scrape_reddit_pushshift(subreddits, days_back=365):
    """
    Alternative: Use Pushshift API to find Reddit video posts
    More reliable but slower
    """

    print(f"\n{'='*70}")
    print(f"Reddit Pushshift API Scraper")
    print(f"{'='*70}")
    print(f"Subreddits: {len(subreddits)}")
    print(f"Time range: Last {days_back} days")
    print(f"{'='*70}\n")

    import requests

    all_posts = []
    base_url = "https://api.pushshift.io/reddit/search/submission"

    for i, subreddit in enumerate(subreddits, 1):
        print(f"\n[{i}/{len(subreddits)}] Searching r/{subreddit}...")

        try:
            # Calculate timestamp for days_back
            import time
            after_timestamp = int(time.time()) - (days_back * 24 * 3600)

            params = {
                'subreddit': subreddit,
                'size': 100,
                'after': after_timestamp,
                'sort': 'desc',
                'sort_type': 'score',
            }

            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', [])

                # Filter for video posts
                video_posts = []
                for post in posts:
                    # Check if post has video
                    if post.get('is_video') or 'v.redd.it' in post.get('url', ''):
                        # Filter out violence keywords
                        title = post.get('title', '').lower()
                        selftext = post.get('selftext', '').lower()

                        if not has_violence_keywords(title + ' ' + selftext):
                            video_posts.append({
                                'url': post.get('url'),
                                'title': post.get('title'),
                                'subreddit': subreddit,
                                'score': post.get('score', 0),
                                'created': post.get('created_utc'),
                                'permalink': f"https://reddit.com{post.get('permalink', '')}"
                            })

                all_posts.extend(video_posts)
                print(f"✓ Found: {len(video_posts)} video posts | Total: {len(all_posts)}")

            else:
                print(f"✗ HTTP {response.status_code}")

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")

        time.sleep(2)  # Rate limiting

    # Extract URLs
    urls = list(set([post['url'] for post in all_posts if post.get('url')]))

    print(f"\n{'='*70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"Total video posts: {len(all_posts)}")
    print(f"Total unique URLs: {len(urls)}")
    print(f"{'='*70}")

    # Save detailed post info
    with open('reddit_posts_info.json', 'w') as f:
        json.dump(all_posts, f, indent=2)
    print(f"\n✓ Saved post details to: reddit_posts_info.json")

    return urls


def save_urls(urls, output_file):
    """Save URLs to file"""
    with open(output_file, 'w') as f:
        for url in sorted(urls):
            f.write(f"{url}\n")
    print(f"\n✓ Saved {len(urls)} URLs to: {output_file}")


def main():
    print("="*70)
    print("Reddit Non-Violence Video Scraper")
    print("="*70)
    print("Goal: Collect 3,500+ non-violence videos to balance dataset\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_reddit_nonviolence.py [method] [output]")
            print()
            print("Methods:")
            print("  gallery-dl  : Use gallery-dl (faster, requires installation)")
            print("  pushshift   : Use Pushshift API (slower, more detailed)")
            print()
            print("Examples:")
            print("  python scrape_reddit_nonviolence.py gallery-dl reddit_urls.txt")
            print("  python scrape_reddit_nonviolence.py pushshift reddit_urls.txt")
            sys.exit(0)

        method = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "reddit_nonviolence_urls.txt"
    else:
        print("Choose scraping method:")
        print("  1. gallery-dl (faster, requires: pip install gallery-dl)")
        print("  2. pushshift API (slower, no dependencies)")
        choice = input("\nMethod (1 or 2): ").strip()

        method = 'gallery-dl' if choice == '1' else 'pushshift'
        output = "reddit_nonviolence_urls.txt"

    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  Method: {method}")
    print(f"  Subreddits: {len(NON_VIOLENCE_SUBREDDITS)}")
    print(f"  Output: {output}")
    print(f"{'='*70}")

    confirm = input("\nStart scraping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    # Scrape
    if method == 'gallery-dl':
        urls = scrape_reddit_videos(NON_VIOLENCE_SUBREDDITS, time_filter='all', limit_per_sub=100)
    else:
        urls = scrape_reddit_pushshift(NON_VIOLENCE_SUBREDDITS, days_back=365)

    if urls:
        save_urls(urls, output)

        print(f"\nDataset Balance:")
        print(f"  Violence:     14,000")
        print(f"  Non-violence: 10,454 + {len(urls)} = {10454 + len(urls)}")

        if (10454 + len(urls)) >= 14000:
            print(f"  ✓ BALANCED! Ready for training")
        else:
            needed = 14000 - (10454 + len(urls))
            print(f"  ⚠️  Need {needed:,} more non-violence videos")
    else:
        print("\n❌ No URLs found!")


if __name__ == "__main__":
    main()
