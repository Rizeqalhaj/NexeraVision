#!/usr/bin/env python3
"""
Quick Subreddit Verification Script
Tests which subreddits are accessible and return posts
"""

import requests
import time

# Test all fight subreddits
TEST_SUBREDDITS = [
    'fightporn',
    'PublicFreakout',
    'CrazyFuckingVideos',
    'AbruptChaos',
    'StreetFights',
    'ActualFreakouts',
    'DocumentedFights',
    'BrutalBeatdowns',
    'ThunderThots',
    'StreetMartialArts',
    'fights',
    'StreetFightManiac',
]

def test_subreddit(subreddit_name):
    """Test if a subreddit is accessible and has posts"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    url = f"https://www.reddit.com/r/{subreddit_name}/top.json"
    params = {'limit': 25, 't': 'all'}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'children' in data['data']:
                post_count = len(data['data']['children'])
                if post_count > 0:
                    print(f"✅ r/{subreddit_name:25s} - {post_count:2d} posts - WORKING")
                    return True
                else:
                    print(f"⚠️  r/{subreddit_name:25s} - 0 posts - EMPTY")
                    return False
        elif response.status_code == 403:
            print(f"🚫 r/{subreddit_name:25s} - FORBIDDEN (private/banned)")
            return False
        elif response.status_code == 404:
            print(f"❌ r/{subreddit_name:25s} - NOT FOUND (doesn't exist)")
            return False
        elif response.status_code == 429:
            print(f"⏳ r/{subreddit_name:25s} - RATE LIMITED (wait)")
            time.sleep(10)
            return test_subreddit(subreddit_name)  # Retry
        else:
            print(f"⚠️  r/{subreddit_name:25s} - HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ r/{subreddit_name:25s} - ERROR: {e}")
        return False

def main():
    print("="*70)
    print("REDDIT SUBREDDIT VERIFICATION TEST")
    print("="*70)
    print()

    working = []
    not_working = []

    for subreddit in TEST_SUBREDDITS:
        if test_subreddit(subreddit):
            working.append(subreddit)
        else:
            not_working.append(subreddit)
        time.sleep(1)  # Be nice to Reddit

    print()
    print("="*70)
    print(f"RESULTS: {len(working)}/{len(TEST_SUBREDDITS)} subreddits working")
    print("="*70)
    print()

    if working:
        print("✅ WORKING SUBREDDITS:")
        for sub in working:
            print(f"   • r/{sub}")

    if not_working:
        print()
        print("❌ NOT WORKING:")
        for sub in not_working:
            print(f"   • r/{sub}")

    print()
    print("💡 Use only the WORKING subreddits for downloads")
    print()

if __name__ == "__main__":
    main()
