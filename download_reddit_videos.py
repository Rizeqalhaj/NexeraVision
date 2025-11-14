#!/usr/bin/env python3
"""
Reddit Video Downloader
Downloads videos from Reddit subreddits using Reddit JSON API
Works WITHOUT Reddit API credentials - uses public JSON endpoints
"""

import requests
import subprocess
import time
import json
from pathlib import Path
from tqdm import tqdm
import random

# Best subreddits for CCTV fight footage
FIGHT_SUBREDDITS = [
    {'name': 'fightporn', 'cctv_percent': 70},
    {'name': 'DocumentedFights', 'cctv_percent': 80},
    {'name': 'StreetFights', 'cctv_percent': 60},
    {'name': 'ActualFreakouts', 'cctv_percent': 50},
    {'name': 'PublicFreakout', 'cctv_percent': 40},
    {'name': 'CrazyFuckingVideos', 'cctv_percent': 55},
    {'name': 'AbruptChaos', 'cctv_percent': 45},
]

# Normal surveillance subreddits
NORMAL_SUBREDDITS = [
    {'name': 'IdiotsInCars', 'cctv_percent': 70},
    {'name': 'CCTV', 'cctv_percent': 90},
    {'name': 'SecurityCameras', 'cctv_percent': 85},
    {'name': 'homedefense', 'cctv_percent': 60},
]

def get_reddit_posts(subreddit, sort='top', time_filter='all', limit=1000):
    """
    Get posts from Reddit using public JSON API (no auth needed)

    Args:
        subreddit: Subreddit name (without r/)
        sort: 'top', 'hot', 'new'
        time_filter: 'all', 'year', 'month', 'week', 'day'
        limit: Max posts to retrieve
    """
    posts = []
    after = None

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"  Fetching posts from r/{subreddit} ({sort}/{time_filter})...")

    # Reddit API returns 100 posts max per request
    while len(posts) < limit:
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params = {
            'limit': min(100, limit - len(posts)),
            't': time_filter
        }

        if after:
            params['after'] = after

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code != 200:
                print(f"    ⚠️  HTTP {response.status_code} - stopping")
                break

            data = response.json()

            if 'data' not in data or 'children' not in data['data']:
                print(f"    ⚠️  Invalid response format")
                break

            children = data['data']['children']

            if not children:
                print(f"    ℹ️  No more posts available")
                break

            posts.extend(children)
            after = data['data'].get('after')

            print(f"    Fetched {len(posts)} posts...", end='\r')

            if not after:
                break

            # Rate limiting - be nice to Reddit
            time.sleep(2)

        except Exception as e:
            print(f"    ❌ Error: {e}")
            break

    print(f"    ✅ Fetched {len(posts)} posts total")
    return posts

def extract_video_url(post_data):
    """Extract video URL from Reddit post"""
    try:
        data = post_data['data']

        # Skip non-video posts
        if not data.get('is_video', False):
            # Check for external video hosts
            url = data.get('url', '')
            domain = data.get('domain', '')

            # Check for common video hosts
            video_hosts = ['gfycat.com', 'imgur.com', 'youtube.com', 'youtu.be',
                          'streamable.com', 'v.redd.it', 'redgifs.com']

            if any(host in domain or host in url for host in video_hosts):
                return url

            return None

        # Reddit hosted video
        if 'media' in data and data['media']:
            reddit_video = data['media'].get('reddit_video', {})
            if 'fallback_url' in reddit_video:
                return reddit_video['fallback_url']

        # Crosspost video
        if 'crosspost_parent_list' in data and data['crosspost_parent_list']:
            parent = data['crosspost_parent_list'][0]
            if 'media' in parent and parent['media']:
                reddit_video = parent['media'].get('reddit_video', {})
                if 'fallback_url' in reddit_video:
                    return reddit_video['fallback_url']

        return None

    except Exception as e:
        return None

def download_video(video_url, output_dir, video_id):
    """Download video using yt-dlp"""
    output_template = str(output_dir / f"{video_id}.%(ext)s")

    cmd = [
        'yt-dlp',
        video_url,
        '-f', 'best[height<=720]',
        '-o', output_template,
        '--restrict-filenames',
        '--no-playlist',
        '--ignore-errors',
        '--no-warnings',
        '--quiet',
        '--no-check-certificate'
    ]

    try:
        result = subprocess.run(cmd, timeout=120, capture_output=True)

        # Check if file was created
        for ext in ['.mp4', '.webm', '.mkv', '.avi']:
            if (output_dir / f"{video_id}{ext}").exists():
                return True

        return False

    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False

def download_from_subreddit(subreddit_name, output_dir, max_videos=1000, sort='top', time_filter='all'):
    """Download videos from a subreddit"""

    subreddit_dir = output_dir / f"r_{subreddit_name}"
    subreddit_dir.mkdir(parents=True, exist_ok=True)

    # Get posts
    posts = get_reddit_posts(subreddit_name, sort, time_filter, limit=max_videos * 3)  # Get 3x to account for non-videos

    # Extract video URLs
    video_posts = []
    for post in posts:
        video_url = extract_video_url(post)
        if video_url:
            post_id = post['data']['id']
            post_title = post['data']['title']
            video_posts.append({
                'id': post_id,
                'url': video_url,
                'title': post_title
            })

    print(f"  Found {len(video_posts)} video posts out of {len(posts)} total posts")

    if not video_posts:
        print(f"  ⚠️  No videos found in r/{subreddit_name}")
        return 0

    # Limit to max_videos
    video_posts = video_posts[:max_videos]

    # Download videos
    print(f"  Downloading {len(video_posts)} videos...")
    downloaded = 0

    for video_post in tqdm(video_posts, desc=f"  r/{subreddit_name}"):
        success = download_video(video_post['url'], subreddit_dir, video_post['id'])
        if success:
            downloaded += 1

        # Rate limiting
        time.sleep(1)

    print(f"  ✅ Downloaded {downloaded} videos from r/{subreddit_name}")
    return downloaded

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Reddit Video Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/reddit_videos',
                       help='Output directory')
    parser.add_argument('--category', choices=['fight', 'normal', 'all'], default='fight',
                       help='Category: fight (violent), normal (non-violent), or all')
    parser.add_argument('--max-per-subreddit', type=int, default=1000,
                       help='Max videos per subreddit')
    parser.add_argument('--sort', choices=['top', 'hot', 'new'], default='top',
                       help='Reddit sorting method')
    parser.add_argument('--time', choices=['all', 'year', 'month', 'week', 'day'], default='all',
                       help='Time filter for top posts')
    parser.add_argument('--subreddits', nargs='+',
                       help='Specific subreddits to download (overrides category)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("REDDIT VIDEO DOWNLOADER")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Category: {args.category}")
    print(f"Max per subreddit: {args.max_per_subreddit}")
    print(f"Sort: {args.sort}/{args.time}")
    print("")

    # Determine which subreddits to download from
    if args.subreddits:
        subreddits = [{'name': s, 'cctv_percent': 50} for s in args.subreddits]
    elif args.category == 'fight':
        subreddits = FIGHT_SUBREDDITS
    elif args.category == 'normal':
        subreddits = NORMAL_SUBREDDITS
    else:  # all
        subreddits = FIGHT_SUBREDDITS + NORMAL_SUBREDDITS

    print("📋 SUBREDDITS TO DOWNLOAD:")
    for sub in subreddits:
        print(f"  • r/{sub['name']} (est. {sub['cctv_percent']}% CCTV)")
    print("")

    # Download from each subreddit
    total_downloaded = 0
    stats = {}

    for sub in subreddits:
        print(f"\n{'='*60}")
        print(f"r/{sub['name']}")
        print('='*60)

        downloaded = download_from_subreddit(
            sub['name'],
            output_dir,
            args.max_per_subreddit,
            args.sort,
            args.time
        )

        stats[sub['name']] = downloaded
        total_downloaded += downloaded

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print("")
    print("📊 STATISTICS BY SUBREDDIT:")
    print("-"*60)

    for sub_name, count in stats.items():
        print(f"  r/{sub_name:25s}: {count:6d} videos")

    print("-"*60)
    print(f"  {'TOTAL':25s}: {total_downloaded:6d} videos")
    print("")

    print(f"📁 Location: {output_dir}")
    print("")

    # Count actual files
    all_videos = list(output_dir.rglob('*.mp4')) + list(output_dir.rglob('*.webm')) + \
                 list(output_dir.rglob('*.mkv')) + list(output_dir.rglob('*.avi'))

    print(f"✅ Verified on disk: {len(all_videos)} video files")
    print("")

    if total_downloaded >= 5000:
        print("🎉 EXCELLENT: 5,000+ Reddit videos!")
    elif total_downloaded >= 1000:
        print("✅ GOOD: 1,000+ Reddit videos")
    else:
        print(f"⚠️  Downloaded {total_downloaded} videos")

    print("")
    print("🔄 NEXT STEPS:")
    print("1. Validate dataset quality:")
    print(f"   python3 validate_violent_videos.py --dataset-dir {output_dir} --sample-size 100")
    print("")
    print("2. Clean suspicious videos:")
    print("   python3 clean_dataset.py --validation-report validation_results/validation_report.json")
    print("")
    print("3. Combine with other datasets:")
    print("   bash balance_and_combine.sh")
    print("")

if __name__ == "__main__":
    main()
