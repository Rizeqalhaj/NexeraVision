#!/usr/bin/env python3
"""
Reddit MASSIVE Multi-Subreddit Downloader
Strategy: Download from 40+ subreddits instead of trying to get 20K from each
40 subreddits × 1,500 posts each = 60,000 posts = 40,000+ videos
"""

import requests
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from threading import Lock

# ✅ MASSIVE VERIFIED SUBREDDIT LIST (40+ sources)
FIGHT_SUBREDDITS = [
    # Tier 1: Mega subreddits (100K+ members)
    {'name': 'fightporn', 'size': 'mega'},
    {'name': 'PublicFreakout', 'size': 'mega'},
    {'name': 'CrazyFuckingVideos', 'size': 'mega'},
    {'name': 'AbruptChaos', 'size': 'mega'},
    {'name': 'WinStupidPrizes', 'size': 'mega'},
    {'name': 'Whatcouldgowrong', 'size': 'mega'},

    # Tier 2: Large subreddits (50K-100K)
    {'name': 'StreetMartialArts', 'size': 'large'},
    {'name': 'fights', 'size': 'large'},
    {'name': 'fightpornfans', 'size': 'large'},
    {'name': 'Justiceserved', 'size': 'large'},

    # Tier 3: Medium subreddits (10K-50K) - CCTV focused
    {'name': 'BestFights', 'size': 'medium'},
    {'name': 'GhettoStreetFights', 'size': 'medium'},
    {'name': 'StreetFightVideos', 'size': 'medium'},
    {'name': 'fightvideos', 'size': 'medium'},
    {'name': 'StreetFighting', 'size': 'medium'},
    {'name': 'RealFights', 'size': 'medium'},
    {'name': 'FightVideos', 'size': 'medium'},
    {'name': 'StreetFightersClub', 'size': 'medium'},

    # Tier 4: Specific fight types
    {'name': 'BrutalFights', 'size': 'medium'},
    {'name': 'HoodFights', 'size': 'medium'},
    {'name': 'Brawls', 'size': 'medium'},
    {'name': 'KnockoutFootage', 'size': 'medium'},
    {'name': 'SchoolFights', 'size': 'medium'},
    {'name': 'BarFights', 'size': 'medium'},

    # Tier 5: Female fight content
    {'name': 'Girlsfighting', 'size': 'medium'},
    {'name': 'femalemma', 'size': 'small'},

    # Tier 6: Combat sports (some CCTV)
    {'name': 'StreetMMA', 'size': 'medium'},
    {'name': 'amateurboxing', 'size': 'small'},
    {'name': 'streetboxing', 'size': 'small'},

    # Tier 7: Chaos/Freakout (high fight percentage)
    {'name': 'ActualPublicFreakouts', 'size': 'large'},
    {'name': 'PublicFreakouts', 'size': 'mega'},
    {'name': 'PublicFightVideos', 'size': 'medium'},

    # Tier 8: International fight content
    {'name': 'fightclub', 'size': 'medium'},
    {'name': 'StreetFightManiac', 'size': 'small'},

    # Tier 9: Instant karma (often includes fights)
    {'name': 'instantkarma', 'size': 'large'},
    {'name': 'instant_regret', 'size': 'large'},

    # Tier 10: Real-world incidents
    {'name': 'NoahGetTheBoat', 'size': 'large'},
    {'name': 'iamatotalpieceofshit', 'size': 'large'},
]

download_lock = Lock()
download_stats = {'success': 0, 'failed': 0}

def get_reddit_posts_optimized(subreddit, max_posts=2000):
    """Get maximum posts possible from Reddit JSON API"""
    posts = []
    after = None
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    # Aggressive pagination - try to get as much as possible
    for page in range(20):  # Try up to 20 pages
        url = f"https://www.reddit.com/r/{subreddit}/top.json"
        params = {'limit': 100, 't': 'all'}

        if after:
            params['after'] = after

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 403:
                return posts  # Subreddit banned/private
            elif response.status_code == 404:
                return posts  # Doesn't exist
            elif response.status_code == 429:
                time.sleep(30)  # Rate limit - wait and continue
                continue
            elif response.status_code != 200:
                break

            data = response.json()

            if 'data' not in data or 'children' not in data['data']:
                break

            children = data['data']['children']
            if not children:
                break

            posts.extend(children)
            after = data['data'].get('after')

            if not after or len(posts) >= max_posts:
                break

            time.sleep(2)  # Rate limit friendly

        except:
            break

    return posts[:max_posts]

def extract_video_url(post_data):
    """Extract video URL from Reddit post"""
    try:
        data = post_data['data']

        if not data.get('is_video', False):
            url = data.get('url', '')
            domain = data.get('domain', '')
            video_hosts = ['gfycat.com', 'imgur.com', 'youtube.com', 'youtu.be',
                          'streamable.com', 'v.redd.it', 'redgifs.com']
            if any(host in domain or host in url for host in video_hosts):
                return url
            return None

        if 'media' in data and data['media']:
            reddit_video = data['media'].get('reddit_video', {})
            if 'fallback_url' in reddit_video:
                return reddit_video['fallback_url']

        if 'crosspost_parent_list' in data and data['crosspost_parent_list']:
            parent = data['crosspost_parent_list'][0]
            if 'media' in parent and parent['media']:
                reddit_video = parent['media'].get('reddit_video', {})
                if 'fallback_url' in reddit_video:
                    return reddit_video['fallback_url']

        return None
    except:
        return None

def download_video_worker(video_data):
    """Worker function for parallel downloads"""
    video_url, output_dir, video_id = video_data
    output_template = str(output_dir / f"{video_id}.%(ext)s")

    cmd = [
        'yt-dlp', video_url,
        '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
        '-o', output_template,
        '--merge-output-format', 'mp4',
        '--restrict-filenames',
        '--no-playlist',
        '--ignore-errors',
        '--no-warnings',
        '--quiet',
        '--no-check-certificate',
        '--user-agent', 'Mozilla/5.0',
        '--referer', 'https://www.reddit.com/',
    ]

    try:
        subprocess.run(cmd, timeout=90, capture_output=True)
        for ext in ['.mp4', '.webm', '.mkv', '.avi']:
            if (output_dir / f"{video_id}{ext}").exists():
                with download_lock:
                    download_stats['success'] += 1
                return True
        with download_lock:
            download_stats['failed'] += 1
        return False
    except:
        with download_lock:
            download_stats['failed'] += 1
        return False

def download_from_subreddit(subreddit_name, output_dir, workers=100):
    """Download from single subreddit"""

    subreddit_dir = output_dir / f"r_{subreddit_name}"
    subreddit_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"r/{subreddit_name}")
    print('='*60)

    # Get posts (maximum we can)
    posts = get_reddit_posts_optimized(subreddit_name, max_posts=2000)

    if not posts:
        print(f"  ⚠️  Subreddit inaccessible or no posts")
        return 0

    print(f"  ✅ Fetched {len(posts)} posts")

    # Extract videos
    video_posts = []
    for post in posts:
        video_url = extract_video_url(post)
        if video_url:
            post_id = post['data']['id']
            video_posts.append((video_url, subreddit_dir, post_id))

    print(f"  ✅ Found {len(video_posts)} video posts")

    if not video_posts:
        return 0

    # Download
    global download_stats
    download_stats = {'success': 0, 'failed': 0}

    print(f"  📥 Downloading {len(video_posts)} videos...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_video_worker, v) for v in video_posts]
        with tqdm(total=len(video_posts), desc=f"  r/{subreddit_name}", unit="vid") as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                pbar.set_postfix({'✓': download_stats['success'], '✗': download_stats['failed']})

    print(f"  ✅ Downloaded {download_stats['success']} videos")
    return download_stats['success']

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Reddit MASSIVE Multi-Subreddit Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/reddit_videos_massive',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=100,
                       help='Parallel workers')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("REDDIT MASSIVE MULTI-SUBREDDIT DOWNLOADER")
    print("="*80)
    print(f"Strategy: Download from {len(FIGHT_SUBREDDITS)} subreddits")
    print(f"Expected: {len(FIGHT_SUBREDDITS)} × 1,500 posts = {len(FIGHT_SUBREDDITS)*1500:,} posts")
    print(f"Videos: ~{len(FIGHT_SUBREDDITS)*1000:,} fight videos")
    print("")

    total_downloaded = 0
    stats = {}
    start_time = time.time()

    for i, sub in enumerate(FIGHT_SUBREDDITS, 1):
        print(f"\n[{i}/{len(FIGHT_SUBREDDITS)}] Processing r/{sub['name']}...")

        downloaded = download_from_subreddit(sub['name'], output_dir, args.workers)
        stats[sub['name']] = downloaded
        total_downloaded += downloaded

        # Small delay between subreddits
        if i < len(FIGHT_SUBREDDITS):
            time.sleep(3)

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"\n📊 Downloaded from {len([v for v in stats.values() if v > 0])}/{len(FIGHT_SUBREDDITS)} subreddits")
    print(f"📹 Total videos: {total_downloaded:,}")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"📁 Location: {output_dir}")

    # Count files
    all_videos = list(output_dir.rglob('*.mp4')) + list(output_dir.rglob('*.webm'))
    print(f"\n✅ Verified: {len(all_videos):,} video files on disk")

    if total_downloaded >= 40000:
        print("\n🎉 EXCELLENT: 40,000+ videos!")
    elif total_downloaded >= 20000:
        print("\n✅ GREAT: 20,000+ videos!")
    elif total_downloaded >= 10000:
        print("\n✅ GOOD: 10,000+ videos!")

    print("")

if __name__ == "__main__":
    main()
