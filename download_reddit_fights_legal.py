#!/usr/bin/env python3
"""
Download fight videos from Reddit using official API (legal & ethical)
Uses PRAW (Python Reddit API Wrapper) - approved by Reddit
"""

import praw
import requests
import os
from pathlib import Path
import time
import json

print("="*80)
print("REDDIT FIGHT VIDEO DOWNLOADER - Legal API Method")
print("="*80)

# Setup
output_dir = Path("datasets/reddit_fights")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nVideos will be saved to: {output_dir.absolute()}")

print("\n⚠️  SETUP REQUIRED:")
print("1. Create Reddit app: https://www.reddit.com/prefs/apps")
print("2. Click 'create app' or 'create another app'")
print("3. Choose 'script' type")
print("4. Copy client_id and client_secret")
print("5. Update this script with your credentials")

# Reddit API credentials (REPLACE WITH YOUR OWN)
REDDIT_CONFIG = {
    'client_id': 'YOUR_CLIENT_ID',  # Replace with your client_id
    'client_secret': 'YOUR_CLIENT_SECRET',  # Replace with your secret
    'user_agent': 'violence_detection_research v1.0'
}

print("\n" + "="*80)
print("CONFIGURATION")
print("="*80)

if REDDIT_CONFIG['client_id'] == 'YOUR_CLIENT_ID':
    print("\n❌ You need to configure Reddit API credentials!")
    print("\nSteps:")
    print("1. Go to: https://www.reddit.com/prefs/apps")
    print("2. Click 'create another app'")
    print("3. Fill in:")
    print("   - name: violence_detection_research")
    print("   - type: script")
    print("   - redirect uri: http://localhost:8080")
    print("4. Copy 'client_id' (under 'personal use script')")
    print("5. Copy 'secret'")
    print("6. Update REDDIT_CONFIG in this script")
    print("\n" + "="*80)
    exit(1)

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CONFIG['client_id'],
    client_secret=REDDIT_CONFIG['client_secret'],
    user_agent=REDDIT_CONFIG['user_agent']
)

print(f"✅ Connected to Reddit API")
print(f"   Rate limit: 60 requests/minute (Reddit's limit)")

# Subreddits with fight content (public, allowed)
SUBREDDITS = [
    'fightporn',      # 1.5M members - street fights, sports
    'streetfights',   # 200K members - public altercations
    'publicfreakout', # 4M members - includes some fights
    'actualpublicfreakouts',  # Backup subreddit
]

print(f"\nTargeting {len(SUBREDDITS)} subreddits:")
for sub in SUBREDDITS:
    print(f"  - r/{sub}")

# Search terms for CCTV/surveillance footage
SEARCH_TERMS = [
    'cctv',
    'security camera',
    'surveillance',
    'camera caught',
    'caught on camera',
]

print(f"\nSearch terms for CCTV footage:")
for term in SEARCH_TERMS:
    print(f"  - '{term}'")

def download_video(url, filepath):
    """Download video from URL"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"    Error downloading: {e}")
    return False

def scrape_subreddit(subreddit_name, search_term, max_videos=100):
    """Scrape videos from a subreddit"""

    print(f"\n{'='*80}")
    print(f"Searching r/{subreddit_name} for '{search_term}'")
    print(f"{'='*80}")

    subreddit = reddit.subreddit(subreddit_name)
    downloaded = 0

    # Search for posts
    for submission in subreddit.search(search_term, limit=max_videos, time_filter='year'):

        # Check if it's a video post
        if hasattr(submission, 'is_video') and submission.is_video:
            video_url = submission.media['reddit_video']['fallback_url']

            # Create filename
            safe_title = "".join(c for c in submission.title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            filename = f"{subreddit_name}_{submission.id}_{safe_title}.mp4"
            filepath = output_dir / filename

            # Skip if already downloaded
            if filepath.exists():
                print(f"  ⏭️  Skip: {filename} (already exists)")
                continue

            print(f"  📥 Downloading: {filename}")

            if download_video(video_url, filepath):
                downloaded += 1
                print(f"    ✅ Success ({downloaded} total)")

                # Save metadata
                metadata = {
                    'id': submission.id,
                    'title': submission.title,
                    'subreddit': subreddit_name,
                    'url': submission.url,
                    'score': submission.score,
                    'created_utc': submission.created_utc,
                    'search_term': search_term
                }

                metadata_path = filepath.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            # Respect rate limits
            time.sleep(1)

            if downloaded >= max_videos:
                break

    print(f"\n✅ Downloaded {downloaded} videos from r/{subreddit_name}")
    return downloaded

# Main scraping loop
print("\n" + "="*80)
print("STARTING DOWNLOAD")
print("="*80)

total_downloaded = 0

for subreddit in SUBREDDITS:
    for search_term in SEARCH_TERMS:
        try:
            count = scrape_subreddit(subreddit, search_term, max_videos=50)
            total_downloaded += count

            # Respect rate limits
            time.sleep(2)

        except Exception as e:
            print(f"Error with r/{subreddit} - {search_term}: {e}")
            continue

# Summary
print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)

print(f"\n✅ Total videos downloaded: {total_downloaded}")
print(f"✅ Saved to: {output_dir.absolute()}")

print("\n📋 Next steps:")
print("1. Review downloaded videos")
print("2. Filter for CCTV/surveillance footage only")
print("3. Label as violent/non-violent")
print("4. Add to training dataset")
print("5. Retrain model")

print("\n" + "="*80)
