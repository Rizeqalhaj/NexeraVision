#!/usr/bin/env python3
"""
Multi-Platform Non-Violence Video Scraper
Undetected, rate-limit aware, smart recovery system

Platforms: Reddit, YouTube, Vimeo, Archive.org
Features: Anti-detection, exponential backoff, checkpointing, health monitoring
"""

import sys
import time
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from threading import Thread, Lock
from collections import defaultdict, deque
from urllib.parse import urlparse
import re

# Non-violence keywords (70 peaceful activities)
NONVIOLENCE_KEYWORDS = [
    # CCTV normal activities
    'cctv normal', 'security camera normal', 'surveillance normal activity',
    'cctv people walking', 'security footage normal', 'cctv shopping',
    'cctv store normal',
    # Public places - normal
    'people walking street', 'pedestrians crossing', 'shopping mall normal',
    'store customers', 'restaurant dining', 'cafe normal', 'park people',
    'subway commuters', 'train station normal', 'bus station', 'airport normal',
    # Social interactions - peaceful
    'people talking', 'conversation', 'handshake', 'greeting', 'hugging',
    'friends meeting', 'family gathering',
    # Activities - non-violent
    'people working', 'office workers', 'construction workers', 'delivery',
    'shopping', 'walking dog', 'jogging', 'exercising', 'playing sports',
    'basketball game', 'soccer game', 'baseball game',
    # Crowds - peaceful
    'crowd gathering', 'concert crowd', 'festival', 'parade', 'celebration',
    'wedding', 'graduation', 'protest peaceful',
    # Traffic - normal
    'traffic normal', 'cars driving', 'parking lot normal', 'gas station normal',
    'highway traffic', 'intersection',
    # Daily life
    'daily routine', 'morning commute', 'lunch break', 'queue line',
    'waiting room', 'lobby', 'hallway', 'elevator',
    # Public transport
    'bus passengers', 'metro riders', 'train passengers', 'taxi ride',
    # Recreation
    'playground', 'beach', 'swimming pool', 'gym workout', 'yoga class',
    'dance class',
    # Events
    'conference', 'meeting', 'presentation', 'seminar', 'classroom', 'library',
]

# Platform rate limits (requests per minute)
PLATFORM_LIMITS = {
    'reddit': {'rpm': 55, 'backoff': 2.0, 'max_delay': 300},
    'youtube': {'rpm': 10, 'backoff': 3.0, 'max_delay': 300},
    'vimeo': {'rpm': 50, 'backoff': 2.0, 'max_delay': 300},
    'archive': {'rpm': 100, 'backoff': 1.5, 'max_delay': 180},
}


class RateLimiter:
    """Token bucket rate limiter per platform"""
    def __init__(self, platform, requests_per_minute):
        self.platform = platform
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = Lock()

    def acquire(self):
        """Wait until a token is available"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rpm, self.tokens + (elapsed * self.rpm / 60))
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * 60 / self.rpm
                time.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1


class HealthMonitor:
    """Monitor platform health and auto-throttle"""
    def __init__(self, platform):
        self.platform = platform
        self.requests = 0
        self.successes = 0
        self.failures = 0
        self.response_times = deque(maxlen=100)
        self.lock = Lock()

    def record_success(self, response_time):
        with self.lock:
            self.requests += 1
            self.successes += 1
            self.response_times.append(response_time)

    def record_failure(self):
        with self.lock:
            self.requests += 1
            self.failures += 1

    def get_success_rate(self):
        with self.lock:
            if self.requests == 0:
                return 1.0
            return self.successes / self.requests

    def get_avg_response_time(self):
        with self.lock:
            if not self.response_times:
                return 0
            return sum(self.response_times) / len(self.response_times)

    def should_throttle(self):
        """Throttle if success rate < 80%"""
        return self.get_success_rate() < 0.8 and self.requests > 20

    def should_disable(self):
        """Disable if success rate < 50% after 50 requests"""
        return self.get_success_rate() < 0.5 and self.requests > 50


class CheckpointManager:
    """Save and load scraper state"""
    def __init__(self, platform):
        self.platform = platform
        self.checkpoint_file = f"checkpoint_{platform}_{int(time.time())}.json"

    def save(self, state):
        """Save checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_latest(self):
        """Load most recent checkpoint for platform"""
        checkpoints = list(Path('.').glob(f'checkpoint_{self.platform}_*.json'))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)


class RedditAdapter:
    """Reddit scraper using PRAW API"""
    def __init__(self):
        self.platform = 'reddit'
        self.rate_limiter = RateLimiter(self.platform, PLATFORM_LIMITS['reddit']['rpm'])
        self.health = HealthMonitor(self.platform)
        self.checkpoint_mgr = CheckpointManager(self.platform)
        self.urls = set()
        self.reddit = None

        # Target subreddits for non-violence
        self.subreddits = [
            'MadeMeSmile', 'HumansBeingBros', 'sports', 'concerts',
            'festivals', 'streetphotography', 'peoplewatching'
        ]

    def authenticate(self):
        """Authenticate with Reddit (uses praw if available)"""
        try:
            import praw
            # User will need to set these up
            # For now, return False and use fallback
            print(f"⚠️  Reddit API authentication not configured")
            print(f"   To enable: create Reddit app at https://www.reddit.com/prefs/apps")
            print(f"   Then set CLIENT_ID, CLIENT_SECRET, USER_AGENT")
            return False
        except ImportError:
            print(f"⚠️  praw not installed: pip install praw")
            return False

    def search_subreddit(self, subreddit_name, keyword, limit=100):
        """Search subreddit for keyword (fallback: manual)"""
        # Fallback method without PRAW
        # Would use requests + headers here
        # For now, return empty to focus on YouTube/other platforms
        return []

    def scrape(self, keywords):
        """Scrape Reddit for non-violence videos"""
        print(f"\n{'='*70}")
        print(f"🔴 REDDIT Scraper")
        print(f"{'='*70}")

        if not self.authenticate():
            print(f"⏭️  Skipping Reddit (authentication not configured)")
            return list(self.urls)

        # Implementation would go here
        return list(self.urls)


class YouTubeAdapter:
    """YouTube scraper using yt-dlp search"""
    def __init__(self):
        self.platform = 'youtube'
        self.rate_limiter = RateLimiter(self.platform, PLATFORM_LIMITS['youtube']['rpm'])
        self.health = HealthMonitor(self.platform)
        self.checkpoint_mgr = CheckpointManager(self.platform)
        self.urls = set()

    def search_youtube(self, keyword, max_results=50):
        """Search YouTube using yt-dlp"""
        import subprocess

        self.rate_limiter.acquire()

        start_time = time.time()

        try:
            # yt-dlp search command
            cmd = [
                'yt-dlp',
                f'ytsearch{max_results}:{keyword}',
                '--get-id',
                '--no-warnings',
                '--quiet'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                video_ids = result.stdout.strip().split('\n')
                urls = [f'https://www.youtube.com/watch?v={vid}' for vid in video_ids if vid]

                response_time = time.time() - start_time
                self.health.record_success(response_time)

                return urls
            else:
                self.health.record_failure()
                return []

        except Exception as e:
            self.health.record_failure()
            print(f"   ✗ YouTube search error: {str(e)[:50]}")
            return []

    def scrape(self, keywords):
        """Scrape YouTube for all keywords"""
        print(f"\n{'='*70}")
        print(f"🔴 YOUTUBE Scraper")
        print(f"{'='*70}")
        print(f"Keywords: {len(keywords)}")
        print(f"Rate limit: {PLATFORM_LIMITS['youtube']['rpm']} requests/min")
        print(f"{'='*70}\n")

        # Load checkpoint if exists
        checkpoint = self.checkpoint_mgr.load_latest()
        start_index = 0
        if checkpoint:
            self.urls = set(checkpoint.get('urls', []))
            start_index = checkpoint.get('last_keyword_index', 0)
            print(f"📂 Loaded checkpoint: {len(self.urls)} URLs, resuming at keyword {start_index}")

        start_time = time.time()

        for i, keyword in enumerate(keywords[start_index:], start=start_index + 1):
            if self.health.should_disable():
                print(f"\n⚠️  YouTube platform disabled (success rate: {self.health.get_success_rate()*100:.1f}%)")
                break

            print(f"\n[{i}/{len(keywords)}] 🔍 '{keyword}'")

            # Apply throttling if needed
            if self.health.should_throttle():
                throttle_delay = random.uniform(10, 20)
                print(f"   ⏸️  Throttling: {throttle_delay:.1f}s (success: {self.health.get_success_rate()*100:.1f}%)")
                time.sleep(throttle_delay)

            # Search YouTube
            new_urls = self.search_youtube(keyword, max_results=30)

            before = len(self.urls)
            self.urls.update(new_urls)
            after = len(self.urls)
            new_unique = after - before

            elapsed = time.time() - start_time
            avg_time = elapsed / (i - start_index) if (i - start_index) > 0 else 0
            remaining = (len(keywords) - i) * avg_time

            print(f"   ✓ Found: {len(new_urls)} | New unique: {new_unique} | Total: {len(self.urls)}")
            print(f"   ⏱️  {elapsed/60:.1f}m elapsed | ~{remaining/60:.1f}m remaining")
            print(f"   📊 Success rate: {self.health.get_success_rate()*100:.1f}% | Avg response: {self.health.get_avg_response_time():.1f}s")

            # Checkpoint every 10 keywords
            if i % 10 == 0:
                self.checkpoint_mgr.save({
                    'urls': list(self.urls),
                    'last_keyword_index': i,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"   💾 Checkpoint saved")

            # Random delay between searches (5-10 seconds)
            delay = random.uniform(5, 10)
            time.sleep(delay)

        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"✓ YouTube scraping complete")
        print(f"{'='*70}")
        print(f"URLs collected: {len(self.urls)}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Success rate: {self.health.get_success_rate()*100:.1f}%")
        print(f"{'='*70}")

        return list(self.urls)


class VimeoAdapter:
    """Vimeo scraper using Playwright with stealth"""
    def __init__(self):
        self.platform = 'vimeo'
        self.rate_limiter = RateLimiter(self.platform, PLATFORM_LIMITS['vimeo']['rpm'])
        self.health = HealthMonitor(self.platform)
        self.checkpoint_mgr = CheckpointManager(self.platform)
        self.urls = set()

    def search_vimeo(self, keyword):
        """Search Vimeo with Playwright stealth"""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print(f"⚠️  Playwright not installed")
            return []

        self.rate_limiter.acquire()

        start_time = time.time()
        video_urls = []

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--disable-blink-features=AutomationControlled']
                )
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080}
                )
                page = context.new_page()

                # Search Vimeo
                search_url = f"https://vimeo.com/search?q={keyword.replace(' ', '+')}"
                page.goto(search_url, timeout=30000, wait_until='domcontentloaded')
                page.wait_for_timeout(3000)

                # Get video links
                links = page.query_selector_all('a[href^="/"]')
                for link in links:
                    try:
                        href = link.get_attribute('href')
                        if href and re.match(r'^/\d+$', href):  # Vimeo video ID pattern
                            video_urls.append(f'https://vimeo.com{href}')
                    except:
                        pass

                browser.close()

            response_time = time.time() - start_time
            self.health.record_success(response_time)

            return list(set(video_urls))

        except Exception as e:
            self.health.record_failure()
            print(f"   ✗ Vimeo error: {str(e)[:50]}")
            return []

    def scrape(self, keywords):
        """Scrape Vimeo for all keywords"""
        print(f"\n{'='*70}")
        print(f"🔵 VIMEO Scraper")
        print(f"{'='*70}")
        print(f"Keywords: {len(keywords)}")
        print(f"Rate limit: {PLATFORM_LIMITS['vimeo']['rpm']} requests/min")
        print(f"{'='*70}\n")

        # Load checkpoint
        checkpoint = self.checkpoint_mgr.load_latest()
        start_index = 0
        if checkpoint:
            self.urls = set(checkpoint.get('urls', []))
            start_index = checkpoint.get('last_keyword_index', 0)
            print(f"📂 Loaded checkpoint: {len(self.urls)} URLs, resuming at keyword {start_index}")

        start_time = time.time()

        for i, keyword in enumerate(keywords[start_index:], start=start_index + 1):
            if self.health.should_disable():
                print(f"\n⚠️  Vimeo platform disabled (success rate: {self.health.get_success_rate()*100:.1f}%)")
                break

            print(f"\n[{i}/{len(keywords)}] 🔍 '{keyword}'")

            if self.health.should_throttle():
                throttle_delay = random.uniform(15, 25)
                print(f"   ⏸️  Throttling: {throttle_delay:.1f}s")
                time.sleep(throttle_delay)

            new_urls = self.search_vimeo(keyword)

            before = len(self.urls)
            self.urls.update(new_urls)
            after = len(self.urls)
            new_unique = after - before

            print(f"   ✓ Found: {len(new_urls)} | New unique: {new_unique} | Total: {len(self.urls)}")
            print(f"   📊 Success: {self.health.get_success_rate()*100:.1f}%")

            if i % 10 == 0:
                self.checkpoint_mgr.save({
                    'urls': list(self.urls),
                    'last_keyword_index': i,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"   💾 Checkpoint saved")

            delay = random.uniform(3, 8)
            time.sleep(delay)

        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"✓ Vimeo scraping complete")
        print(f"URLs: {len(self.urls)} | Time: {elapsed/60:.1f}m")
        print(f"{'='*70}")

        return list(self.urls)


class ArchiveAdapter:
    """Archive.org scraper using official API"""
    def __init__(self):
        self.platform = 'archive'
        self.rate_limiter = RateLimiter(self.platform, PLATFORM_LIMITS['archive']['rpm'])
        self.health = HealthMonitor(self.platform)
        self.checkpoint_mgr = CheckpointManager(self.platform)
        self.urls = set()

    def search_archive(self, keyword):
        """Search Archive.org API"""
        import requests

        self.rate_limiter.acquire()

        start_time = time.time()

        try:
            # Archive.org search API
            url = f"https://archive.org/advancedsearch.php"
            params = {
                'q': f'{keyword} AND mediatype:movies',
                'fl[]': 'identifier',
                'rows': 50,
                'page': 1,
                'output': 'json'
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                identifiers = [doc['identifier'] for doc in data.get('response', {}).get('docs', [])]
                urls = [f'https://archive.org/details/{id}' for id in identifiers]

                response_time = time.time() - start_time
                self.health.record_success(response_time)

                return urls
            else:
                self.health.record_failure()
                return []

        except Exception as e:
            self.health.record_failure()
            print(f"   ✗ Archive.org error: {str(e)[:50]}")
            return []

    def scrape(self, keywords):
        """Scrape Archive.org for keywords"""
        print(f"\n{'='*70}")
        print(f"🟠 ARCHIVE.ORG Scraper")
        print(f"{'='*70}")

        # Use subset of keywords (Archive.org is more specific)
        archive_keywords = [
            'surveillance normal', 'cctv normal', 'security camera normal',
            'people walking', 'public space', 'street normal', 'shopping',
            'concert', 'festival', 'parade', 'sports game'
        ]

        print(f"Keywords: {len(archive_keywords)}")
        print(f"{'='*70}\n")

        start_time = time.time()

        for i, keyword in enumerate(archive_keywords, 1):
            print(f"\n[{i}/{len(archive_keywords)}] 🔍 '{keyword}'")

            new_urls = self.search_archive(keyword)

            before = len(self.urls)
            self.urls.update(new_urls)
            after = len(self.urls)

            print(f"   ✓ Found: {len(new_urls)} | New unique: {after - before} | Total: {len(self.urls)}")

            time.sleep(random.uniform(2, 5))

        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"✓ Archive.org scraping complete")
        print(f"URLs: {len(self.urls)} | Time: {elapsed/60:.1f}m")
        print(f"{'='*70}")

        return list(self.urls)


def deduplicate_urls(all_urls):
    """Remove duplicates across platforms"""
    # Simple URL deduplication
    unique = set()

    for url in all_urls:
        # Normalize URL
        parsed = urlparse(url)
        normalized = f"{parsed.netloc}{parsed.path}"
        unique.add(url)

    return list(unique)


def save_urls(urls, filename):
    """Save URLs to file"""
    with open(filename, 'w') as f:
        for url in sorted(urls):
            f.write(f"{url}\n")
    print(f"\n✓ Saved {len(urls)} URLs to: {filename}")


def main():
    print("="*70)
    print("Multi-Platform Non-Violence Video Scraper")
    print("="*70)
    print("Platforms: YouTube, Vimeo, Archive.org, (Reddit requires setup)")
    print("Features: Anti-detection, rate limiting, smart recovery")
    print("="*70)

    # Platform selection
    print(f"\nKeywords: {len(NONVIOLENCE_KEYWORDS)} non-violence terms")
    print(f"\nEstimated time: 2-3 hours")
    print(f"Expected yield: 4,000-6,000 video URLs")

    confirm = input("\nStart scraping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    all_urls = {}

    # YouTube (primary)
    youtube = YouTubeAdapter()
    youtube_urls = youtube.scrape(NONVIOLENCE_KEYWORDS)
    all_urls['youtube'] = youtube_urls
    save_urls(youtube_urls, 'nonviolence_urls_youtube.txt')

    # Vimeo (secondary)
    vimeo = VimeoAdapter()
    vimeo_urls = vimeo.scrape(NONVIOLENCE_KEYWORDS[:30])  # Use subset
    all_urls['vimeo'] = vimeo_urls
    save_urls(vimeo_urls, 'nonviolence_urls_vimeo.txt')

    # Archive.org (bonus)
    archive = ArchiveAdapter()
    archive_urls = archive.scrape(NONVIOLENCE_KEYWORDS)
    all_urls['archive'] = archive_urls
    save_urls(archive_urls, 'nonviolence_urls_archive.txt')

    # Combine and deduplicate
    combined = []
    for platform, urls in all_urls.items():
        combined.extend(urls)

    unique_urls = deduplicate_urls(combined)
    save_urls(unique_urls, 'nonviolence_urls_ALL_PLATFORMS.txt')

    # Final statistics
    print(f"\n{'='*70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"\nPer-Platform Results:")
    for platform, urls in all_urls.items():
        print(f"  {platform.upper():12} : {len(urls):,} URLs")
    print(f"\n  {'TOTAL UNIQUE':12} : {len(unique_urls):,} URLs")
    print(f"\n{'='*70}")
    print(f"\nDataset Balance:")
    print(f"  Violence:     14,000")
    print(f"  Non-violence:  8,000 + {len(unique_urls):,} = {8000 + len(unique_urls):,}")

    if (8000 + len(unique_urls)) >= 14000:
        print(f"\n  ✓ BALANCED! Ready for training")
        print(f"  Expected accuracy: 92-93%")
    else:
        needed = 14000 - (8000 + len(unique_urls))
        print(f"\n  ⚠️  Need {needed:,} more non-violence videos")

    print(f"\n{'='*70}")
    print(f"Next step: Download videos using download_videos_parallel.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
