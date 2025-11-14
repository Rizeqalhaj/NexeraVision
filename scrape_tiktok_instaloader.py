#!/usr/bin/env python3
"""
TikTok Scraper using Alternative Method - Direct API Calls
Bypasses browser detection by using direct HTTP requests
"""

import requests
import json
import time
import random
from pathlib import Path
from datetime import datetime

class TikTokDirectScraper:
    def __init__(self, output_file="tiktok_direct_videos.json"):
        self.output_file = Path(output_file)
        self.scraped_videos = set()
        self.video_data = []

        # Load existing data
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing = json.load(f)
                self.video_data = existing
                self.scraped_videos = {item['video_id'] for item in existing}

        # TikTok hashtags
        self.hashtags = [
            "cctv", "cctvfootage", "securitycamera", "securityfootage",
            "surveillance", "surveillancecamera", "caughtoncamera",
            "caughtoncctv", "cctvfight", "securitycamerafight",
            "fight", "streetfight", "fightvideos", "knockout", "brawl",
            "fighting", "realfight", "violentfight", "publicfight"
        ]

        # User agents to rotate
        self.user_agents = [
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 TikTok/21.1.0',
            'Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36 TikTok/21.2.0',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
        ]

    def get_headers(self):
        """Get randomized headers"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.tiktok.com/',
            'Origin': 'https://www.tiktok.com',
        }

    def scrape_hashtag_direct(self, hashtag, max_videos=300):
        """Try to scrape using TikTok's unofficial API endpoints"""
        print(f"\n{'='*80}")
        print(f"📍 Hashtag: #{hashtag}")
        print(f"🎯 Target: {max_videos} videos")
        print(f"{'='*80}\n")

        collected = 0

        # Try multiple TikTok API endpoints
        api_urls = [
            f"https://www.tiktok.com/api/challenge/detail/?challengeName={hashtag}",
            f"https://m.tiktok.com/api/challenge/item_list/?challengeID={hashtag}&count=30",
        ]

        for api_url in api_urls:
            try:
                print(f"🔍 Trying API: {api_url[:60]}...")

                response = requests.get(
                    api_url,
                    headers=self.get_headers(),
                    timeout=15
                )

                if response.status_code == 200:
                    print(f"✅ Got response (status 200)")

                    try:
                        data = response.json()
                        print(f"📊 Response keys: {list(data.keys())}")

                        # Try to extract video data from response
                        # Structure varies by endpoint
                        videos_found = self.extract_videos_from_response(data, hashtag)
                        collected += videos_found

                        if videos_found > 0:
                            print(f"✅ Found {videos_found} videos from this API")
                            break
                    except json.JSONDecodeError:
                        print(f"⚠️  Response is not JSON")
                else:
                    print(f"⚠️  Status {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {str(e)[:100]}")
                continue

            time.sleep(2)  # Rate limit

        return collected

    def extract_videos_from_response(self, data, hashtag):
        """Extract video URLs from API response"""
        count = 0

        # Try different response structures
        possible_paths = [
            ['itemList'],
            ['challengeInfo', 'stats', 'videoCount'],
            ['body', 'itemList'],
            ['items'],
        ]

        for path in possible_paths:
            try:
                items = data
                for key in path:
                    if isinstance(items, dict):
                        items = items.get(key, [])
                    else:
                        break

                if isinstance(items, list) and len(items) > 0:
                    print(f"📹 Found {len(items)} items in path {path}")

                    for item in items:
                        video_id = item.get('id', item.get('videoId', 'unknown'))

                        if video_id and video_id not in self.scraped_videos:
                            self.scraped_videos.add(video_id)
                            self.video_data.append({
                                'video_id': video_id,
                                'url': f"https://www.tiktok.com/@user/video/{video_id}",
                                'hashtag': hashtag,
                                'source': 'tiktok_direct_api',
                                'scraped_at': datetime.now().isoformat(),
                            })
                            count += 1

                    if count > 0:
                        return count
            except Exception as e:
                continue

        return count

    def save_progress(self):
        """Save scraped data"""
        with open(self.output_file, 'w') as f:
            json.dump(self.video_data, f, indent=2)
        print(f"💾 Saved {len(self.video_data)} total videos")

    def scrape_all_hashtags(self, max_per_hashtag=300):
        """Scrape all hashtags"""
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              TIKTOK DIRECT API SCRAPER                                     ║")
        print("║              Uses HTTP requests instead of browser                         ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total hashtags: {len(self.hashtags)}")
        print(f"🎯 Max per hashtag: {max_per_hashtag}")
        print(f"📁 Output: {self.output_file}")
        print(f"✅ Already have: {len(self.scraped_videos)} videos")
        print()

        for idx, hashtag in enumerate(self.hashtags, 1):
            print(f"\n📍 Progress: {idx}/{len(self.hashtags)} hashtags")

            try:
                self.scrape_hashtag_direct(hashtag, max_per_hashtag)
                self.save_progress()

                # Delay between hashtags
                if idx < len(self.hashtags):
                    delay = random.randint(5, 10)
                    print(f"\n💤 Resting {delay}s...")
                    time.sleep(delay)

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error on hashtag '{hashtag}': {str(e)[:100]}")
                continue

        print("\n" + "="*80)
        print("📊 SCRAPING COMPLETE")
        print("="*80)
        print(f"✅ Total unique videos: {len(self.video_data)}")
        print(f"📁 Saved to: {self.output_file}")
        print("="*80)


def main():
    scraper = TikTokDirectScraper(output_file="tiktok_direct_videos.json")
    scraper.scrape_all_hashtags(max_per_hashtag=300)


if __name__ == "__main__":
    main()
