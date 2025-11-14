#!/usr/bin/env python3
"""
TikTok Video Collection using TikTokApi
Uses TikTokApi library - NO LOGIN REQUIRED!
Works by using TikTok's internal API endpoints
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from TikTokApi import TikTokApi

class TikTokApiCollector:
    def __init__(self, output_file="tiktok_videos_api.json"):
        self.output_file = Path(output_file)
        self.scraped_videos = set()
        self.video_data = []

        # Load existing data if file exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing = json.load(f)
                self.video_data = existing
                self.scraped_videos = {item['video_id'] for item in existing}

        # TikTok hashtags - CCTV focused
        self.hashtags = [
            "cctv",
            "cctvfootage",
            "securitycamera",
            "securityfootage",
            "surveillance",
            "surveillancecamera",
            "caughtoncamera",
            "caughtoncctv",
            "cctvfight",
            "securitycamerafight",
            "fight",
            "streetfight",
            "fightvideos",
            "fightcaughtoncamera",
            "knockout",
            "brawl",
            "fighting",
            "realfight",
            "violentfight",
            "brutalfight",
            "publicfight",
            "parkingfight",
            "gasstationfight",
            "storefight",
            "barfight",
        ]

    async def save_progress(self):
        """Save scraped data to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.video_data, f, indent=2)
        print(f"💾 Saved {len(self.video_data)} total videos")

    async def collect_from_hashtag(self, api, hashtag, max_videos=300):
        """Collect videos from a single TikTok hashtag"""

        print(f"\n{'='*80}")
        print(f"📍 Hashtag: #{hashtag}")
        print(f"🎯 Target: {max_videos} videos")
        print(f"{'='*80}\n")

        hashtag_new_count = 0

        try:
            tag = api.hashtag(name=hashtag)

            print(f"Collecting videos from #{hashtag}...")

            async for video in tag.videos(count=max_videos):
                try:
                    video_id = video.id

                    # Skip if already collected
                    if video_id in self.scraped_videos:
                        continue

                    # Extract video info
                    video_url = f"https://www.tiktok.com/@{video.author.username}/video/{video_id}"

                    video_info = {
                        'video_id': video_id,
                        'url': video_url,
                        'description': video.desc if hasattr(video, 'desc') else 'Unknown',
                        'author': video.author.username if hasattr(video.author, 'username') else 'unknown',
                        'hashtag': hashtag,
                        'stats': {
                            'views': video.stats['playCount'] if hasattr(video, 'stats') else 0,
                            'likes': video.stats['diggCount'] if hasattr(video, 'stats') else 0,
                            'comments': video.stats['commentCount'] if hasattr(video, 'stats') else 0,
                            'shares': video.stats['shareCount'] if hasattr(video, 'stats') else 0,
                        },
                        'source': 'tiktok_api',
                        'scraped_at': datetime.now().isoformat(),
                    }

                    self.scraped_videos.add(video_id)
                    self.video_data.append(video_info)
                    hashtag_new_count += 1

                    print(f"📹 +{hashtag_new_count} videos | Total: {len(self.video_data)}", end='\r')

                    if hashtag_new_count >= max_videos:
                        break

                except Exception as e:
                    print(f"\n⚠️  Error processing video: {str(e)[:100]}")
                    continue

            print(f"\n✅ Hashtag complete: Found {hashtag_new_count} videos from #{hashtag}")
            return hashtag_new_count

        except Exception as e:
            print(f"\n❌ Error on hashtag #{hashtag}: {str(e)[:200]}")
            return 0

    async def collect_all_hashtags(self, max_per_hashtag=300):
        """Collect videos from all hashtags"""

        start_time = datetime.now()

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              TIKTOK VIDEO COLLECTOR (API)                                  ║")
        print("║              NO LOGIN REQUIRED - Uses TikTok Internal API                  ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total hashtags: {len(self.hashtags)}")
        print(f"🎯 Max per hashtag: {max_per_hashtag}")
        print(f"📊 Expected total: {len(self.hashtags) * max_per_hashtag} videos")
        print(f"📁 Output: {self.output_file}")
        print(f"✅ Already have: {len(self.scraped_videos)} videos")
        print()
        print("⚙️  Initializing TikTok API (this may take a moment)...")
        print()

        async with TikTokApi() as api:
            # Create sessions (required for TikTokApi)
            await api.create_sessions(num_sessions=1, sleep_after=3)

            print("✅ TikTok API ready!")
            print()

            for idx, hashtag in enumerate(self.hashtags, 1):
                print(f"\n📍 Progress: {idx}/{len(self.hashtags)} hashtags")

                try:
                    await self.collect_from_hashtag(api, hashtag, max_per_hashtag)

                    # Save after each hashtag
                    await self.save_progress()

                    # Delay between hashtags to avoid rate limiting
                    if idx < len(self.hashtags):
                        delay = 30  # 30 seconds between hashtags
                        print(f"\n💤 Resting {delay}s before next hashtag...")
                        await asyncio.sleep(delay)

                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted by user")
                    break
                except Exception as e:
                    print(f"\n❌ Error on hashtag '{hashtag}': {str(e)[:100]}")
                    print("   Saving progress and continuing...")
                    await self.save_progress()
                    continue

        # Final statistics
        runtime = (datetime.now() - start_time).total_seconds() / 60

        print("\n" + "="*80)
        print("📊 COLLECTION COMPLETE")
        print("="*80)
        print(f"✅ Total unique videos: {len(self.video_data)}")
        print(f"🔍 Hashtags completed: {len(self.hashtags)}")
        print(f"⏱️  Total runtime: {runtime:.1f} minutes ({runtime/60:.1f} hours)")
        print(f"📁 Saved to: {self.output_file}")
        print("="*80)
        print()
        print("Next step: Download videos using:")
        print(f"  python3 download_tiktok_api_videos.py {self.output_file}")
        print("="*80)


async def main():
    import sys

    # Configuration
    OUTPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "tiktok_videos_api.json"
    MAX_PER_HASHTAG = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print()
    print("Usage:")
    print(f"  python3 {sys.argv[0]} [output_file] [max_per_hashtag]")
    print()
    print("Example:")
    print(f"  python3 {sys.argv[0]} tiktok_videos_api.json 300")
    print()
    print("Current settings:")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Max per hashtag: {MAX_PER_HASHTAG}")
    print()

    # Run collector
    collector = TikTokApiCollector(output_file=OUTPUT_FILE)
    await collector.collect_all_hashtags(max_per_hashtag=MAX_PER_HASHTAG)


if __name__ == "__main__":
    asyncio.run(main())
