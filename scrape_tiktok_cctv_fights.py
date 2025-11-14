#!/usr/bin/env python3
"""
TikTok CCTV Fight Video Scraper
Scrapes CCTV fight videos from TikTok hashtags using infinite scroll
"""

import asyncio
import random
import json
import re
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

class TikTokCCTVScraper:
    def __init__(self, output_file="tiktok_cctv_fights.json"):
        self.output_file = Path(output_file)
        self.scraped_links = set()
        self.video_data = []

        # Load existing data if file exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing = json.load(f)
                self.video_data = existing
                self.scraped_links = {item['url'] for item in existing}

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

    async def random_delay(self, min_sec, max_sec):
        await asyncio.sleep(random.uniform(min_sec, max_sec))

    async def extract_tiktok_videos(self, page):
        """Extract TikTok video links from current page"""
        videos = await page.evaluate("""
            () => {
                const results = [];
                const seen = new Set();

                // Method 1: Find video containers
                const videoContainers = document.querySelectorAll('div[data-e2e="user-post-item"]');

                videoContainers.forEach(container => {
                    // Try to find video link
                    const linkElem = container.querySelector('a[href*="/video/"]');
                    if (linkElem) {
                        const url = linkElem.href;

                        if (!seen.has(url)) {
                            seen.add(url);

                            // Try to extract description
                            let description = 'Unknown';
                            const descElem = container.querySelector('div[data-e2e="video-desc"]');
                            if (descElem) {
                                description = descElem.textContent.trim();
                            }

                            // Extract video ID from URL
                            const match = url.match(/\\/video\\/([0-9]+)/);
                            const videoId = match ? match[1] : 'unknown';

                            results.push({
                                url: url,
                                description: description,
                                videoId: videoId
                            });
                        }
                    }
                });

                // Method 2: Direct video links
                const videoLinks = document.querySelectorAll('a[href*="/video/"]');
                videoLinks.forEach(link => {
                    const url = link.href;

                    if (!seen.has(url) && url.includes('/video/')) {
                        seen.add(url);

                        const match = url.match(/\\/video\\/([0-9]+)/);
                        const videoId = match ? match[1] : 'unknown';

                        results.push({
                            url: url,
                            description: 'Unknown',
                            videoId: videoId
                        });
                    }
                });

                return results;
            }
        """)

        return videos

    async def save_progress(self):
        """Save scraped data to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.video_data, f, indent=2)
        print(f"💾 Saved {len(self.video_data)} total videos")

    async def scrape_single_hashtag(self, browser, hashtag, hashtag_num, total_hashtags):
        """Scrape TikTok videos for a single hashtag"""
        print(f"\n{'='*80}")
        print(f"📍 Hashtag {hashtag_num}/{total_hashtags}: #{hashtag}")
        print(f"{'='*80}\n")

        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
        )

        # Anti-detection
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
            window.navigator.chrome = { runtime: {} };
        """)

        page = await context.new_page()

        # TikTok hashtag URL
        url = f"https://www.tiktok.com/tag/{hashtag}"

        print(f"🌐 Loading: {url}")
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
        except Exception as e:
            print(f"   ⚠️ Timeout loading page: {str(e)[:100]}")
            await context.close()
            return 0

        await self.random_delay(5, 8)

        # Handle any popups/dialogs
        try:
            # Close login popup if it appears
            close_buttons = await page.query_selector_all('button[aria-label*="Close"], button[data-e2e="modal-close-inner-button"]')
            for btn in close_buttons:
                await btn.click()
                await self.random_delay(1, 2)
                break
        except:
            pass

        consecutive_no_new = 0
        hashtag_new_count = 0

        print("Scrolling and collecting videos...")

        # Dynamic patience
        while hashtag_new_count < 500:
            max_no_new = 5 if hashtag_new_count >= 100 else 20

            if consecutive_no_new >= max_no_new:
                break

            # Extract videos
            new_videos = await self.extract_tiktok_videos(page)

            new_count = 0
            for video_data in new_videos:
                if video_data['url'] not in self.scraped_links:
                    self.scraped_links.add(video_data['url'])
                    self.video_data.append({
                        'url': video_data['url'],
                        'description': video_data['description'],
                        'videoId': video_data['videoId'],
                        'hashtag': hashtag,
                        'source': 'tiktok',
                        'scraped_at': datetime.now().isoformat(),
                    })
                    new_count += 1
                    hashtag_new_count += 1

            if new_count > 0:
                consecutive_no_new = 0
                print(f"📹 +{new_count} videos | Hashtag total: {hashtag_new_count} | Overall: {len(self.video_data)}")
            else:
                consecutive_no_new += 1
                print(f"⏳ Scrolling... ({consecutive_no_new}/{max_no_new}) | Got: {hashtag_new_count}")

            # TikTok infinite scroll - multiple small scrolls
            for _ in range(5):
                await page.evaluate("window.scrollBy(0, 800)")
                await asyncio.sleep(0.3)

            # Wait for content to load
            await asyncio.sleep(3)

            # Early exit if we have enough
            if hashtag_new_count >= 300 and consecutive_no_new >= 5:
                print(f"   ✅ Got {hashtag_new_count} videos, moving on")
                break

        print(f"\n✅ Hashtag complete: Found {hashtag_new_count} new videos from #{hashtag}")

        await context.close()
        return hashtag_new_count

    async def scrape_all_hashtags(self):
        """Scrape all hashtags"""
        start_time = datetime.now()

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              TIKTOK CCTV FIGHT VIDEO SCRAPER                               ║")
        print("║              Hashtag-based collection with CCTV focus                      ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total hashtags: {len(self.hashtags)}")
        print(f"📊 Focus: #cctv #securitycamera #surveillance")
        print(f"📊 Expected: ~200-300 videos per hashtag")
        print(f"📊 Total expected: 5,000-8,000 unique videos")
        print(f"📁 Output: {self.output_file}")
        print(f"✅ Already have: {len(self.scraped_links)} videos")
        print()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                ]
            )

            try:
                for idx, hashtag in enumerate(self.hashtags, 1):
                    try:
                        await self.scrape_single_hashtag(browser, hashtag, idx, len(self.hashtags))

                        # Save after each hashtag
                        await self.save_progress()

                        # Longer delay between hashtags to avoid rate limits
                        if idx < len(self.hashtags):
                            delay = random.randint(45, 75)
                            print(f"\n💤 Resting {delay}s before next hashtag...")
                            await asyncio.sleep(delay)

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"\n❌ Error on hashtag '{hashtag}': {str(e)[:100]}")
                        print("   Saving progress and continuing...")
                        await self.save_progress()
                        continue

            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")

            finally:
                await self.save_progress()

                runtime = (datetime.now() - start_time).total_seconds() / 60
                print("\n" + "="*80)
                print("📊 SCRAPING COMPLETE")
                print("="*80)
                print(f"✅ Total unique videos: {len(self.video_data)}")
                print(f"🔍 Hashtags completed: {len(self.hashtags)}")
                print(f"⏱️  Total runtime: {runtime:.1f} minutes ({runtime/60:.1f} hours)")
                print(f"📁 Saved to: {self.output_file}")
                print("="*80)

                await browser.close()


async def main():
    scraper = TikTokCCTVScraper(output_file="tiktok_cctv_fights.json")
    await scraper.scrape_all_hashtags()


if __name__ == "__main__":
    asyncio.run(main())
