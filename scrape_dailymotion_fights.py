#!/usr/bin/env python3
"""
Dailymotion Fight Video Scraper
NO LOGIN REQUIRED - Works on Vast.ai
Scrapes CCTV and fight videos from Dailymotion search
"""

import asyncio
import random
import json
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

class DailymotionFightScraper:
    def __init__(self, output_file="dailymotion_fight_videos.json"):
        self.output_file = Path(output_file)
        self.scraped_links = set()
        self.video_data = []

        # Load existing data if file exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing = json.load(f)
                self.video_data = existing
                self.scraped_links = {item['url'] for item in existing}

        # Search queries - CCTV focused
        self.queries = [
            "cctv fight",
            "security camera fight",
            "surveillance fight",
            "caught on camera fight",
            "cctv violence",
            "security camera violence",
            "surveillance footage",
            "cctv brawl",
            "security footage fight",
            "cctv street fight",
            "caught on cctv",
            "security cam fight",
            "cctv attack",
            "surveillance camera violence",
            "fight caught on camera",
            "street fight",
            "fight video",
            "knockout",
            "brawl",
            "real fight",
            "violent fight",
            "brutal fight",
            "public fight",
            "bar fight",
            "parking lot fight",
            "gas station fight",
        ]

    async def random_delay(self, min_sec, max_sec):
        await asyncio.sleep(random.uniform(min_sec, max_sec))

    async def extract_video_links(self, page):
        """Extract video links from Dailymotion search results"""
        videos = await page.evaluate("""
            () => {
                const results = [];
                const seen = new Set();

                // Method 1: Find ALL video links (more aggressive)
                const allLinks = document.querySelectorAll('a');

                allLinks.forEach(elem => {
                    try {
                        const href = elem.getAttribute('href');
                        if (!href || !href.includes('/video/')) return;

                        // Build full URL
                        let url = href;
                        if (href.startsWith('/')) {
                            url = 'https://www.dailymotion.com' + href;
                        } else if (!href.startsWith('http')) {
                            return; // Skip invalid URLs
                        }

                        // Remove query parameters
                        url = url.split('?')[0].split('#')[0];

                        if (seen.has(url)) return;
                        seen.add(url);

                        // Try to extract title from multiple sources
                        let title = 'Unknown';

                        // Try parent container text
                        const parent = elem.closest('[class*="video"], [class*="item"], [class*="card"]');
                        if (parent) {
                            const titleElem = parent.querySelector('[class*="title"]') ||
                                            parent.querySelector('h3') ||
                                            parent.querySelector('h2') ||
                                            parent.querySelector('[class*="name"]');
                            if (titleElem) {
                                title = titleElem.textContent.trim();
                            }
                        }

                        // Fallback: use link's own text or aria-label
                        if (title === 'Unknown') {
                            title = elem.getAttribute('aria-label') || elem.getAttribute('title') || elem.textContent.trim() || 'Unknown';
                        }

                        // Extract video ID from URL
                        const match = url.match(/\\/video\\/([^?#\\/]+)/);
                        const videoId = match ? match[1] : 'unknown';

                        results.push({
                            url: url,
                            title: title,
                            videoId: videoId
                        });
                    } catch (e) {
                        // Skip problematic elements
                    }
                });

                // Method 2: Look for data attributes
                const dataVideos = document.querySelectorAll('[data-video-id]');
                dataVideos.forEach(elem => {
                    try {
                        const videoId = elem.getAttribute('data-video-id');
                        if (!videoId) return;

                        const url = `https://www.dailymotion.com/video/${videoId}`;

                        if (seen.has(url)) return;
                        seen.add(url);

                        let title = 'Unknown';
                        const titleElem = elem.querySelector('[class*="title"]');
                        if (titleElem) {
                            title = titleElem.textContent.trim();
                        }

                        results.push({
                            url: url,
                            title: title,
                            videoId: videoId
                        });
                    } catch (e) {
                        // Skip
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

    async def scrape_single_query(self, browser, query, query_num, total_queries):
        """Scrape Dailymotion for a single query"""
        print(f"\n{'='*80}")
        print(f"📍 Query {query_num}/{total_queries}: \"{query}\"")
        print(f"{'='*80}\n")

        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
        )

        page = await context.new_page()

        # Dailymotion search URL
        query_encoded = query.replace(' ', '+')
        url = f"https://www.dailymotion.com/search/{query_encoded}/videos"

        print(f"🌐 Loading: {url}")
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
        except Exception as e:
            print(f"   ⚠️ Timeout loading page: {str(e)[:100]}")
            await context.close()
            return 0

        await self.random_delay(3, 5)

        consecutive_no_new = 0
        query_new_count = 0

        print("Scrolling and collecting videos...")

        # Dynamic patience
        while query_new_count < 300:
            max_no_new = 5 if query_new_count >= 150 else 15

            if consecutive_no_new >= max_no_new:
                break

            # Extract videos
            new_videos = await self.extract_video_links(page)

            new_count = 0
            for video_data in new_videos:
                if video_data['url'] not in self.scraped_links:
                    self.scraped_links.add(video_data['url'])
                    self.video_data.append({
                        'url': video_data['url'],
                        'title': video_data['title'],
                        'videoId': video_data['videoId'],
                        'query': query,
                        'source': 'dailymotion',
                        'scraped_at': datetime.now().isoformat(),
                    })
                    new_count += 1
                    query_new_count += 1

            if new_count > 0:
                consecutive_no_new = 0
                print(f"📹 +{new_count} videos | Query total: {query_new_count} | Overall: {len(self.video_data)}")
            else:
                consecutive_no_new += 1
                print(f"⏳ Scrolling... ({consecutive_no_new}/{max_no_new}) | Got: {query_new_count}")

            # Scroll - Dailymotion uses infinite scroll
            # Scroll to bottom first
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)

            # Then do multiple small scrolls to trigger lazy loading
            for _ in range(5):
                await page.evaluate("window.scrollBy(0, 800)")
                await asyncio.sleep(0.3)

            # Wait longer for Dailymotion to load new content
            await asyncio.sleep(4)

            # Try to click "Load More" button if it exists
            try:
                load_more = await page.query_selector('button:has-text("Load more"), button:has-text("Show more"), [class*="load"], [class*="more"]')
                if load_more:
                    await load_more.click()
                    await asyncio.sleep(2)
            except:
                pass

            # Early exit if we have enough
            if query_new_count >= 200 and consecutive_no_new >= 5:
                print(f"   ✅ Got {query_new_count} videos, moving on")
                break

        print(f"\n✅ Query complete: Found {query_new_count} new videos from \"{query}\"")

        await context.close()
        return query_new_count

    async def scrape_all_queries(self):
        """Scrape all queries"""
        start_time = datetime.now()

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              DAILYMOTION FIGHT VIDEO SCRAPER                               ║")
        print("║              NO LOGIN REQUIRED - Works on Vast.ai                          ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total queries: {len(self.queries)}")
        print(f"📊 Focus: CCTV and security camera footage")
        print(f"📊 Expected: ~100-200 videos per query")
        print(f"📊 Total expected: 3,000-5,000 unique videos")
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
                for idx, query in enumerate(self.queries, 1):
                    try:
                        await self.scrape_single_query(browser, query, idx, len(self.queries))

                        # Save after each query
                        await self.save_progress()

                        # Delay between queries
                        if idx < len(self.queries):
                            delay = random.randint(20, 40)
                            print(f"\n💤 Resting {delay}s before next query...")
                            await asyncio.sleep(delay)

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"\n❌ Error on query '{query}': {str(e)[:100]}")
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
                print(f"🔍 Queries completed: {len(self.queries)}")
                print(f"⏱️  Total runtime: {runtime:.1f} minutes ({runtime/60:.1f} hours)")
                print(f"📁 Saved to: {self.output_file}")
                print("="*80)

                await browser.close()


async def main():
    scraper = DailymotionFightScraper(output_file="dailymotion_fight_videos.json")
    await scraper.scrape_all_queries()


if __name__ == "__main__":
    asyncio.run(main())
