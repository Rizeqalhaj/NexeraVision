#!/usr/bin/env python3
"""
YouTube Shorts Fight Video Scraper
Scrapes fight videos from YouTube Shorts infinite scroll
"""

import asyncio
import random
import json
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

class YouTubeShortsScraper:
    def __init__(self, output_file="youtube_shorts_fights.json"):
        self.output_file = Path(output_file)
        self.scraped_links = set()
        self.video_data = []

        # Load existing data if file exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing = json.load(f)
                self.video_data = existing
                self.scraped_links = {item['url'] for item in existing}

        # Search queries for YouTube Shorts - CCTV focused!
        self.queries = [
            "cctv fight",
            "cctv violence",
            "security camera fight",
            "surveillance fight",
            "caught on camera fight",
            "cctv footage fight",
            "security footage fight",
            "cctv brawl",
            "cctv street fight",
            "surveillance camera violence",
            "cctv assault",
            "security camera violence",
            "cctv attack",
            "store security fight",
            "gas station fight cctv",
            "parking lot security camera",
            "cctv fighting",
            "surveillance footage violence",
            "security cam fight",
            "cctv real fight",
            "street fight",
            "fight caught on camera",
            "real fight",
            "fight video",
            "knockout",
            "brawl",
            "public fight",
            "fight footage",
            "violent fight",
            "brutal fight",
        ]

    async def random_delay(self, min_sec, max_sec):
        await asyncio.sleep(random.uniform(min_sec, max_sec))

    async def extract_shorts_links(self, page):
        """Extract YouTube Shorts video links"""
        links = await page.evaluate("""
            () => {
                const results = [];
                const seen = new Set();

                // Find all Shorts links
                const shortLinks = document.querySelectorAll('a[href*="/shorts/"]');

                shortLinks.forEach(link => {
                    const url = 'https://www.youtube.com' + link.getAttribute('href').split('?')[0];

                    if (!seen.has(url)) {
                        seen.add(url);

                        // Try to get title
                        let title = 'Unknown';
                        const titleElem = link.querySelector('#video-title');
                        if (titleElem) {
                            title = titleElem.textContent.trim();
                        }

                        // Extract video ID
                        const match = url.match(/\\/shorts\\/([^?]+)/);
                        const videoId = match ? match[1] : 'unknown';

                        results.push({
                            url: url,
                            title: title,
                            videoId: videoId
                        });
                    }
                });

                return results;
            }
        """)

        return links

    async def save_progress(self):
        """Save scraped data to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.video_data, f, indent=2)
        print(f"💾 Saved {len(self.video_data)} total videos")

    async def scrape_single_query(self, browser, query, query_num, total_queries):
        """Scrape YouTube Shorts for a single query"""
        print(f"\n{'='*80}")
        print(f"📍 Query {query_num}/{total_queries}: \"{query}\"")
        print(f"{'='*80}\n")

        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
        )

        page = await context.new_page()

        # YouTube Shorts search URL
        query_encoded = query.replace(' ', '+')
        url = f"https://www.youtube.com/results?search_query={query_encoded}&sp=EgIYAQ%253D%253D"
        # sp=EgIYAQ%253D%253D is the filter for "Short" videos

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

        # Dynamic patience based on results
        print("Scrolling and collecting Shorts...")

        # Dynamic patience: be more patient early on
        while query_new_count < 500:
            max_no_new = 5 if query_new_count >= 100 else 20  # More patience at start

            if consecutive_no_new >= max_no_new:
                break
            # Extract links
            new_links = await self.extract_shorts_links(page)

            new_count = 0
            for link_data in new_links:
                if link_data['url'] not in self.scraped_links:
                    self.scraped_links.add(link_data['url'])
                    self.video_data.append({
                        'url': link_data['url'],
                        'title': link_data['title'],
                        'videoId': link_data['videoId'],
                        'query': query,
                        'source': 'youtube_shorts',
                        'scraped_at': datetime.now().isoformat(),
                    })
                    new_count += 1
                    query_new_count += 1

            if new_count > 0:
                consecutive_no_new = 0
                print(f"📹 +{new_count} shorts | Query total: {query_new_count} | Overall: {len(self.video_data)}")
            else:
                consecutive_no_new += 1
                print(f"⏳ Scrolling... ({consecutive_no_new}/{max_no_new}) | Got: {query_new_count}")

            # Scroll to bottom - YouTube needs multiple small scrolls
            for _ in range(3):
                await page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(0.5)

            # Wait for content to load
            await asyncio.sleep(3)

            # Early exit if we have enough
            if query_new_count >= 300 and consecutive_no_new >= 5:
                print(f"   ✅ Got {query_new_count} shorts, moving on")
                break

        print(f"\n✅ Query complete: Found {query_new_count} new shorts from \"{query}\"")

        await context.close()
        return query_new_count

    async def scrape_all_queries(self):
        """Scrape all queries"""
        start_time = datetime.now()

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              YOUTUBE SHORTS FIGHT VIDEO SCRAPER                            ║")
        print("║              Collect short-form fight videos from YouTube                 ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total queries: {len(self.queries)}")
        print(f"📊 Focus: CCTV and security camera footage")
        print(f"📊 Expected: ~100-200 shorts per query")
        print(f"📊 Total expected: 3,000-6,000 unique shorts")
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
                print(f"✅ Total unique shorts: {len(self.video_data)}")
                print(f"🔍 Queries completed: {len(self.queries)}")
                print(f"⏱️  Total runtime: {runtime:.1f} minutes ({runtime/60:.1f} hours)")
                print(f"📁 Saved to: {self.output_file}")
                print("="*80)

                await browser.close()


async def main():
    scraper = YouTubeShortsScraper(output_file="youtube_shorts_fights.json")
    await scraper.scrape_all_queries()


if __name__ == "__main__":
    asyncio.run(main())
