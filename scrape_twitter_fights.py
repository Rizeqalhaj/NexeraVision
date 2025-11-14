#!/usr/bin/env python3
"""
Twitter/X Fight Video Scraper
Scrapes fight videos from Twitter search with CCTV focus
"""

import asyncio
import random
import json
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

class TwitterFightScraper:
    def __init__(self, output_file="twitter_fight_videos.json"):
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
            "surveillance footage fight",
            "caught on camera fight",
            "cctv violence",
            "security camera violence",
            "cctv brawl",
            "cctv assault",
            "surveillance fight",
            "caught on cctv",
            "security footage violence",
            "cctv street fight",
            "security cam fight",
            "cctv attack",
            "surveillance camera violence",
            "fight caught on camera",
            "street fight",
            "fight video",
            "knockout",
            "brawl",
            "public fight",
            "real fight",
            "violent fight",
            "brutal fight",
            "bar fight",
            "parking lot fight",
        ]

    async def random_delay(self, min_sec, max_sec):
        await asyncio.sleep(random.uniform(min_sec, max_sec))

    async def extract_tweet_videos(self, page):
        """Extract video tweets from current page"""
        tweets = await page.evaluate("""
            () => {
                const results = [];
                const seen = new Set();

                // Method 1: Find all tweet articles
                const articles = document.querySelectorAll('article[data-testid="tweet"]');

                articles.forEach(article => {
                    try {
                        // Look for video indicator
                        const videoDiv = article.querySelector('div[data-testid="videoPlayer"]');
                        if (!videoDiv) return;

                        // Get tweet link
                        const timeLink = article.querySelector('a[href*="/status/"]');
                        if (!timeLink) return;

                        const url = 'https://twitter.com' + timeLink.getAttribute('href').split('?')[0];

                        if (seen.has(url)) return;
                        seen.add(url);

                        // Get tweet text
                        let text = 'Unknown';
                        const textDiv = article.querySelector('div[data-testid="tweetText"]');
                        if (textDiv) {
                            text = textDiv.textContent.trim();
                        }

                        // Get username
                        let username = 'unknown';
                        const userLink = article.querySelector('a[href^="/"]');
                        if (userLink) {
                            const href = userLink.getAttribute('href');
                            if (href && href.startsWith('/') && !href.includes('/status/')) {
                                username = href.substring(1).split('/')[0];
                            }
                        }

                        // Extract tweet ID
                        const match = url.match(/\\/status\\/([0-9]+)/);
                        const tweetId = match ? match[1] : 'unknown';

                        results.push({
                            url: url,
                            text: text,
                            username: username,
                            tweetId: tweetId
                        });
                    } catch (e) {
                        // Skip problematic tweets
                    }
                });

                // Method 2: Find video player elements directly
                const videoPlayers = document.querySelectorAll('div[data-testid="videoPlayer"]');
                videoPlayers.forEach(player => {
                    try {
                        // Navigate up to article
                        let article = player.closest('article[data-testid="tweet"]');
                        if (!article) return;

                        const timeLink = article.querySelector('a[href*="/status/"]');
                        if (!timeLink) return;

                        const url = 'https://twitter.com' + timeLink.getAttribute('href').split('?')[0];

                        if (seen.has(url)) return;
                        seen.add(url);

                        const match = url.match(/\\/status\\/([0-9]+)/);
                        const tweetId = match ? match[1] : 'unknown';

                        results.push({
                            url: url,
                            text: 'Video tweet',
                            username: 'unknown',
                            tweetId: tweetId
                        });
                    } catch (e) {
                        // Skip problematic elements
                    }
                });

                return results;
            }
        """)

        return tweets

    async def save_progress(self):
        """Save scraped data to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.video_data, f, indent=2)
        print(f"💾 Saved {len(self.video_data)} total videos")

    async def scrape_single_query(self, browser, query, query_num, total_queries):
        """Scrape Twitter for a single query"""
        print(f"\n{'='*80}")
        print(f"📍 Query {query_num}/{total_queries}: \"{query}\"")
        print(f"{'='*80}\n")

        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
        )

        page = await context.new_page()

        # Twitter search URL with video filter
        query_encoded = query.replace(' ', '%20')
        url = f"https://twitter.com/search?q={query_encoded}%20filter%3Avideos&src=typed_query&f=live"

        print(f"🌐 Loading: {url}")
        print("⚠️  Note: Twitter may show login prompt - will try to bypass")

        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
        except Exception as e:
            print(f"   ⚠️ Timeout loading page: {str(e)[:100]}")
            await context.close()
            return 0

        await self.random_delay(5, 8)

        # Check if login is required
        login_check = await page.evaluate("""
            () => {
                const loginText = document.body.innerText.toLowerCase();
                return loginText.includes('sign in') ||
                       loginText.includes('log in') ||
                       loginText.includes('create account');
            }
        """)

        if login_check:
            print("⚠️  Twitter requires login - this query will be limited")
            print("   Continuing with available content...")

        consecutive_no_new = 0
        query_new_count = 0

        print("Scrolling and collecting videos...")

        # Dynamic patience
        while query_new_count < 300:
            max_no_new = 5 if query_new_count >= 150 else 15

            if consecutive_no_new >= max_no_new:
                break

            # Extract tweets
            new_tweets = await self.extract_tweet_videos(page)

            new_count = 0
            for tweet_data in new_tweets:
                if tweet_data['url'] not in self.scraped_links:
                    self.scraped_links.add(tweet_data['url'])
                    self.video_data.append({
                        'url': tweet_data['url'],
                        'text': tweet_data['text'],
                        'username': tweet_data['username'],
                        'tweetId': tweet_data['tweetId'],
                        'query': query,
                        'source': 'twitter',
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

            # Scroll - Twitter uses multiple small scrolls
            for _ in range(3):
                await page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(0.5)

            # Wait for content to load
            await asyncio.sleep(3)

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
        print("║              TWITTER/X FIGHT VIDEO SCRAPER                                 ║")
        print("║              CCTV and security camera focus                                ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total queries: {len(self.queries)}")
        print(f"📊 Focus: CCTV, security camera, surveillance footage")
        print(f"📊 Expected: ~100-200 videos per query")
        print(f"📊 Total expected: 3,000-5,000 unique videos")
        print(f"📁 Output: {self.output_file}")
        print(f"✅ Already have: {len(self.scraped_links)} videos")
        print()
        print("⚠️  Note: Twitter may limit access without login")
        print("   Results will be best-effort collection")
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
                            delay = random.randint(30, 60)
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
    scraper = TwitterFightScraper(output_file="twitter_fight_videos.json")
    await scraper.scrape_all_queries()


if __name__ == "__main__":
    asyncio.run(main())
