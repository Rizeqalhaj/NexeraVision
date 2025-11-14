#!/usr/bin/env python3
"""
Multi-Query Reddit Scraper
Scrapes multiple search queries to bypass Reddit's ~250 result limit per query
"""

import asyncio
import random
import json
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

class MultiQueryRedditScraper:
    def __init__(self, output_file="reddit_fight_videos_all.json"):
        self.output_file = Path(output_file)
        self.scraped_links = set()
        self.video_data = []

        # Load existing data if file exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing = json.load(f)
                self.video_data = existing
                self.scraped_links = {item['url'] for item in existing}

        # Search queries - each returns ~250 results
        self.queries = [
            "fights",
            "street fight",
            "fight video",
            "brawl",
            "knockout",
            "beatdown",
            "fist fight",
            "bar fight",
            "school fight",
            "fight compilation",
            "fight caught on camera",
            "violent fight",
            "brutal fight",
            "street brawl",
            "parking lot fight",
            "subway fight",
            "fight in public",
            "caught on camera fight",
            "fight worldstar",
            "fight footage",
            "fight recording",
            "fighting",
            "altercation",
            "physical fight",
            "real fight",
            "street fighting",
            "hood fight",
            "ghetto fight",
            "fight scene",
            "fight real life",
        ]

        self.scroll_delay = (2, 5)

    async def random_delay(self, min_sec, max_sec):
        await asyncio.sleep(random.uniform(min_sec, max_sec))

    async def extract_video_links(self, page):
        """Extract video links from current page"""
        links = await page.evaluate("""
            () => {
                const results = [];
                const seen = new Set();

                const postLinks = document.querySelectorAll('a[href*="/comments/"]');
                postLinks.forEach(link => {
                    const url = link.href.split('?')[0].split('#')[0];
                    if (url && !seen.has(url)) {
                        seen.add(url);

                        let container = link.closest('shreddit-post, article, [data-testid="post-container"]');
                        let title = 'Unknown';
                        let subreddit = 'Unknown';

                        if (container) {
                            const titleElem = container.querySelector('h3, [slot="title"], h1');
                            if (titleElem) title = titleElem.textContent.trim();

                            const subElem = container.querySelector('[data-testid="subreddit-name"], a[href*="/r/"]');
                            if (subElem) {
                                const subText = subElem.textContent || subElem.getAttribute('href') || '';
                                const match = subText.match(/r\\/([^\\/\\s]+)/);
                                if (match) subreddit = match[1];
                            }
                        }

                        if (subreddit === 'Unknown') {
                            const urlMatch = url.match(/\\/r\\/([^\\/]+)/);
                            if (urlMatch) subreddit = urlMatch[1];
                        }

                        results.push({url, title, subreddit});
                    }
                });

                const shredditPosts = document.querySelectorAll('shreddit-post');
                shredditPosts.forEach(post => {
                    const permalink = post.getAttribute('permalink');
                    if (permalink) {
                        const url = 'https://www.reddit.com' + permalink.split('?')[0];
                        if (!seen.has(url)) {
                            seen.add(url);
                            const title = post.getAttribute('post-title') || 'Unknown';
                            const subreddit = post.getAttribute('subreddit-prefixed-name')?.replace('r/', '') || 'Unknown';
                            results.push({url, title, subreddit});
                        }
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
        print(f"💾 Saved {len(self.video_data)} total unique videos")

    async def scrape_single_query(self, browser, query, query_num, total_queries):
        """Scrape a single query"""
        print(f"\n{'='*80}")
        print(f"📍 Query {query_num}/{total_queries}: \"{query}\"")
        print(f"{'='*80}\n")

        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York',
        )

        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
        """)

        page = await context.new_page()

        # Build search URL
        query_encoded = query.replace(' ', '%20')
        url = f"https://www.reddit.com/search/?q={query_encoded}&type=media"

        print(f"🌐 Loading: {url}")
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)  # 60 second timeout
        except Exception as e:
            print(f"   ⚠️ Timeout loading page, retrying with load event...")
            await page.goto(url, wait_until='load', timeout=60000)

        await self.random_delay(5, 8)

        # Handle popups
        try:
            buttons = await page.query_selector_all('button')
            for btn in buttons[:5]:
                text = await btn.inner_text()
                if any(word in text.lower() for word in ['continue', 'close', 'not now', 'maybe later']):
                    await btn.click()
                    await self.random_delay(1, 2)
                    break
        except:
            pass

        consecutive_no_new = 0
        query_new_count = 0

        # Scroll and collect (aiming for ~250 results per query)
        # Dynamic patience: more scrolls if we haven't reached the limit yet
        max_no_new = 5 if query_new_count >= 200 else 15  # Be patient until we get close to 250

        while consecutive_no_new < max_no_new and query_new_count < 300:
            new_links = await self.extract_video_links(page)

            new_count = 0
            for link_data in new_links:
                if link_data['url'] not in self.scraped_links:
                    self.scraped_links.add(link_data['url'])
                    self.video_data.append({
                        'url': link_data['url'],
                        'title': link_data['title'],
                        'subreddit': link_data['subreddit'],
                        'query': query,
                        'scraped_at': datetime.now().isoformat(),
                    })
                    new_count += 1
                    query_new_count += 1

            if new_count > 0:
                consecutive_no_new = 0
                # Update max_no_new dynamically based on progress
                max_no_new = 5 if query_new_count >= 200 else 15
                print(f"📹 +{new_count} videos | Query total: {query_new_count} | Overall: {len(self.video_data)}")
            else:
                consecutive_no_new += 1
                max_no_new = 5 if query_new_count >= 200 else 15
                print(f"⏳ Scrolling... ({consecutive_no_new}/{max_no_new}) | Got: {query_new_count}")

            # Early exit if we've hit the expected limit
            if query_new_count >= 240 and consecutive_no_new >= 5:
                print(f"   ✅ Reached ~250 videos, stopping")
                break

            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Wait for Reddit to load more content (use 3 seconds - matches working test)
            await page.wait_for_timeout(3000)

            await self.random_delay(1, 2)  # Additional small delay

        print(f"\n✅ Query complete: Found {query_new_count} new videos from \"{query}\"")

        await context.close()
        return query_new_count

    async def scrape_all_queries(self):
        """Scrape all queries"""
        start_time = datetime.now()

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║              MULTI-QUERY REDDIT SCRAPER                                    ║")
        print("║              Bypass 250-result limit with multiple queries                ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 Total queries: {len(self.queries)}")
        print(f"📊 Target: ~250 results per query (fast collection)")
        print(f"📊 Expected yield: {len(self.queries) * 200} - {len(self.queries) * 250} unique videos")
        print(f"   ({len(self.queries)} queries × 200-250 videos = 6,000-7,500 total)")
        print(f"📁 Output: {self.output_file}")
        print(f"✅ Already have: {len(self.scraped_links)} videos")
        print()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--disable-web-security',
                    '--no-sandbox',
                ]
            )

            try:
                for idx, query in enumerate(self.queries, 1):
                    try:
                        await self.scrape_single_query(browser, query, idx, len(self.queries))

                        # Save after each query
                        await self.save_progress()

                        # Longer delay between queries to avoid rate limits
                        if idx < len(self.queries):
                            delay = random.randint(30, 60)
                            print(f"\n💤 Resting {delay}s before next query...")
                            await asyncio.sleep(delay)

                    except KeyboardInterrupt:
                        raise  # Don't catch Ctrl+C
                    except Exception as e:
                        print(f"\n❌ Error on query '{query}': {str(e)[:100]}")
                        print("   Saving progress and continuing to next query...")
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
    scraper = MultiQueryRedditScraper(output_file="reddit_fight_videos_all.json")
    await scraper.scrape_all_queries()


if __name__ == "__main__":
    asyncio.run(main())
