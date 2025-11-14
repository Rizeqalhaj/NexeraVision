#!/usr/bin/env python3
"""
Undetected Reddit Infinite Scroll Scraper V2
Enhanced extraction logic for Reddit's current layout
"""

import asyncio
import random
import json
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

class StealthRedditScraper:
    def __init__(self, output_file="reddit_fight_videos.json"):
        self.output_file = Path(output_file)
        self.scraped_links = set()
        self.video_data = []

        # Load existing data if file exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing = json.load(f)
                self.video_data = existing
                self.scraped_links = {item['url'] for item in existing}

        # Human-like timing
        self.scroll_delay = (2, 5)
        self.action_delay = (0.5, 2)

    async def random_delay(self, min_sec, max_sec):
        """Random delay to mimic human behavior"""
        await asyncio.sleep(random.uniform(min_sec, max_sec))

    async def human_like_scroll(self, page):
        """Scroll like a human - variable distances and pauses"""
        scroll_distance = random.randint(300, 800)

        if random.random() < 0.15:
            await page.evaluate(f"window.scrollBy(0, -{random.randint(100, 300)})")
            await self.random_delay(0.5, 1.5)

        await page.evaluate(f"window.scrollBy(0, {scroll_distance})")
        await self.random_delay(*self.scroll_delay)

        if random.random() < 0.3:
            await self.random_delay(3, 8)

    async def move_mouse_randomly(self, page):
        """Randomly move mouse to appear human"""
        if random.random() < 0.4:
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            await page.mouse.move(x, y)
            await self.random_delay(0.1, 0.5)

    async def extract_video_links(self, page):
        """Extract video links - enhanced for Reddit's current structure"""
        links = await page.evaluate("""
            () => {
                const results = [];
                const seen = new Set();

                // Method 1: Find all links to post pages
                const postLinks = document.querySelectorAll('a[href*="/comments/"]');
                postLinks.forEach(link => {
                    const url = link.href.split('?')[0].split('#')[0];
                    if (url && !seen.has(url)) {
                        seen.add(url);

                        // Try to find post container
                        let container = link.closest('shreddit-post, article, [data-testid="post-container"]');
                        let title = 'Unknown';
                        let subreddit = 'Unknown';

                        if (container) {
                            // Try to extract title
                            const titleElem = container.querySelector('h3, [slot="title"], h1');
                            if (titleElem) title = titleElem.textContent.trim();

                            // Try to extract subreddit
                            const subElem = container.querySelector('[data-testid="subreddit-name"], a[href*="/r/"]');
                            if (subElem) {
                                const subText = subElem.textContent || subElem.getAttribute('href') || '';
                                const match = subText.match(/r\\/([^\\/\\s]+)/);
                                if (match) subreddit = match[1];
                            }
                        }

                        // Try to extract from URL if not found
                        if (subreddit === 'Unknown') {
                            const urlMatch = url.match(/\\/r\\/([^\\/]+)/);
                            if (urlMatch) subreddit = urlMatch[1];
                        }

                        results.push({url, title, subreddit});
                    }
                });

                // Method 2: Find shreddit-post elements directly
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

                // Method 3: Look for data-post-click-location
                const dataPostElems = document.querySelectorAll('[data-post-click-location]');
                dataPostElems.forEach(elem => {
                    const linkElem = elem.querySelector('a[href*="/comments/"]');
                    if (linkElem) {
                        const url = linkElem.href.split('?')[0].split('#')[0];
                        if (!seen.has(url)) {
                            seen.add(url);

                            const titleElem = elem.querySelector('h3, h1, [id*="post-title"]');
                            const title = titleElem ? titleElem.textContent.trim() : 'Unknown';

                            const urlMatch = url.match(/\\/r\\/([^\\/]+)/);
                            const subreddit = urlMatch ? urlMatch[1] : 'Unknown';

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

        print(f"\n💾 Saved {len(self.video_data)} videos to {self.output_file}")

    async def debug_page_structure(self, page):
        """Debug: Print page structure to understand what's available"""
        structure = await page.evaluate("""
            () => {
                const info = {
                    url: window.location.href,
                    title: document.title,
                    shredditPosts: document.querySelectorAll('shreddit-post').length,
                    commentLinks: document.querySelectorAll('a[href*="/comments/"]').length,
                    articles: document.querySelectorAll('article').length,
                    h3Elements: document.querySelectorAll('h3').length,
                    sampleHTML: ''
                };

                // Get sample of first post HTML
                const firstPost = document.querySelector('shreddit-post, article, [data-testid="post-container"]');
                if (firstPost) {
                    info.sampleHTML = firstPost.outerHTML.substring(0, 500);
                }

                return info;
            }
        """)

        print("\n🔍 DEBUG: Page Structure")
        print(f"   URL: {structure['url']}")
        print(f"   Title: {structure['title']}")
        print(f"   shreddit-post elements: {structure['shredditPosts']}")
        print(f"   /comments/ links: {structure['commentLinks']}")
        print(f"   article elements: {structure['articles']}")
        print(f"   h3 elements: {structure['h3Elements']}")
        print(f"\n   Sample HTML (first 500 chars):")
        print(f"   {structure['sampleHTML']}")
        print()

    async def scrape_reddit_fights(self, target_url, max_videos=10000, max_runtime_minutes=180):
        """
        Scrape Reddit fight videos with stealth techniques

        Args:
            target_url: Reddit search URL
            max_videos: Stop after collecting this many videos
            max_runtime_minutes: Stop after this many minutes
        """
        start_time = datetime.now()

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

                window.navigator.chrome = {
                    runtime: {}
                };

                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
            """)

            page = await context.new_page()

            print("🚀 Starting Reddit scraper...")
            print(f"📊 Target: {max_videos} videos or {max_runtime_minutes} minutes")
            print(f"📁 Output: {self.output_file}")
            print(f"✅ Already have: {len(self.scraped_links)} videos")
            print()

            print(f"🌐 Loading: {target_url}")
            await page.goto(target_url, wait_until='domcontentloaded')
            await self.random_delay(5, 10)  # Longer initial wait for Reddit

            # Debug page structure on first load
            await self.debug_page_structure(page)

            # Handle popups/dialogs
            try:
                # Look for "Get App" or "Continue" buttons
                buttons = await page.query_selector_all('button')
                for btn in buttons[:5]:  # Check first 5 buttons
                    text = await btn.inner_text()
                    if any(word in text.lower() for word in ['continue', 'close', 'not now', 'maybe later']):
                        await btn.click()
                        await self.random_delay(1, 2)
                        break
            except:
                pass

            consecutive_no_new = 0
            scroll_count = 0
            last_save = datetime.now()

            try:
                while True:
                    runtime = (datetime.now() - start_time).total_seconds() / 60
                    if len(self.video_data) >= max_videos:
                        print(f"\n✅ Reached target: {len(self.video_data)} videos")
                        break

                    if runtime >= max_runtime_minutes:
                        print(f"\n⏰ Reached time limit: {runtime:.1f} minutes")
                        break

                    if consecutive_no_new >= 20:
                        print(f"\n⚠️ No new videos found in 20 scrolls - reached end")
                        break

                    # Extract links
                    new_links = await self.extract_video_links(page)

                    # Process new links
                    new_count = 0
                    for link_data in new_links:
                        if link_data['url'] not in self.scraped_links:
                            self.scraped_links.add(link_data['url'])
                            self.video_data.append({
                                'url': link_data['url'],
                                'title': link_data['title'],
                                'subreddit': link_data['subreddit'],
                                'scraped_at': datetime.now().isoformat(),
                            })
                            new_count += 1

                    if new_count > 0:
                        consecutive_no_new = 0
                        print(f"📹 Found {new_count} new videos | Total: {len(self.video_data)} | Runtime: {runtime:.1f}min")
                    else:
                        consecutive_no_new += 1
                        print(f"⏳ Scrolling... ({consecutive_no_new}/20 no new) | Total: {len(self.video_data)}")

                    # Debug every 10 scrolls if still finding nothing
                    if consecutive_no_new > 0 and consecutive_no_new % 10 == 0 and len(self.video_data) < 10:
                        print(f"\n⚠️ Still finding nothing after {consecutive_no_new} scrolls - debugging...")
                        await self.debug_page_structure(page)

                    await self.move_mouse_randomly(page)
                    await self.human_like_scroll(page)

                    scroll_count += 1

                    # Save progress every 5 minutes
                    if (datetime.now() - last_save).total_seconds() > 300:
                        await self.save_progress()
                        last_save = datetime.now()

                    # Random longer pause every ~10 scrolls
                    if scroll_count % 10 == 0:
                        print(f"💤 Taking a short break...")
                        await self.random_delay(10, 20)

            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")

            finally:
                await self.save_progress()

                print("\n" + "="*80)
                print("📊 SCRAPING COMPLETE")
                print("="*80)
                print(f"✅ Total videos collected: {len(self.video_data)}")
                print(f"⏱️  Runtime: {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes")
                print(f"📁 Saved to: {self.output_file}")
                print("="*80)

                await browser.close()


async def main():
    REDDIT_URL = "https://www.reddit.com/search/?q=fights&type=media&cId=79848f7c-3362-4d50-80e5-177ad738eeda&iId=81cdbaa0-5532-4ac9-bd47-8fc91af93b9e"
    OUTPUT_FILE = "reddit_fight_videos.json"
    MAX_VIDEOS = 10000
    MAX_RUNTIME = 180

    scraper = StealthRedditScraper(output_file=OUTPUT_FILE)
    await scraper.scrape_reddit_fights(
        target_url=REDDIT_URL,
        max_videos=MAX_VIDEOS,
        max_runtime_minutes=MAX_RUNTIME
    )


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║              UNDETECTED REDDIT INFINITE SCROLL SCRAPER V2                  ║")
    print("║              Enhanced Extraction - Stealth Mode                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()

    asyncio.run(main())
