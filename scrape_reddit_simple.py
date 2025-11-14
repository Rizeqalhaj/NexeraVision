#!/usr/bin/env python3
"""
Simple Reddit Scraper - Matches the working test exactly
Uses the exact timing that successfully reached 500 results
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

async def scrape_reddit_simple(query, output_file):
    """
    Simple scraper that uses exact timing from successful test
    """
    output_path = Path(output_file)

    # Load existing data
    scraped_links = set()
    video_data = []

    if output_path.exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
            video_data = existing
            scraped_links = {item['url'] for item in existing}

    print(f"Starting scrape for query: '{query}'")
    print(f"Already have: {len(scraped_links)} videos")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await context.new_page()

        # Build URL
        query_encoded = query.replace(' ', '%20')
        url = f"https://www.reddit.com/search/?q={query_encoded}&type=media"

        print(f"Loading: {url}")
        await page.goto(url, wait_until='domcontentloaded')
        await asyncio.sleep(5)

        prev_count = 0
        no_change_count = 0
        scroll_num = 0

        print("\nScrolling and collecting...")
        print("="*60)

        while no_change_count < 30:  # More patient than test (30 vs 20)
            # Extract links
            links = await page.evaluate("""
                () => {
                    const results = [];
                    const seen = new Set();

                    const postLinks = document.querySelectorAll('a[href*="/comments/"]');
                    postLinks.forEach(link => {
                        const url = link.href.split('?')[0].split('#')[0];
                        if (url && !seen.has(url)) {
                            seen.add(url);

                            let container = link.closest('shreddit-post, article');
                            let title = 'Unknown';
                            let subreddit = 'Unknown';

                            if (container) {
                                const titleElem = container.querySelector('h3, [slot="title"], h1');
                                if (titleElem) title = titleElem.textContent.trim();
                            }

                            const urlMatch = url.match(/\\/r\\/([^\\/]+)/);
                            if (urlMatch) subreddit = urlMatch[1];

                            results.push({url, title, subreddit});
                        }
                    });

                    return results;
                }
            """)

            # Add new videos
            new_count = 0
            for link_data in links:
                if link_data['url'] not in scraped_links:
                    scraped_links.add(link_data['url'])
                    video_data.append({
                        'url': link_data['url'],
                        'title': link_data['title'],
                        'subreddit': link_data['subreddit'],
                        'query': query,
                        'scraped_at': datetime.now().isoformat(),
                    })
                    new_count += 1

            current_total = len(video_data)

            # Check progress
            if new_count > 0:
                no_change_count = 0
                print(f"Scroll {scroll_num:2d}: {current_total} total (+{new_count} new)")
            else:
                no_change_count += 1
                print(f"Scroll {scroll_num:2d}: {current_total} total (no change {no_change_count}/30)")

            # Scroll to bottom (exactly like test)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)  # Exact timing from successful test

            scroll_num += 1

            # Save progress every 10 scrolls
            if scroll_num % 10 == 0:
                with open(output_path, 'w') as f:
                    json.dump(video_data, f, indent=2)
                print(f"   💾 Saved progress ({len(video_data)} videos)")

        print("="*60)
        print(f"\n✅ Complete: Collected {len(video_data)} total videos")
        print(f"   Query '{query}' found {current_total - len(scraped_links) + new_count} new videos")

        # Final save
        with open(output_path, 'w') as f:
            json.dump(video_data, f, indent=2)

        await browser.close()

        return len(video_data)


async def scrape_multiple_queries():
    """Scrape multiple queries sequentially"""

    queries = [
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

    output_file = "reddit_fight_videos_all.json"

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║              SIMPLE MULTI-QUERY REDDIT SCRAPER                             ║")
    print("║              Uses timing that successfully reached 500 results             ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"📊 Total queries: {len(queries)}")
    print(f"📊 Expected: ~500 results per query")
    print(f"📊 Total expected: 10,000-15,000 unique videos")
    print(f"📁 Output: {output_file}")
    print()

    start_time = datetime.now()

    for idx, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {idx}/{len(queries)}: '{query}'")
        print(f"{'='*80}\n")

        try:
            total = await scrape_reddit_simple(query, output_file)
            print(f"\n✅ Query {idx} complete. Running total: {total} videos")

            # Rest between queries
            if idx < len(queries):
                print(f"\n💤 Resting 60 seconds before next query...")
                await asyncio.sleep(60)

        except Exception as e:
            print(f"\n❌ Error on query '{query}': {e}")
            print("   Continuing to next query...")
            continue

    runtime = (datetime.now() - start_time).total_seconds() / 60

    print("\n" + "="*80)
    print("📊 ALL QUERIES COMPLETE")
    print("="*80)

    # Load final count
    with open(output_file, 'r') as f:
        final_data = json.load(f)

    print(f"✅ Total unique videos: {len(final_data)}")
    print(f"⏱️  Total runtime: {runtime:.1f} minutes ({runtime/60:.1f} hours)")
    print(f"📁 Saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(scrape_multiple_queries())
