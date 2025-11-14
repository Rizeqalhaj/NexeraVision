#!/usr/bin/env python3
"""
Quick test to find Reddit's actual search result limit
"""

import asyncio
from playwright.async_api import async_playwright

async def test_reddit_limit():
    print("Testing Reddit's search result limit...")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()

        url = "https://www.reddit.com/search/?q=fights&type=media"
        print(f"Loading: {url}")
        await page.goto(url, wait_until='domcontentloaded')
        await asyncio.sleep(5)

        prev_count = 0
        no_change_count = 0
        scroll_num = 0

        print("\nScrolling to find limit...")
        print("="*60)

        while no_change_count < 20:  # Stop after 20 scrolls with no change
            # Get current count
            current_count = await page.evaluate('document.querySelectorAll("a[href*=\\"/comments/\\"]").length')

            # Check if changed
            if current_count > prev_count:
                new_items = current_count - prev_count
                print(f"Scroll {scroll_num:2d}: {current_count} total (+{new_items} new)")
                no_change_count = 0
            else:
                no_change_count += 1
                print(f"Scroll {scroll_num:2d}: {current_count} total (no change {no_change_count}/20)")

            prev_count = current_count

            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)  # Wait for content to load

            scroll_num += 1

        print("="*60)
        print(f"\n✅ Final result: Reddit's limit appears to be ~{current_count} results")
        print(f"   (Stopped after {no_change_count} scrolls with no new content)")
        print()

        # Check for "Show more" or pagination buttons
        buttons = await page.query_selector_all('button')
        print("Looking for pagination buttons...")
        for btn in buttons[:10]:
            try:
                text = await btn.inner_text()
                if any(word in text.lower() for word in ['more', 'next', 'load', 'show']):
                    print(f"   Found button: '{text}'")
            except:
                pass

        await browser.close()

        # Conclusion
        print()
        print("="*60)
        if current_count < 300:
            print("⚠️  Reddit limits single query to ~250-300 results")
            print("✅ Solution: Use multi-query scraper with different keywords")
            print("   30 queries × 250 = 7,500+ videos")
        else:
            print(f"✅ Limit is higher than expected: {current_count} results!")
            print("   Single-query scraper should work well")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(test_reddit_limit())
