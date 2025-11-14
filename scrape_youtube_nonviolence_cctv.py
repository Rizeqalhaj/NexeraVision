#!/usr/bin/env python3
"""
Scrape YouTube for NON-VIOLENCE CCTV footage - UNDETECTED
Get 10K+ links for normal surveillance videos
"""

import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
from pathlib import Path

# Non-violence daily activities keywords
KEYWORDS = [
    # Eating and food
    "people eating lunch",
    "family dinner table",
    "eating breakfast",
    "restaurant customers dining",
    "cafe people drinking coffee",
    "food court eating",
    "picnic outdoor eating",
    "snack time break",

    # Cooking
    "cooking dinner kitchen",
    "baking cookies",
    "meal preparation",
    "chef cooking food",
    "home cooking recipe",
    "cooking together family",
    "making breakfast",

    # Shopping
    "grocery shopping store",
    "people browsing clothes",
    "shopping mall customers",
    "supermarket shopping cart",
    "buying vegetables market",
    "retail store shoppers",
    "window shopping",

    # Working and office
    "office workers desk",
    "typing computer work",
    "meeting room discussion",
    "working from home",
    "warehouse employees",
    "construction workers building",
    "delivery person job",

    # Walking and movement
    "people walking street",
    "pedestrians crosswalk",
    "walking in park",
    "hiking trail nature",
    "jogging morning run",
    "strolling downtown",

    # Talking and socializing
    "friends conversation",
    "people chatting",
    "group discussion",
    "phone call talking",
    "laughing together",
    "friendly chat",

    # Daily routines
    "brushing teeth morning",
    "getting dressed",
    "reading newspaper",
    "watching television",
    "playing cards game",
    "studying homework",
    "cleaning house",
    "doing laundry",
    "watering plants garden",

    # Exercise and health
    "yoga stretching",
    "gym workout exercise",
    "morning stretches",
    "walking dog park",
    "bicycle riding",
    "playground children playing",

    # Waiting and transit
    "waiting bus stop",
    "train commute passengers",
    "airport waiting lounge",
    "sitting bench resting",
    "queue standing line",
]

output_file = Path("youtube_nonviolence_cctv_links.json")

print("="*80)
print("YOUTUBE SHORTS NON-VIOLENCE CCTV SCRAPER")
print("Undetected mode - scraping 10K+ normal surveillance SHORTS")
print("="*80)
print()

# Setup undetected Chrome
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# Random user agents
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]
chrome_options.add_argument(f'user-agent={random.choice(user_agents)}')

all_links = set()

try:
    driver = webdriver.Chrome(options=chrome_options)

    # Make browser look less like automation
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    for i, keyword in enumerate(KEYWORDS, 1):
        print(f"\n[{i}/{len(KEYWORDS)}] Searching: '{keyword}'")

        try:
            # Build search URL with SHORTS ONLY filter
            search_url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}&sp=EgIYAQ%3D%3D"

            driver.get(search_url)
            time.sleep(random.uniform(2, 4))

            # Scroll to load more results
            scroll_pause = random.uniform(1.5, 2.5)
            last_height = driver.execute_script("return document.documentElement.scrollHeight")

            scrolls = 0
            max_scrolls = 10  # Get ~200 videos per keyword

            while scrolls < max_scrolls:
                # Scroll down
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(scroll_pause)

                # Calculate new scroll height
                new_height = driver.execute_script("return document.documentElement.scrollHeight")

                if new_height == last_height:
                    break

                last_height = new_height
                scrolls += 1

            # Extract Shorts links
            video_elements = driver.find_elements(By.CSS_SELECTOR, "a#video-title, a[href*='/shorts/']")

            new_links = 0
            for elem in video_elements:
                link = elem.get_attribute('href')
                if link:
                    # Handle both /shorts/ and regular watch URLs
                    if '/shorts/' in link:
                        video_id = link.split('/shorts/')[1].split('?')[0]
                        full_link = f"https://www.youtube.com/shorts/{video_id}"
                    elif 'watch?v=' in link:
                        video_id = link.split('watch?v=')[1].split('&')[0]
                        full_link = f"https://www.youtube.com/watch?v={video_id}"
                    else:
                        continue

                    if full_link not in all_links:
                        all_links.add(full_link)
                        new_links += 1

            print(f"  Found {new_links} new videos (Total: {len(all_links)})")

            # Save progress
            if len(all_links) % 100 == 0:
                with open(output_file, 'w') as f:
                    json.dump({
                        'keyword': keyword,
                        'total_links': len(all_links),
                        'links': list(all_links)
                    }, f, indent=2)

            # Random delay between searches
            time.sleep(random.uniform(3, 6))

            # Stop if we have enough
            if len(all_links) >= 10000:
                print(f"\n✅ Reached 10,000 videos! Stopping.")
                break

        except Exception as e:
            print(f"  ⚠️  Error on keyword '{keyword}': {str(e)[:100]}")
            continue

finally:
    try:
        driver.quit()
    except:
        pass

# Final save
print("\n" + "="*80)
print("SCRAPING COMPLETE")
print("="*80)
print()

final_links = list(all_links)

# Save final results
with open(output_file, 'w') as f:
    json.dump({
        'total_links': len(final_links),
        'keywords_used': KEYWORDS,
        'links': final_links
    }, f, indent=2)

print(f"✅ Scraped {len(final_links):,} non-violence CCTV video links")
print(f"📄 Saved to: {output_file}")
print()
print("Next: Run download script to get the videos")
print("="*80)
