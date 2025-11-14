#!/usr/bin/env python3
"""
Scrape YouTube SHORTS for NON-VIOLENCE normal activities
People walking, talking, eating, cooking, working, shopping
"""

import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
from pathlib import Path

# NON-VIOLENCE NORMAL ACTIVITY KEYWORDS
KEYWORDS = [
    # Walking and moving
    "people walking street",
    "pedestrian crosswalk",
    "walking in mall",
    "people walking sidewalk",
    "walking corridor hallway",
    "people walking park",

    # Talking and socializing
    "people talking conversation",
    "people chatting cafe",
    "friends talking",
    "people discussion meeting",
    "people socializing",

    # Eating and food
    "people eating restaurant",
    "eating food cafeteria",
    "lunch break eating",
    "people dining",
    "eating meal together",
    "family eating dinner",

    # Cooking
    "cooking kitchen",
    "cooking food preparation",
    "chef cooking restaurant",
    "home cooking",
    "people cooking together",

    # Working
    "people working office",
    "office workers desk",
    "working warehouse",
    "store employees working",
    "people working computer",

    # Shopping
    "people shopping mall",
    "grocery shopping",
    "customers shopping store",
    "browsing store shelves",
    "shopping center people",

    # Public places
    "waiting room people",
    "people queue line",
    "subway station commute",
    "train station passengers",
    "airport travelers",
    "bus stop waiting",

    # Daily activities
    "people exercising gym",
    "reading library",
    "studying students",
    "cleaning housework",
    "gardening outdoor",

    # Add "normal" and "everyday" variants
    "normal daily activity",
    "everyday life routine",
    "typical day people",
    "regular activity",
    "peaceful activity",
]

output_file = Path("youtube_nonviolence_shorts.json")

print("="*80)
print("YOUTUBE SHORTS - NON-VIOLENCE SCRAPER")
print("Normal activities: walking, talking, eating, cooking, working, shopping")
print("Target: 10,000+ shorts")
print("="*80)
print()

# Setup Chrome
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]
chrome_options.add_argument(f'user-agent={random.choice(user_agents)}')

all_links = set()

try:
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    for i, keyword in enumerate(KEYWORDS, 1):
        print(f"\n[{i}/{len(KEYWORDS)}] '{keyword}'")

        try:
            # SHORTS ONLY filter
            search_url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}&sp=EgIYAQ%3D%3D"

            driver.get(search_url)
            time.sleep(random.uniform(2, 4))

            # Scroll to load results
            for scroll in range(10):  # 10 scrolls per keyword
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(random.uniform(1.5, 2.5))

            # Extract links
            video_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/shorts/'], a#video-title")

            new_links = 0
            for elem in video_elements:
                link = elem.get_attribute('href')
                if link:
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

            print(f"  +{new_links} videos (Total: {len(all_links)})")

            # Save progress
            if len(all_links) % 100 == 0:
                with open(output_file, 'w') as f:
                    json.dump({'total': len(all_links), 'links': list(all_links)}, f, indent=2)

            time.sleep(random.uniform(3, 6))

            if len(all_links) >= 10000:
                print(f"\n✅ Reached 10,000 videos!")
                break

        except Exception as e:
            print(f"  ⚠️  Error: {str(e)[:100]}")
            continue

finally:
    try:
        driver.quit()
    except:
        pass

# Save final
print("\n" + "="*80)
print("SCRAPING COMPLETE")
print("="*80)
print()

with open(output_file, 'w') as f:
    json.dump({
        'total_links': len(all_links),
        'keywords': KEYWORDS,
        'links': list(all_links)
    }, f, indent=2)

print(f"✅ Scraped {len(all_links):,} non-violence shorts")
print(f"📄 Saved to: {output_file}")
print()
print("Next: Run download script")
print("="*80)
