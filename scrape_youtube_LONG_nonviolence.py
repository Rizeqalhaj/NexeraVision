#!/usr/bin/env python3
"""
Scrape YouTube for LONG-FORM (5-30 min) non-violence videos
NOT Shorts - Real full-length videos for CCTV training
"""

import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION - LONG FORM VIDEOS ONLY
# ============================================================================

# Keywords for boring/normal CCTV-style content
KEYWORDS = [
    # Empty/boring CCTV
    "empty hallway security camera",
    "parking lot surveillance footage",
    "office lobby camera 24 hours",
    "building entrance security",
    "corridor surveillance camera",
    "waiting room security footage",
    "elevator security camera",
    "street surveillance camera",
    "mall security camera empty",
    "store surveillance quiet hours",
    "parking garage security",
    "apartment hallway camera",

    # Normal daily activities (long form)
    "people walking in mall",
    "office workers daily routine",
    "restaurant customers dining",
    "grocery store shoppers",
    "park people walking",
    "train station commuters",
    "airport terminal people",
    "library people studying",
    "cafe customers relaxing",
    "street pedestrians",

    # Boring surveillance scenarios
    "security camera compilation",
    "cctv footage normal day",
    "surveillance camera recording",
    "security footage nothing happening",
    "cctv normal activity",
    "surveillance video peaceful",

    # Long-form activity videos
    "time lapse people walking",
    "shopping center activity",
    "office building lobby",
    "subway station commute",
    "bus terminal waiting",
    "hospital waiting room",
    "university campus walking",
    "city street pedestrians",
]

OUTPUT_FILE = Path("/workspace/youtube_long_nonviolence_links.json")

# ============================================================================
# FILTERS FOR LONG VIDEOS ONLY
# ============================================================================

def is_long_video(duration_text):
    """
    Check if video is long (5+ minutes)
    Duration format: "5:23", "12:45", "1:23:45"
    """
    if not duration_text:
        return False

    try:
        parts = duration_text.split(':')
        if len(parts) == 2:  # MM:SS
            minutes = int(parts[0])
            return minutes >= 5  # At least 5 minutes
        elif len(parts) == 3:  # HH:MM:SS
            return True  # Any video with hours is long enough
    except:
        return False

    return False

# ============================================================================
# MAIN SCRAPER
# ============================================================================

print("=" * 80)
print("YOUTUBE LONG-FORM NON-VIOLENCE SCRAPER")
print("Target: 5-30 minute videos (NOT Shorts)")
print("=" * 80)
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
        print(f"\n[{i}/{len(KEYWORDS)}] Searching: '{keyword}'")

        try:
            # Build search URL - NO SHORTS FILTER, ADD DURATION FILTER
            # sp parameter for >4 minutes: EgQQARgB
            search_url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}&sp=EgQQARgB"

            driver.get(search_url)
            time.sleep(random.uniform(3, 5))

            # Scroll to load more results
            scroll_pause = random.uniform(2, 3)
            last_height = driver.execute_script("return document.documentElement.scrollHeight")

            scrolls = 0
            max_scrolls = 8  # Get ~150 videos per keyword

            while scrolls < max_scrolls:
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(scroll_pause)

                new_height = driver.execute_script("return document.documentElement.scrollHeight")
                if new_height == last_height:
                    break

                last_height = new_height
                scrolls += 1

            # Extract video links with duration check
            video_elements = driver.find_elements(By.CSS_SELECTOR, "a#video-title")

            new_links = 0
            for elem in video_elements:
                link = elem.get_attribute('href')

                if link and 'watch?v=' in link:
                    # Get duration from parent element
                    try:
                        # Find duration span near the video title
                        parent = elem.find_element(By.XPATH, "../..")
                        duration_elem = parent.find_element(By.CSS_SELECTOR, "span.style-scope.ytd-thumbnail-overlay-time-status-renderer")
                        duration_text = duration_elem.text.strip()

                        # Only add if duration is 5+ minutes
                        if is_long_video(duration_text):
                            video_id = link.split('watch?v=')[1].split('&')[0]
                            full_link = f"https://www.youtube.com/watch?v={video_id}"

                            if full_link not in all_links:
                                all_links.add(full_link)
                                new_links += 1
                                print(f"  + {duration_text} - {video_id}")
                    except:
                        # If can't find duration, assume it might be long and add it
                        video_id = link.split('watch?v=')[1].split('&')[0]
                        full_link = f"https://www.youtube.com/watch?v={video_id}"

                        if full_link not in all_links:
                            all_links.add(full_link)
                            new_links += 1

            print(f"  Found {new_links} new long videos (Total: {len(all_links)})")

            # Save progress every 5 keywords
            if i % 5 == 0:
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump({
                        'total_links': len(all_links),
                        'keywords_searched': i,
                        'links': list(all_links)
                    }, f, indent=2)
                print(f"  [Progress saved: {len(all_links)} videos]")

            # Random delay
            time.sleep(random.uniform(4, 7))

            # Stop if we have enough
            if len(all_links) >= 5000:
                print(f"\n✅ Reached 5,000 videos! Stopping.")
                break

        except Exception as e:
            print(f"  ⚠️  Error: {str(e)[:100]}")
            continue

finally:
    try:
        driver.quit()
    except:
        pass

# ============================================================================
# FINAL SAVE
# ============================================================================

print("\n" + "=" * 80)
print("SCRAPING COMPLETE")
print("=" * 80)
print()

final_links = list(all_links)

with open(OUTPUT_FILE, 'w') as f:
    json.dump({
        'total_links': len(final_links),
        'keywords_used': KEYWORDS,
        'video_type': 'long_form_5min_plus',
        'links': final_links
    }, f, indent=2)

print(f"✅ Scraped {len(final_links):,} long-form non-violence video links")
print(f"📄 Saved to: {OUTPUT_FILE}")
print()
print("Video duration: 5-30 minutes (NOT Shorts)")
print("Content: Boring CCTV, normal activities, surveillance footage")
print()
print("Next: Download videos")
print("  python3 download_youtube_LONG_videos.py")
print("=" * 80)
