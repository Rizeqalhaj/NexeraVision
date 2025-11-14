#!/usr/bin/env python3
"""
ROBUST YouTube Shorts Scraper - NON-VIOLENCE
Multiple fallback methods, retry logic, handles failures
"""

import time
import random
import json
from pathlib import Path
import subprocess

output_file = Path("youtube_nonviolence_shorts.json")

# NON-VIOLENCE keywords
KEYWORDS = [
    "people walking", "people talking", "eating food", "cooking kitchen",
    "people working office", "shopping mall", "grocery shopping",
    "people sitting", "reading book", "studying", "waiting room",
    "commuting train", "bus passengers", "airport travelers",
    "exercising gym", "yoga", "stretching", "cleaning house",
    "gardening", "playing cards", "board games", "family dinner",
    "cafe customers", "library quiet", "people queue",
    "street pedestrians", "park walking", "jogging running",
]

print("="*80)
print("ROBUST YOUTUBE SHORTS SCRAPER - NON-VIOLENCE")
print("Multiple methods with fallbacks")
print("="*80)
print()

all_links = set()

# ============================================================================
# METHOD 1: Selenium (Primary)
# ============================================================================
print("METHOD 1: Trying Selenium...")
print()

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    for i, keyword in enumerate(KEYWORDS, 1):
        if len(all_links) >= 10000:
            break

        print(f"[{i}/{len(KEYWORDS)}] {keyword}")

        try:
            # Shorts filter
            url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}&sp=EgIYAQ%3D%3D"
            driver.get(url)
            time.sleep(random.uniform(3, 5))

            # Scroll
            for _ in range(8):
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(random.uniform(1, 2))

            # Extract links
            elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/shorts/'], a#video-title")

            for elem in elements:
                try:
                    href = elem.get_attribute('href')
                    if href:
                        if '/shorts/' in href:
                            vid_id = href.split('/shorts/')[1].split('?')[0]
                            link = f"https://www.youtube.com/shorts/{vid_id}"
                        elif 'watch?v=' in href:
                            vid_id = href.split('watch?v=')[1].split('&')[0]
                            link = f"https://www.youtube.com/watch?v={vid_id}"
                        else:
                            continue

                        if link not in all_links:
                            all_links.add(link)
                except:
                    continue

            print(f"  Total: {len(all_links)}")
            time.sleep(random.uniform(2, 4))

        except Exception as e:
            print(f"  Error: {str(e)[:50]}")
            continue

    driver.quit()
    print(f"\n✅ Selenium method: {len(all_links)} links")

except Exception as e:
    print(f"❌ Selenium failed: {str(e)[:100]}")

# ============================================================================
# METHOD 2: yt-dlp search (Fallback)
# ============================================================================
if len(all_links) < 1000:
    print("\n" + "="*80)
    print("METHOD 2: Trying yt-dlp search...")
    print("="*80)
    print()

    subprocess.run(['pip', 'install', '-q', 'yt-dlp'], check=False)

    for keyword in KEYWORDS[:20]:  # Limit to 20 keywords
        try:
            print(f"Searching: {keyword}")

            cmd = [
                'yt-dlp',
                '--flat-playlist',
                '--print', 'url',
                '--match-filter', 'duration < 60',  # Shorts only
                f'ytsearch50:{keyword}'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('http'):
                        all_links.add(line)

                print(f"  Total: {len(all_links)}")

        except subprocess.TimeoutExpired:
            print(f"  Timeout")
        except Exception as e:
            print(f"  Error: {str(e)[:50]}")
            continue

    print(f"\n✅ yt-dlp method: {len(all_links)} total links")

# ============================================================================
# METHOD 3: YouTube API search (if available)
# ============================================================================
if len(all_links) < 5000:
    print("\n" + "="*80)
    print("METHOD 3: Trying YouTube Data API...")
    print("="*80)
    print()

    try:
        import requests

        # Try without API key (limited results)
        for keyword in KEYWORDS[:10]:
            try:
                print(f"Searching: {keyword}")

                # Use invidious API (YouTube mirror)
                url = f"https://vid.puffyan.us/api/v1/search?q={keyword.replace(' ', '+')}&type=short"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        if 'videoId' in item:
                            link = f"https://www.youtube.com/shorts/{item['videoId']}"
                            all_links.add(link)

                    print(f"  Total: {len(all_links)}")
                    time.sleep(2)

            except Exception as e:
                print(f"  Error: {str(e)[:50]}")
                continue

        print(f"\n✅ API method: {len(all_links)} total links")

    except Exception as e:
        print(f"❌ API method failed: {str(e)[:100]}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print()

final_links = list(all_links)

with open(output_file, 'w') as f:
    json.dump({
        'total_links': len(final_links),
        'keywords': KEYWORDS,
        'links': final_links
    }, f, indent=2)

print(f"✅ Total scraped: {len(final_links):,} links")
print(f"📄 Saved to: {output_file}")
print()

if len(final_links) > 0:
    print("✅ SUCCESS - Ready to download")
else:
    print("❌ FAILED - No links found")
    print()
    print("Troubleshooting:")
    print("1. Install dependencies: pip install selenium yt-dlp requests")
    print("2. Check internet connection")
    print("3. Try running with --headless=false to see browser")

print("="*80)
