#!/usr/bin/env python3
"""
Scrape MULTIPLE PLATFORMS for non-violence CCTV footage
Dailymotion, Vimeo, Pexels, Pixabay, Archive.org
"""

import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup

output_file = Path("multiplatform_nonviolence_links.json")

print("="*80)
print("MULTI-PLATFORM NON-VIOLENCE CCTV SCRAPER")
print("Dailymotion, Vimeo, Pexels, Pixabay, Archive.org")
print("="*80)
print()

all_links = {
    'dailymotion': [],
    'vimeo': [],
    'pexels': [],
    'pixabay': [],
    'archive': [],
}

# Setup Chrome
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

# Keywords for normal CCTV
keywords = [
    "cctv normal",
    "surveillance footage",
    "security camera",
    "normal activity",
    "public space",
]

# ============================================================================
# 1. DAILYMOTION
# ============================================================================
print("\n" + "="*80)
print("1. DAILYMOTION")
print("="*80)

try:
    driver = webdriver.Chrome(options=chrome_options)

    for keyword in keywords[:3]:  # Limit keywords
        print(f"Searching: '{keyword}'")

        search_url = f"https://www.dailymotion.com/search/{keyword.replace(' ', '%20')}/videos"
        driver.get(search_url)
        time.sleep(3)

        # Scroll to load
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # Get links
        video_links = driver.find_elements(By.CSS_SELECTOR, "a[data-testid='video-title']")

        for link in video_links:
            href = link.get_attribute('href')
            if href and '/video/' in href:
                all_links['dailymotion'].append(href)

        print(f"  Found {len(all_links['dailymotion'])} videos")
        time.sleep(random.uniform(2, 4))

    driver.quit()

except Exception as e:
    print(f"Dailymotion error: {e}")

print(f"✅ Dailymotion: {len(all_links['dailymotion'])} videos")

# ============================================================================
# 2. VIMEO
# ============================================================================
print("\n" + "="*80)
print("2. VIMEO")
print("="*80)

try:
    driver = webdriver.Chrome(options=chrome_options)

    for keyword in keywords[:3]:
        print(f"Searching: '{keyword}'")

        search_url = f"https://vimeo.com/search?q={keyword.replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(3)

        # Scroll
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # Get links
        video_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/']")

        for link in video_links:
            href = link.get_attribute('href')
            if href and 'vimeo.com/' in href and href.split('/')[-1].isdigit():
                all_links['vimeo'].append(href)

        print(f"  Found {len(all_links['vimeo'])} videos")
        time.sleep(random.uniform(2, 4))

    driver.quit()

except Exception as e:
    print(f"Vimeo error: {e}")

print(f"✅ Vimeo: {len(all_links['vimeo'])} videos")

# ============================================================================
# 3. PEXELS (Free stock footage)
# ============================================================================
print("\n" + "="*80)
print("3. PEXELS")
print("="*80)

pexels_keywords = ["surveillance", "security camera", "cctv", "monitoring", "public space"]

for keyword in pexels_keywords:
    print(f"Searching: '{keyword}'")

    try:
        url = f"https://www.pexels.com/search/videos/{keyword.replace(' ', '%20')}/"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')

        video_links = soup.find_all('a', href=True)

        for link in video_links:
            href = link['href']
            if '/video/' in href and 'pexels.com' in href:
                full_url = href if href.startswith('http') else f"https://www.pexels.com{href}"
                all_links['pexels'].append(full_url)

        print(f"  Found {len(all_links['pexels'])} videos")
        time.sleep(2)

    except Exception as e:
        print(f"  Error: {e}")

print(f"✅ Pexels: {len(all_links['pexels'])} videos")

# ============================================================================
# 4. PIXABAY (Free stock footage)
# ============================================================================
print("\n" + "="*80)
print("4. PIXABAY")
print("="*80)

for keyword in pexels_keywords:
    print(f"Searching: '{keyword}'")

    try:
        url = f"https://pixabay.com/videos/search/{keyword.replace(' ', '%20')}/"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')

        video_links = soup.find_all('a', href=True)

        for link in video_links:
            href = link['href']
            if '/videos/' in href and 'pixabay.com' in href:
                full_url = href if href.startswith('http') else f"https://pixabay.com{href}"
                all_links['pixabay'].append(full_url)

        print(f"  Found {len(all_links['pixabay'])} videos")
        time.sleep(2)

    except Exception as e:
        print(f"  Error: {e}")

print(f"✅ Pixabay: {len(all_links['pixabay'])} videos")

# ============================================================================
# 5. ARCHIVE.ORG (Public domain CCTV)
# ============================================================================
print("\n" + "="*80)
print("5. ARCHIVE.ORG")
print("="*80)

archive_keywords = ["surveillance", "cctv", "security footage", "public camera"]

for keyword in archive_keywords:
    print(f"Searching: '{keyword}'")

    try:
        url = f"https://archive.org/search?query={keyword.replace(' ', '+')}&mediatype=movies"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')

        video_links = soup.find_all('a', href=True)

        for link in video_links:
            href = link['href']
            if '/details/' in href:
                full_url = f"https://archive.org{href}"
                all_links['archive'].append(full_url)

        print(f"  Found {len(all_links['archive'])} videos")
        time.sleep(2)

    except Exception as e:
        print(f"  Error: {e}")

print(f"✅ Archive.org: {len(all_links['archive'])} videos")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SCRAPING COMPLETE")
print("="*80)
print()

# Remove duplicates
for platform in all_links:
    all_links[platform] = list(set(all_links[platform]))

total = sum(len(links) for links in all_links.values())

print("Results by platform:")
for platform, links in all_links.items():
    print(f"  {platform.upper()}: {len(links)} videos")

print()
print(f"TOTAL: {total:,} videos across all platforms")

# Save
with open(output_file, 'w') as f:
    json.dump(all_links, f, indent=2)

print(f"\n📄 Saved to: {output_file}")
print("="*80)
