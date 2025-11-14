#!/usr/bin/env python3
"""
FIXED Multi-Platform NON-VIOLENCE Scraper
Dailymotion, Vimeo, Pexels, Pixabay - WORKING selectors
"""

import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup

output_file = Path("multiplatform_nonviolence_fixed.json")

print("="*80)
print("MULTI-PLATFORM NON-VIOLENCE SCRAPER - FIXED")
print("="*80)
print()

all_links = {
    'dailymotion': [],
    'vimeo': [],
    'pexels': [],
    'pixabay': [],
}

keywords = [
    "people walking",
    "people talking",
    "eating food",
    "cooking",
    "shopping",
    "office work",
]

# Setup Chrome
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

# ============================================================================
# 1. DAILYMOTION
# ============================================================================
print("="*80)
print("1. DAILYMOTION")
print("="*80)

try:
    driver = webdriver.Chrome(options=chrome_options)

    for keyword in keywords:
        print(f"Searching: '{keyword}'")

        search_url = f"https://www.dailymotion.com/search/{keyword.replace(' ', '%20')}"
        driver.get(search_url)
        time.sleep(3)

        # Scroll
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # Find video links - FIXED selector
        try:
            videos = driver.find_elements(By.CSS_SELECTOR, "a[href*='/video/']")

            for video in videos:
                href = video.get_attribute('href')
                if href and '/video/' in href and 'dailymotion.com' in href:
                    all_links['dailymotion'].append(href)

            # Remove duplicates
            all_links['dailymotion'] = list(set(all_links['dailymotion']))

        except Exception as e:
            print(f"  Error extracting: {str(e)[:50]}")

        time.sleep(2)

    driver.quit()
    print(f"✅ Dailymotion: {len(all_links['dailymotion'])} videos\n")

except Exception as e:
    print(f"❌ Dailymotion failed: {str(e)[:100]}\n")

# ============================================================================
# 2. VIMEO
# ============================================================================
print("="*80)
print("2. VIMEO")
print("="*80)

try:
    driver = webdriver.Chrome(options=chrome_options)

    for keyword in keywords:
        print(f"Searching: '{keyword}'")

        search_url = f"https://vimeo.com/search?q={keyword.replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(3)

        # Scroll
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # FIXED selector
        try:
            videos = driver.find_elements(By.CSS_SELECTOR, "a.iris_link[href^='/']")

            for video in videos:
                href = video.get_attribute('href')
                if href and href.startswith('/') and href[1:].split('/')[0].isdigit():
                    full_url = f"https://vimeo.com{href}"
                    all_links['vimeo'].append(full_url)

            all_links['vimeo'] = list(set(all_links['vimeo']))

        except Exception as e:
            print(f"  Error extracting: {str(e)[:50]}")

        time.sleep(2)

    driver.quit()
    print(f"✅ Vimeo: {len(all_links['vimeo'])} videos\n")

except Exception as e:
    print(f"❌ Vimeo failed: {str(e)[:100]}\n")

# ============================================================================
# 3. PEXELS (Using API)
# ============================================================================
print("="*80)
print("3. PEXELS")
print("="*80)

try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for keyword in keywords:
        print(f"Searching: '{keyword}'")

        # Pexels videos page
        url = f"https://www.pexels.com/search/videos/{keyword.replace(' ', '%20')}/"

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # FIXED selector - find video page links
        video_articles = soup.find_all('article', class_='Video')

        for article in video_articles:
            link_tag = article.find('a', href=True)
            if link_tag:
                href = link_tag['href']
                if href.startswith('/'):
                    full_url = f"https://www.pexels.com{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue

                if '/video/' in full_url:
                    all_links['pexels'].append(full_url)

        all_links['pexels'] = list(set(all_links['pexels']))
        time.sleep(2)

    print(f"✅ Pexels: {len(all_links['pexels'])} videos\n")

except Exception as e:
    print(f"❌ Pexels failed: {str(e)[:100]}\n")

# ============================================================================
# 4. PIXABAY (Already working, make better)
# ============================================================================
print("="*80)
print("4. PIXABAY")
print("="*80)

try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for keyword in keywords:
        print(f"Searching: '{keyword}'")

        url = f"https://pixabay.com/videos/search/{keyword.replace(' ', '%20')}/"

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find video links
        video_links = soup.find_all('a', href=True)

        for link in video_links:
            href = link['href']
            if '/videos/' in href and 'pixabay.com' in href or href.startswith('/videos/'):
                if href.startswith('/'):
                    full_url = f"https://pixabay.com{href}"
                else:
                    full_url = href

                # Filter out non-video pages
                if '/videos/id-' in full_url or href.split('/')[-2] == 'videos':
                    all_links['pixabay'].append(full_url)

        all_links['pixabay'] = list(set(all_links['pixabay']))
        time.sleep(2)

    print(f"✅ Pixabay: {len(all_links['pixabay'])} videos\n")

except Exception as e:
    print(f"❌ Pixabay failed: {str(e)[:100]}\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("="*80)
print("SCRAPING COMPLETE")
print("="*80)
print()

total = sum(len(links) for links in all_links.values())

print("Results by platform:")
for platform, links in all_links.items():
    print(f"  {platform.upper()}: {len(links)} videos")

print()
print(f"TOTAL: {total:,} videos")

# Save
with open(output_file, 'w') as f:
    json.dump(all_links, f, indent=2)

print(f"\n📄 Saved to: {output_file}")
print()

if total > 0:
    print("✅ SUCCESS - Ready to download")
else:
    print("❌ No videos found - check selectors may have changed")

print("="*80)
