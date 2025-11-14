#!/usr/bin/env python3
"""
Debug script to see the actual HTML structure of the page
This helps us understand how videos are embedded
"""

import requests
from bs4 import BeautifulSoup
import sys

def analyze_page(url):
    """Analyze page HTML structure"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"\n{'='*70}")
    print(f"Analyzing: {url}")
    print(f"{'='*70}\n")

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"✗ Error: Status code {response.status_code}")
            return

        print(f"✓ Page loaded successfully (Status 200)")
        print(f"✓ Content length: {len(response.content)} bytes\n")

        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Check for all <a> tags
        all_links = soup.find_all('a', href=True)
        print(f"{'='*70}")
        print(f"Found {len(all_links)} total <a> tags")
        print(f"{'='*70}")
        if all_links:
            print("\nFirst 10 <a> tag hrefs:")
            for i, link in enumerate(all_links[:10], 1):
                print(f"  {i}. {link.get('href')}")

        # 2. Check for <video> tags
        videos = soup.find_all('video')
        print(f"\n{'='*70}")
        print(f"Found {len(videos)} <video> tags")
        print(f"{'='*70}")
        if videos:
            for i, video in enumerate(videos[:5], 1):
                print(f"\nVideo {i}:")
                print(f"  src: {video.get('src')}")
                print(f"  poster: {video.get('poster')}")
                sources = video.find_all('source')
                for j, source in enumerate(sources, 1):
                    print(f"  source {j}: {source.get('src')}")

        # 3. Check for common video container classes/ids
        print(f"\n{'='*70}")
        print("Searching for common video container patterns:")
        print(f"{'='*70}")

        patterns = [
            ('div', 'class', 'video'),
            ('div', 'class', 'item'),
            ('div', 'class', 'post'),
            ('div', 'class', 'card'),
            ('div', 'id', 'video'),
            ('article', None, None),
        ]

        for tag, attr, value in patterns:
            if attr and value:
                elements = soup.find_all(tag, {attr: lambda x: x and value in x.lower() if x else False})
            elif attr is None:
                elements = soup.find_all(tag)
            else:
                elements = soup.find_all(tag)

            if elements:
                print(f"\n✓ Found {len(elements)} <{tag}> with '{value}' in {attr}")
                # Show first element structure
                if len(elements) > 0:
                    print(f"\nFirst element structure:")
                    print(f"  {elements[0].prettify()[:500]}...")

        # 4. Look for data attributes (common in JS frameworks)
        print(f"\n{'='*70}")
        print("Looking for data-* attributes (JS frameworks):")
        print(f"{'='*70}")

        data_attrs = soup.find_all(attrs={"data-src": True})
        print(f"\n✓ Found {len(data_attrs)} elements with data-src")
        for i, elem in enumerate(data_attrs[:5], 1):
            print(f"  {i}. data-src: {elem.get('data-src')}")

        data_video = soup.find_all(attrs={"data-video": True})
        print(f"\n✓ Found {len(data_video)} elements with data-video")
        for i, elem in enumerate(data_video[:5], 1):
            print(f"  {i}. data-video: {elem.get('data-video')}")

        data_url = soup.find_all(attrs={"data-url": True})
        print(f"\n✓ Found {len(data_url)} elements with data-url")
        for i, elem in enumerate(data_url[:5], 1):
            print(f"  {i}. data-url: {elem.get('data-url')}")

        # 5. Search for .mp4, .webm in page source
        print(f"\n{'='*70}")
        print("Searching for video file extensions in page source:")
        print(f"{'='*70}")

        page_text = str(soup)
        extensions = ['.mp4', '.webm', '.avi', '.mov', '.flv', '.mkv']

        for ext in extensions:
            if ext in page_text.lower():
                print(f"\n✓ Found '{ext}' in page source")
                # Find surrounding context
                import re
                matches = re.finditer(rf'[\w/\-\.]+{ext}', page_text, re.IGNORECASE)
                for i, match in enumerate(list(matches)[:3], 1):
                    print(f"  {i}. {match.group()}")

        # 6. Save full HTML for manual inspection
        output_file = "page_debug.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())

        print(f"\n{'='*70}")
        print(f"✓ Full HTML saved to: {output_file}")
        print(f"  You can open this in a text editor to inspect manually")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_page_structure.py <url>")
        print("\nExample:")
        print("  python debug_page_structure.py 'https://example.com/search?search=fight&page=1'")
        sys.exit(1)

    url = sys.argv[1]
    analyze_page(url)

if __name__ == "__main__":
    main()
