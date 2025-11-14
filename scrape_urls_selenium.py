#!/usr/bin/env python3
"""
Selenium-based URL Scraper
Handles JavaScript-rendered content by using a real browser
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin

def scrape_video_urls_selenium(base_url, search_query, max_pages=5):
    """
    Scrape video URLs using Selenium (handles JavaScript)

    Args:
        base_url: Website base URL
        search_query: Search keywords
        max_pages: Number of pages to scrape

    Returns:
        List of video URLs
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException
    except ImportError:
        print("❌ Selenium not installed!")
        print("\nInstall with:")
        print("  pip install selenium")
        print("\nAlso need Chrome/Chromium browser and chromedriver")
        sys.exit(1)

    video_urls = []

    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run without GUI
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

    print(f"\n{'='*70}")
    print(f"Selenium Scraper (JavaScript-enabled)")
    print(f"{'='*70}")
    print(f"Website: {base_url}")
    print(f"Search: {search_query}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*70}\n")

    try:
        # Initialize driver
        print("🌐 Starting Chrome browser...")
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)

        for page in range(1, max_pages + 1):
            # Try multiple search URL patterns
            search_urls = [
                f"{base_url}/search?search={search_query}&page={page}",
                f"{base_url}/search?q={search_query}&page={page}",
                f"{base_url}/browse?search={search_query}&page={page}",
                f"{base_url}/videos?search={search_query}&page={page}",
            ]

            page_urls = []
            success = False

            for search_url in search_urls:
                try:
                    print(f"\n📄 Page {page}: {search_url}")
                    driver.get(search_url)

                    # Wait for page to load (adjust selector as needed)
                    time.sleep(3)  # Give JavaScript time to render

                    # Check if page loaded successfully
                    if "404" in driver.title.lower() or "not found" in driver.page_source.lower():
                        continue

                    success = True
                    print(f"   ✓ Page loaded, extracting URLs...")

                    # Method 1: Find all <a> tags with video-related hrefs
                    links = driver.find_elements(By.TAG_NAME, 'a')
                    for link in links:
                        try:
                            href = link.get_attribute('href')
                            if href and any(keyword in href.lower() for keyword in ['video', 'watch', '/v/', '/v?']):
                                page_urls.append(href)
                        except:
                            pass

                    # Method 2: Find <video> elements
                    videos = driver.find_elements(By.TAG_NAME, 'video')
                    for video in videos:
                        try:
                            src = video.get_attribute('src')
                            if src:
                                page_urls.append(urljoin(base_url, src))

                            # Check <source> tags inside video
                            sources = video.find_elements(By.TAG_NAME, 'source')
                            for source in sources:
                                src = source.get_attribute('src')
                                if src:
                                    page_urls.append(urljoin(base_url, src))
                        except:
                            pass

                    # Method 3: Find data-src attributes (lazy loading)
                    elements_with_data_src = driver.find_elements(By.XPATH, "//*[@data-src]")
                    for elem in elements_with_data_src:
                        try:
                            data_src = elem.get_attribute('data-src')
                            if data_src and any(ext in data_src.lower() for ext in ['.mp4', '.webm', '.avi', 'video']):
                                page_urls.append(urljoin(base_url, data_src))
                        except:
                            pass

                    # Method 4: Find data-video attributes
                    elements_with_data_video = driver.find_elements(By.XPATH, "//*[@data-video]")
                    for elem in elements_with_data_video:
                        try:
                            data_video = elem.get_attribute('data-video')
                            if data_video:
                                page_urls.append(urljoin(base_url, data_video))
                        except:
                            pass

                    # Method 5: Search for .mp4, .webm in all href/src attributes
                    all_elements = driver.find_elements(By.XPATH, "//*[@href or @src]")
                    for elem in all_elements:
                        try:
                            for attr in ['href', 'src']:
                                url = elem.get_attribute(attr)
                                if url and any(ext in url.lower() for ext in ['.mp4', '.webm', '.avi', '.mov', '.flv', '.mkv']):
                                    page_urls.append(urljoin(base_url, url))
                        except:
                            pass

                    print(f"   ✓ Found {len(page_urls)} URLs on page {page}")
                    break

                except Exception as e:
                    print(f"   ✗ Error: {e}")
                    continue

            if success:
                video_urls.extend(page_urls)
            else:
                print(f"   ✗ Could not access page {page}")

            time.sleep(2)  # Rate limiting

    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        try:
            driver.quit()
            print("\n🔒 Browser closed")
        except:
            pass

    # Remove duplicates
    video_urls = list(set(video_urls))
    print(f"\n{'='*70}")
    print(f"✓ Total unique URLs found: {len(video_urls)}")
    print(f"{'='*70}")

    return video_urls

def save_urls(urls, output_file):
    """Save URLs to text file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for url in urls:
            f.write(f"{url}\n")

    print(f"\n✓ URLs saved to: {output_path.absolute()}")

def main():
    print("="*70)
    print("Selenium URL Scraper (JavaScript-Enabled)")
    print("="*70)
    print("For websites that load videos with JavaScript\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage:")
            print("  python scrape_urls_selenium.py <base_url> <search_query> [max_pages] [output_file]")
            print()
            print("Examples:")
            print("  python scrape_urls_selenium.py 'https://example.com' 'cctv fight' 10")
            print("  python scrape_urls_selenium.py 'https://example.com' 'violence' 5 urls.txt")
            print()
            print("Requirements:")
            print("  pip install selenium")
            print("  Install Chrome/Chromium browser")
            print("  Install chromedriver (or use webdriver-manager)")
            sys.exit(0)

        base_url = sys.argv[1]
        search_query = sys.argv[2]
        max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        output_file = sys.argv[4] if len(sys.argv) > 4 else "video_urls.txt"

    else:
        # Interactive mode
        base_url = input("Enter website URL: ").strip()
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url

        search_query = input("Enter search query: ").strip()
        max_pages = int(input("Max pages (default 5): ").strip() or "5")
        output_file = input("Output file (default 'video_urls.txt'): ").strip() or "video_urls.txt"

    # Scrape URLs
    urls = scrape_video_urls_selenium(base_url, search_query, max_pages)

    if not urls:
        print("\n❌ No URLs found!")
        print("\nTroubleshooting:")
        print("1. Check if selectors need adjustment for this specific website")
        print("2. Try running without --headless to see what browser sees")
        print("3. Increase wait time if page loads slowly")
        return

    # Save to file
    save_urls(urls, output_file)

    print("\n" + "="*70)
    print("SCRAPING COMPLETE")
    print("="*70)
    print(f"✓ {len(urls)} URLs extracted and saved")
    print("="*70)

if __name__ == "__main__":
    main()
