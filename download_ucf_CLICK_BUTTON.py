#!/usr/bin/env python3
"""
Click the download button on UCF Crime dataset page
https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from pathlib import Path

output_dir = Path("/workspace/ucf_crime_official")
output_dir.mkdir(exist_ok=True)

print("="*80)
print("UCF CRIME DATASET - CLICKING DOWNLOAD BUTTON")
print("="*80)
print()

# Setup Chrome to download files
chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": str(output_dir),
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

print("Opening browser...")
driver = webdriver.Chrome(options=chrome_options)

try:
    # Go to the page
    url = "https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset"
    print(f"Loading: {url}")
    driver.get(url)

    # Wait for page to load
    time.sleep(5)

    print("Looking for download button...")

    # Try multiple selectors for download button
    download_selectors = [
        "//a[contains(text(), 'Download')]",
        "//a[contains(text(), 'download')]",
        "//button[contains(text(), 'Download')]",
        "//a[contains(@href, 'download')]",
        "//a[contains(@class, 'download')]",
        "//input[@type='submit']",
        "//button[@type='submit']",
    ]

    download_clicked = False

    for selector in download_selectors:
        try:
            element = driver.find_element(By.XPATH, selector)
            print(f"Found button: {element.text if element.text else 'Submit'}")
            element.click()
            print("✓ Clicked download button!")
            download_clicked = True
            break
        except:
            continue

    if not download_clicked:
        print("⚠️  No download button found, trying all links...")

        # Get all links and click the most likely one
        all_links = driver.find_elements(By.TAG_NAME, "a")

        for link in all_links:
            href = link.get_attribute('href')
            text = link.text.lower()

            if href and ('download' in href or 'download' in text or '.zip' in href):
                print(f"Trying link: {text[:50]} -> {href}")
                link.click()
                download_clicked = True
                break

    if download_clicked:
        print("\nWaiting for download to complete...")
        print("(This may take 30+ minutes for ~13GB file)")

        # Wait for download (check for .crdownload file)
        max_wait = 7200  # 2 hours
        waited = 0

        while waited < max_wait:
            time.sleep(10)
            waited += 10

            # Check if download is complete
            downloading_files = list(output_dir.glob('*.crdownload'))
            zip_files = list(output_dir.glob('*.zip'))

            if downloading_files:
                print(f"  Downloading... ({waited}s)")
            elif zip_files:
                print(f"\n✅ Download complete: {zip_files[0].name}")
                print(f"   Size: {zip_files[0].stat().st_size / (1024**3):.2f} GB")
                break
            else:
                # No download started yet
                if waited > 60:
                    print("  No download detected after 60s, may have failed")
                    break
    else:
        print("❌ Could not find or click download button")
        print("\nPage source (first 500 chars):")
        print(driver.page_source[:500])

finally:
    driver.quit()

print("\n" + "="*80)
print("DONE")
print("="*80)
print()

# Check what we got
zip_files = list(output_dir.glob('*.zip'))
if zip_files:
    print(f"✅ Downloaded: {zip_files[0]}")
    print(f"\nTo extract, run:")
    print(f"  cd {output_dir}")
    print(f"  unzip {zip_files[0].name}")
else:
    print("❌ No zip file found")
    print(f"Check directory: {output_dir}")
