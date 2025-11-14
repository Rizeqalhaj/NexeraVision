#!/bin/bash
#
# Download UCF Crime from OFFICIAL UNIVERSITY LINK ONLY
# https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset
# NO GOOGLE DRIVE
#

set -e

BASE_DIR="/workspace/ucf_crime_official"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

echo "================================================================================"
echo "UCF CRIME - OFFICIAL UNIVERSITY DOWNLOAD ONLY"
echo "Source: https://visionlab.uncc.edu/"
echo "================================================================================"
echo ""

# Install tools
echo "Installing download tools..."
apt-get update -qq
apt-get install -y -qq wget curl aria2 unzip python3-pip
pip install -q beautifulsoup4 requests selenium

echo ""
echo "================================================================================"
echo "FINDING REAL DOWNLOAD LINK FROM UNIVERSITY PAGE"
echo "================================================================================"
echo ""

# Use Python to scrape the actual download link from the page
python3 - <<'PYTHON_SCRIPT'
import requests
from bs4 import BeautifulSoup
import json

url = "https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset"

print(f"Fetching page: {url}")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find download links
    download_links = []

    # Look for download buttons/links
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text().lower()

        # Check if it's a download link
        if any(keyword in text for keyword in ['download', 'dataset', 'file', 'zip']) or \
           any(ext in href.lower() for ext in ['.zip', '.tar', '.gz', '.rar']):

            # Make absolute URL
            if href.startswith('/'):
                full_url = f"https://visionlab.uncc.edu{href}"
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = f"https://visionlab.uncc.edu/{href}"

            download_links.append(full_url)
            print(f"Found link: {full_url}")

    # Save to file
    if download_links:
        with open('/workspace/ucf_crime_official/download_links.txt', 'w') as f:
            for link in download_links:
                f.write(f"{link}\n")
        print(f"\n✓ Found {len(download_links)} download link(s)")
    else:
        print("\n⚠ No direct download links found on page")
        print("Page may require login or JavaScript interaction")

        # Try to find any file server links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'crcv.ucf.edu' in href or 'webpages' in href or 'download' in href:
                print(f"Alternative: {href}")
else:
    print(f"❌ Failed to fetch page: {response.status_code}")
PYTHON_SCRIPT

echo ""
echo "================================================================================"
echo "DOWNLOADING DATASET"
echo "================================================================================"
echo ""

# Try each link found
if [ -f "download_links.txt" ]; then
    while IFS= read -r DOWNLOAD_URL; do
        echo "Trying: $DOWNLOAD_URL"

        # Try wget with follow redirects
        if wget -c --max-redirect=10 --content-disposition -O ucf_crime.zip "$DOWNLOAD_URL" 2>&1; then
            echo "✓ Download successful!"
            break
        fi

        # Try curl
        if curl -L -o ucf_crime.zip "$DOWNLOAD_URL" 2>&1; then
            echo "✓ Download successful!"
            break
        fi

        # Try aria2
        if aria2c -x 16 -s 16 --follow-torrent=false --max-connection-per-server=16 \
            -o ucf_crime.zip "$DOWNLOAD_URL" 2>&1; then
            echo "✓ Download successful!"
            break
        fi

        echo "⚠ Failed, trying next link..."

    done < download_links.txt
else
    echo "❌ No download links file found"
    echo ""
    echo "Trying known UCF server URLs..."

    # Try direct UCF server paths
    DIRECT_URLS=(
        "http://www.crcv.ucf.edu/data/UCF_Crimes.zip"
        "http://www.crcv.ucf.edu/projects/real-world/UCF_Crimes.zip"
        "https://webpages.charlotte.edu/~szhang16/dataset/UCF_Crimes.zip"
    )

    for URL in "${DIRECT_URLS[@]}"; do
        echo "Trying: $URL"

        if wget -c --tries=3 --timeout=60 -O ucf_crime.zip "$URL"; then
            echo "✓ Download successful!"
            break
        fi
    done
fi

# ============================================================================
# VERIFY DOWNLOAD
# ============================================================================
if [ ! -f "ucf_crime.zip" ] || [ ! -s "ucf_crime.zip" ]; then
    echo ""
    echo "================================================================================"
    echo "❌ DOWNLOAD FAILED"
    echo "================================================================================"
    echo ""
    echo "The university page may require:"
    echo "1. Manual registration/login"
    echo "2. Form submission"
    echo "3. JavaScript interaction"
    echo ""
    echo "Please manually download from:"
    echo "https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset"
    echo ""
    echo "Then upload to: $BASE_DIR/ucf_crime.zip"
    echo "And run: unzip ucf_crime.zip"
    echo ""
    exit 1
fi

# ============================================================================
# EXTRACT
# ============================================================================
echo ""
echo "================================================================================"
echo "EXTRACTING DATASET"
echo "================================================================================"
echo ""

FILESIZE=$(stat -c%s "ucf_crime.zip")
FILESIZE_GB=$(echo "scale=2; $FILESIZE / 1024 / 1024 / 1024" | bc)

echo "Downloaded: ${FILESIZE_GB} GB"
echo "Extracting..."

unzip -q ucf_crime.zip
rm ucf_crime.zip

echo "✓ Extraction complete"

# ============================================================================
# FILTER NORMAL VIDEOS
# ============================================================================
echo ""
echo "================================================================================"
echo "FILTERING - NORMAL VIDEOS ONLY"
echo "================================================================================"
echo ""

NORMAL_DIR="$BASE_DIR/normal_videos"
mkdir -p "$NORMAL_DIR"

find . -type d | while read dir; do
    DIRNAME=$(basename "$dir")
    DIRNAME_LOWER=$(echo "$DIRNAME" | tr '[:upper:]' '[:lower:]')

    if [[ "$DIRNAME_LOWER" == *"normal"* ]] || [[ "$DIRNAME_LOWER" == *"non"* ]]; then
        echo "  Found: $DIRNAME"
        OUTPUT_CAT="$NORMAL_DIR/$DIRNAME"
        mkdir -p "$OUTPUT_CAT"
        find "$dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) -exec cp {} "$OUTPUT_CAT/" \;
    fi
done

# Count
VIDEO_COUNT=$(find "$NORMAL_DIR" -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
TOTAL_SIZE=$(du -sh "$NORMAL_DIR" | cut -f1)

echo ""
echo "✓ Normal videos: $VIDEO_COUNT ($TOTAL_SIZE)"
echo "📁 Saved to: $NORMAL_DIR"

# Cleanup
find . -maxdepth 1 -type d ! -name "." ! -name "normal_videos" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "================================================================================"
echo "COMPLETE"
echo "================================================================================"
