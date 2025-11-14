#!/bin/bash
#
# Download UCF Crime from Dropbox link
#

set -e

BASE_DIR="/workspace/ucf_crime_dropbox"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

echo "================================================================================"
echo "DOWNLOADING FROM DROPBOX"
echo "================================================================================"
echo ""

# Install tools
apt-get update -qq
apt-get install -y -qq wget curl unzip

# Dropbox URL - change dl=0 to dl=1 for direct download
DROPBOX_URL="https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&dl=1"

echo "Downloading from Dropbox..."
echo "(This may take 30+ minutes for large files)"
echo ""

# Try wget
if wget --content-disposition -c "$DROPBOX_URL"; then
    echo "✓ wget download complete!"
elif wget -O dropbox_download.zip "$DROPBOX_URL"; then
    echo "✓ wget download complete!"
elif curl -L -o dropbox_download.zip "$DROPBOX_URL"; then
    echo "✓ curl download complete!"
else
    echo "❌ Download failed"
    echo ""
    echo "Trying alternative method - downloading with Python..."

    python3 - <<'PYTHON_SCRIPT'
import requests
import sys

url = "https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&dl=1"

print("Downloading with Python requests...")

try:
    response = requests.get(url, stream=True, allow_redirects=True)

    if response.status_code == 200:
        with open('dropbox_download.zip', 'wb') as f:
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    if total % (100 * 1024 * 1024) == 0:  # Every 100MB
                        print(f"Downloaded: {total / (1024**3):.2f} GB")

        print("✓ Python download complete!")
    else:
        print(f"❌ HTTP {response.status_code}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
PYTHON_SCRIPT

fi

echo ""
echo "================================================================================"
echo "EXTRACTING"
echo "================================================================================"
echo ""

# Find downloaded file
ZIPFILE=$(ls -t *.zip 2>/dev/null | head -1)

if [ -z "$ZIPFILE" ]; then
    echo "❌ No zip file found"
    echo "Files in directory:"
    ls -lh
    exit 1
fi

echo "Found: $ZIPFILE"
FILESIZE=$(stat -c%s "$ZIPFILE" 2>/dev/null || stat -f%z "$ZIPFILE" 2>/dev/null)
FILESIZE_GB=$(echo "scale=2; $FILESIZE / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "unknown")
echo "Size: ${FILESIZE_GB} GB"

echo "Extracting..."
unzip -q "$ZIPFILE"
rm "$ZIPFILE"

echo "✓ Extracted"

# Filter normal videos
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

VIDEO_COUNT=$(find "$NORMAL_DIR" -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
TOTAL_SIZE=$(du -sh "$NORMAL_DIR" 2>/dev/null | cut -f1)

echo ""
echo "✓ Normal videos: $VIDEO_COUNT ($TOTAL_SIZE)"
echo "📁 Saved to: $NORMAL_DIR"

# Delete violence categories
find . -maxdepth 1 -type d ! -name "." ! -name "normal_videos" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "================================================================================"
echo "COMPLETE"
echo "================================================================================"
