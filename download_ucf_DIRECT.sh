#!/bin/bash
#
# Direct download from UCF Crime URL - NO BUTTON, AUTO DOWNLOAD
#

set -e

BASE_DIR="/workspace/ucf_crime_official"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

URL="https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset"

echo "================================================================================"
echo "UCF CRIME DATASET - DIRECT DOWNLOAD"
echo "URL auto-downloads when visited"
echo "================================================================================"
echo ""

# Install tools
apt-get update -qq
apt-get install -y -qq wget curl aria2 unzip

echo "Downloading from: $URL"
echo "(This may take 30+ minutes for ~13GB)"
echo ""

# Try wget with content-disposition to get real filename
if wget --content-disposition --max-redirect=10 -c "$URL"; then
    echo "✓ Download complete!"
elif wget -O ucf_crime_dataset.zip "$URL"; then
    echo "✓ Download complete!"
elif curl -L -o ucf_crime_dataset.zip "$URL"; then
    echo "✓ Download complete with curl!"
elif aria2c -x 16 "$URL"; then
    echo "✓ Download complete with aria2!"
else
    echo "❌ All download methods failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "EXTRACTING"
echo "================================================================================"
echo ""

# Find the downloaded file
ZIPFILE=$(ls -t *.zip 2>/dev/null | head -1)

if [ -z "$ZIPFILE" ]; then
    echo "❌ No zip file found"
    ls -lh
    exit 1
fi

echo "Found: $ZIPFILE"
FILESIZE=$(stat -c%s "$ZIPFILE" 2>/dev/null || stat -f%z "$ZIPFILE" 2>/dev/null)
FILESIZE_GB=$(echo "scale=2; $FILESIZE / 1024 / 1024 / 1024" | bc)
echo "Size: ${FILESIZE_GB} GB"

echo "Extracting..."
unzip -q "$ZIPFILE"
rm "$ZIPFILE"

echo "✓ Extracted"

# Filter normal videos only
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
