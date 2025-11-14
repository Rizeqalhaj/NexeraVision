#!/bin/bash
#
# Download UCF Crime Dataset - Normal Videos Only
# From official university source
#

set -e

BASE_DIR="/workspace/ucf_crime_official"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

echo "================================================================================"
echo "UCF CRIME DATASET DOWNLOAD - BASH VERSION"
echo "================================================================================"
echo ""

# Install dependencies
echo "Installing download tools..."
pip install -q gdown
apt-get update -qq
apt-get install -y -qq wget curl aria2 unzip

echo ""
echo "================================================================================"
echo "DOWNLOADING DATASET"
echo "================================================================================"
echo ""

# ============================================================================
# METHOD 1: Google Drive (Primary)
# ============================================================================
echo "METHOD 1: Google Drive Download"
echo "--------------------------------------------------------------------------------"

GDRIVE_IDS=(
    "1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1"
    "1t0xR0SRZC_-eXyMRW_Mll_4s6IbGNH4I"
)

SUCCESS=0

for GDRIVE_ID in "${GDRIVE_IDS[@]}"; do
    echo "Trying Google Drive ID: $GDRIVE_ID"

    if gdown --id "$GDRIVE_ID" -O ucf_crime.zip; then
        echo "✓ Download successful!"
        SUCCESS=1
        break
    else
        echo "⚠ Failed, trying next mirror..."
    fi
done

# ============================================================================
# METHOD 2: Direct wget (Fallback)
# ============================================================================
if [ $SUCCESS -eq 0 ]; then
    echo ""
    echo "METHOD 2: Direct wget Download"
    echo "--------------------------------------------------------------------------------"

    URLS=(
        "http://www.crcv.ucf.edu/data/UCF_Crimes.zip"
        "https://webpages.charlotte.edu/~szhang16/dataset/UCF_Crimes.zip"
    )

    for URL in "${URLS[@]}"; do
        echo "Trying: $URL"

        if wget -c -O ucf_crime.zip "$URL"; then
            echo "✓ Download successful!"
            SUCCESS=1
            break
        else
            echo "⚠ Failed, trying next URL..."
        fi
    done
fi

# ============================================================================
# METHOD 3: Aria2 (Multi-connection download)
# ============================================================================
if [ $SUCCESS -eq 0 ]; then
    echo ""
    echo "METHOD 3: Aria2 Multi-connection Download"
    echo "--------------------------------------------------------------------------------"

    if aria2c -x 16 -s 16 --file-allocation=none \
        "http://www.crcv.ucf.edu/data/UCF_Crimes.zip" \
        -o ucf_crime.zip; then
        echo "✓ Download successful!"
        SUCCESS=1
    fi
fi

# ============================================================================
# CHECK IF DOWNLOAD SUCCEEDED
# ============================================================================
if [ ! -f "ucf_crime.zip" ]; then
    echo ""
    echo "================================================================================"
    echo "❌ DOWNLOAD FAILED - ALL METHODS"
    echo "================================================================================"
    echo ""
    echo "Manual download instructions:"
    echo "1. Go to: https://www.crcv.ucf.edu/projects/real-world/"
    echo "2. Click the download link"
    echo "3. Upload the file to: $BASE_DIR/ucf_crime.zip"
    echo "4. Run this script again to extract"
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

if [ -f "ucf_crime.zip" ]; then
    FILESIZE=$(stat -f%z "ucf_crime.zip" 2>/dev/null || stat -c%s "ucf_crime.zip" 2>/dev/null)
    FILESIZE_GB=$(echo "scale=2; $FILESIZE / 1024 / 1024 / 1024" | bc)

    echo "File size: ${FILESIZE_GB} GB"
    echo "Extracting (this may take several minutes)..."

    unzip -q ucf_crime.zip

    echo "✓ Extraction complete"
    echo "Deleting zip file..."
    rm ucf_crime.zip
fi

# ============================================================================
# FILTER FOR NORMAL VIDEOS ONLY
# ============================================================================
echo ""
echo "================================================================================"
echo "FILTERING - NORMAL VIDEOS ONLY"
echo "================================================================================"
echo ""

NORMAL_DIR="$BASE_DIR/normal_videos"
mkdir -p "$NORMAL_DIR"

# Find directories with "normal" or "non" in the name
echo "Searching for normal/non-violence categories..."

find . -type d | while read dir; do
    DIRNAME=$(basename "$dir")
    DIRNAME_LOWER=$(echo "$DIRNAME" | tr '[:upper:]' '[:lower:]')

    if [[ "$DIRNAME_LOWER" == *"normal"* ]] || [[ "$DIRNAME_LOWER" == *"non"* ]]; then
        echo "  Found: $DIRNAME"

        # Create output directory
        OUTPUT_CAT="$NORMAL_DIR/$DIRNAME"
        mkdir -p "$OUTPUT_CAT"

        # Copy all video files
        find "$dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) -exec cp {} "$OUTPUT_CAT/" \;
    fi
done

# ============================================================================
# COUNT VIDEOS
# ============================================================================
echo ""
echo "================================================================================"
echo "RESULTS"
echo "================================================================================"
echo ""

if [ -d "$NORMAL_DIR" ]; then
    # Count videos
    VIDEO_COUNT=$(find "$NORMAL_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) | wc -l)
    TOTAL_SIZE=$(du -sh "$NORMAL_DIR" | cut -f1)

    echo "✓ Normal videos extracted: $VIDEO_COUNT"
    echo "✓ Total size: $TOTAL_SIZE"
    echo ""

    # Show breakdown
    echo "Breakdown by category:"
    for subdir in "$NORMAL_DIR"/*; do
        if [ -d "$subdir" ]; then
            COUNT=$(find "$subdir" -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
            if [ $COUNT -gt 0 ]; then
                echo "  $(basename "$subdir"): $COUNT videos"
            fi
        fi
    done

    echo ""
    echo "📁 Saved to: $NORMAL_DIR"

    # Delete violence categories to save space
    echo ""
    echo "Deleting violence categories to save space..."

    find . -maxdepth 1 -type d ! -name "." ! -name "normal_videos" -exec rm -rf {} + 2>/dev/null || true

    echo "✓ Cleanup complete"
else
    echo "⚠ No normal videos found!"
    echo "Listing all directories found:"
    find . -maxdepth 2 -type d
fi

echo ""
echo "================================================================================"
echo "DOWNLOAD COMPLETE"
echo "================================================================================"
