#!/bin/bash
#
# Download LARGE Research Violence Detection Datasets
# RWF-2000, RLVS, UCF Crime - From Google Drive/Academic Sources
# BYPASSING KAGGLE COMPLETELY
#

set -e

BASE="/workspace/research_datasets"
mkdir -p "$BASE"
cd "$BASE"

echo "================================================================================"
echo "DOWNLOADING LARGE RESEARCH VIOLENCE DATASETS"
echo "RWF-2000 (2000 videos), RLVS (2000 videos), UCF Crime (1900 videos)"
echo "================================================================================"
echo ""

# Install gdown
pip install -q gdown

# ============================================================================
# RWF-2000 - 2,000 videos (1000 fight, 1000 non-fight)
# ============================================================================
echo "================================================================================"
echo "1. RWF-2000 Dataset - 2,000 videos"
echo "================================================================================"

mkdir -p RWF-2000
cd RWF-2000

echo "Downloading RWF-2000 from Google Drive (this is BIG - ~5GB)..."

# Try multiple Google Drive IDs
gdown --id 1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1 || \
gdown --id 1-1gvPCjmMf4TH4PjLM9xTf3tqFKVBL8W || \
gdown "https://drive.google.com/uc?id=1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1" || \
echo "Failed, trying alternative..."

# If gdown fails, try direct link
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1" -O RWF-2000.zip && rm -rf /tmp/cookies.txt || echo "Wget also failed"

# Extract
for zip in *.zip; do
    if [ -f "$zip" ]; then
        echo "Extracting $zip..."
        unzip -q "$zip"
        rm "$zip"
    fi
done

# Count videos
VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo "✓ RWF-2000: $VIDEO_COUNT videos"
echo ""

cd "$BASE"

# ============================================================================
# RLVS - 2,000 videos (1000 violence, 1000 non-violence)
# ============================================================================
echo "================================================================================"
echo "2. RLVS Dataset - 2,000 videos"
echo "================================================================================"

mkdir -p RLVS
cd RLVS

echo "Downloading RLVS from Google Drive..."

# Multiple RLVS mirrors
gdown --id 1xm95qU8mB0GmRgEeT9kdPZdQVWPh_2jS || \
gdown --id 14xS8B3AOybMSTKfQIHan-Umh48FvPOnb || \
gdown "https://drive.google.com/uc?id=1xm95qU8mB0GmRgEeT9kdPZdQVWPh_2jS" || \
echo "All RLVS mirrors failed"

# Extract
for zip in *.zip; do
    if [ -f "$zip" ]; then
        echo "Extracting $zip..."
        unzip -q "$zip"
        rm "$zip"
    fi
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo "✓ RLVS: $VIDEO_COUNT videos"
echo ""

cd "$BASE"

# ============================================================================
# UCF Crime - 1,900 videos
# ============================================================================
echo "================================================================================"
echo "3. UCF Crime Dataset - 1,900 videos"
echo "================================================================================"

mkdir -p UCF-Crime
cd UCF-Crime

echo "Downloading UCF Crime from Google Drive (LARGE - ~13GB)..."

# Multiple UCF Crime mirrors
gdown --id 1t0xR0SRZC_-eXyMRW_Mll_4s6IbGNH4I || \
gdown --id 1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1 || \
echo "UCF Crime mirrors failed"

# Extract
for zip in *.zip; do
    if [ -f "$zip" ]; then
        echo "Extracting $zip..."
        unzip -q "$zip"
        rm "$zip"
    fi
done

for tar in *.tar.gz; do
    if [ -f "$tar" ]; then
        echo "Extracting $tar..."
        tar -xzf "$tar"
        rm "$tar"
    fi
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo "✓ UCF Crime: $VIDEO_COUNT videos"
echo ""

cd "$BASE"

# ============================================================================
# Hockey Fight - 1,000 videos
# ============================================================================
echo "================================================================================"
echo "4. Hockey Fight Dataset - 1,000 videos"
echo "================================================================================"

mkdir -p Hockey-Fight
cd Hockey-Fight

echo "Downloading Hockey Fight from university server..."
wget -c "http://visilab.etsii.uclm.es/personas/oscar/FightDetection/videos.zip" || echo "Server down, trying GitHub..."

# GitHub alternative
if [ ! -f "videos.zip" ]; then
    git clone --depth 1 https://github.com/seymanurakti/fight-detection-surv-dataset.git temp
    if [ -d "temp" ]; then
        find temp -name "*.mp4" -o -name "*.avi" | xargs -I {} mv {} .
        rm -rf temp
    fi
fi

# Extract
for zip in *.zip; do
    if [ -f "$zip" ]; then
        unzip -q "$zip"
        rm "$zip"
    fi
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo "✓ Hockey Fight: $VIDEO_COUNT videos"
echo ""

cd "$BASE"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "DOWNLOAD COMPLETE"
echo "================================================================================"
echo ""

for dir in RWF-2000 RLVS UCF-Crime Hockey-Fight; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
        size=$(du -sh "$dir" | cut -f1)
        echo "  ✓ $dir: $count videos ($size)"
    fi
done

echo ""
TOTAL=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
TOTAL_SIZE=$(du -sh . | cut -f1)
echo "TOTAL: $TOTAL videos ($TOTAL_SIZE)"
echo ""
echo "Saved to: $BASE"
echo "================================================================================"
