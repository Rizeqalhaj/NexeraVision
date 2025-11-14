#!/bin/bash
#
# Download Violence Detection Datasets - ALTERNATIVE SOURCES
# NO KAGGLE - Direct downloads from academic/public sources
# For Vast.ai headless environments
#

set -e

BASE_DIR="/workspace/violence_datasets"
mkdir -p "$BASE_DIR"

echo "================================================================================"
echo "DOWNLOADING VIOLENCE DETECTION DATASETS - ALTERNATIVE SOURCES"
echo "NO Kaggle required - Direct academic/public downloads"
echo "================================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Install tools
echo -e "${BLUE}Installing download tools...${NC}"
pip install -q gdown
apt-get update -qq
apt-get install -y -qq git-lfs unzip wget curl aria2
echo -e "${GREEN}✓ Tools installed${NC}"
echo ""

# ============================================================================
# 1. RWF-2000 - Google Drive (2,000 videos)
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}1. RWF-2000 (Real World Fight)${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$BASE_DIR"
mkdir -p RWF-2000
cd RWF-2000

echo "Downloading from Google Drive..."
# Multiple mirror IDs to try
gdown --fuzzy "1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1" || \
gdown --fuzzy "1-1gvPCjmMf4TH4PjLM9xTf3tqFKVBL8W" || \
echo "Primary mirrors failed, trying alternative..."

# Extract
for zip in *.zip; do
    [ -f "$zip" ] && unzip -q -o "$zip" && rm "$zip"
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo -e "${GREEN}✓ RWF-2000: $VIDEO_COUNT videos${NC}"
echo ""

# ============================================================================
# 2. UCF Crime - Multiple sources (1,900 videos)
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}2. UCF Crime Dataset${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$BASE_DIR"
mkdir -p UCF-Crime
cd UCF-Crime

echo "Trying multiple UCF Crime sources..."

# Try Google Drive mirrors
gdown --fuzzy "1t0xR0SRZC_-eXyMRW_Mll_4s6IbGNH4I" || \
gdown --fuzzy "1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1" || \
echo "Google Drive mirrors exhausted"

# Try academic mirror
wget -c -t 3 "https://webpages.charlotte.edu/~szhang16/dataset/UCF_Crimes.zip" || \
echo "Academic mirror failed"

# Extract all zips
for zip in *.zip; do
    [ -f "$zip" ] && unzip -q -o "$zip" && rm "$zip"
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo -e "${GREEN}✓ UCF Crime: $VIDEO_COUNT videos${NC}"
echo ""

# ============================================================================
# 3. Hockey Fight - Direct download (1,000 videos)
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}3. Hockey Fight Dataset${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$BASE_DIR"
mkdir -p Hockey-Fight
cd Hockey-Fight

echo "Downloading from university server..."
wget -c -t 5 "http://visilab.etsii.uclm.es/personas/oscar/FightDetection/videos.zip" || \
echo "Primary source failed, trying mirrors..."

# Try GitHub LFS alternative
git clone --depth 1 https://github.com/seymanurakti/fight-detection-surv-dataset.git temp_clone || true
if [ -d "temp_clone" ]; then
    cd temp_clone
    git lfs pull
    cd ..
    find temp_clone -type f \( -name "*.mp4" -o -name "*.avi" \) -exec mv {} . \;
    rm -rf temp_clone
fi

# Extract zips
for zip in *.zip; do
    [ -f "$zip" ] && unzip -q -o "$zip" && rm "$zip"
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo -e "${GREEN}✓ Hockey Fight: $VIDEO_COUNT videos${NC}"
echo ""

# ============================================================================
# 4. RLVS - Google Drive (2,000 videos)
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}4. RLVS Dataset${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$BASE_DIR"
mkdir -p RLVS
cd RLVS

echo "Downloading RLVS from Google Drive..."
gdown --fuzzy "1xm95qU8mB0GmRgEeT9kdPZdQVWPh_2jS" || \
gdown --fuzzy "14xS8B3AOybMSTKfQIHan-Umh48FvPOnb" || \
echo "All RLVS mirrors failed"

# Extract
for zip in *.zip; do
    [ -f "$zip" ] && unzip -q -o "$zip" && rm "$zip"
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo -e "${GREEN}✓ RLVS: $VIDEO_COUNT videos${NC}"
echo ""

# ============================================================================
# 5. Violent Flows (250 videos)
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}5. Violent Flows${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$BASE_DIR"
mkdir -p Violent-Flows
cd Violent-Flows

echo "Downloading from Open University..."
wget -c "https://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.tar.gz" || \
echo "Primary source failed"

# Extract
for tar in *.tar.gz; do
    [ -f "$tar" ] && tar -xzf "$tar" && rm "$tar"
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo -e "${GREEN}✓ Violent Flows: $VIDEO_COUNT videos${NC}"
echo ""

# ============================================================================
# 6. GitHub Datasets with LFS
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}6. GitHub LFS Datasets${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$BASE_DIR"

# AIRTLab
echo "Cloning AIRTLab dataset..."
if [ ! -d "AIRTLab" ]; then
    git clone --depth 1 https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git AIRTLab
    cd AIRTLab
    git lfs pull 2>/dev/null || echo "No LFS files"
    cd "$BASE_DIR"
fi
VIDEO_COUNT=$(find AIRTLab -type f \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | wc -l)
echo -e "${GREEN}✓ AIRTLab: $VIDEO_COUNT videos${NC}"

# VioPeru
echo "Cloning VioPeru dataset..."
if [ ! -d "VioPeru" ]; then
    git clone --depth 1 https://github.com/hhuillcen/VioPeru.git
    cd VioPeru
    git lfs pull 2>/dev/null || echo "No LFS files"
    cd "$BASE_DIR"
fi
VIDEO_COUNT=$(find VioPeru -type f \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | wc -l)
echo -e "${GREEN}✓ VioPeru: $VIDEO_COUNT videos${NC}"

# UCA (CVPR 2024)
echo "Cloning UCA dataset..."
if [ ! -d "UCA" ]; then
    git clone --depth 1 https://github.com/Xuange923/Surveillance-Video-Understanding.git UCA
    cd UCA
    git lfs pull 2>/dev/null || echo "No LFS files"
    cd "$BASE_DIR"
fi
VIDEO_COUNT=$(find UCA -type f \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | wc -l)
echo -e "${GREEN}✓ UCA: $VIDEO_COUNT videos${NC}"

echo ""

# ============================================================================
# 7. UCF-101 (Subset with violence classes)
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}7. UCF-101 Subset${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$BASE_DIR"
mkdir -p UCF-101
cd UCF-101

echo "Downloading UCF-101 violence classes..."
# Download only violence-related classes to save space
for class in "Boxing" "PunchingBag" "Fencing" "SwordExercise" "Hammering" "HammerThrow"; do
    echo "  Downloading $class..."
    wget -q -c "https://www.crcv.ucf.edu/data/UCF101/${class}.rar" 2>/dev/null || echo "  $class not available"
done

# Extract rars
for rar in *.rar; do
    [ -f "$rar" ] && (unrar x -o+ "$rar" 2>/dev/null || 7z x -y "$rar") && rm "$rar"
done

VIDEO_COUNT=$(find . -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
echo -e "${GREEN}✓ UCF-101 subset: $VIDEO_COUNT videos${NC}"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DOWNLOAD COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

cd "$BASE_DIR"

echo "Dataset Summary:"
for dir in */; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) | wc -l)
        size=$(du -sh "$dir" | cut -f1)
        echo "  ✓ ${dir%/}: $count videos ($size)"
    fi
done

echo ""
TOTAL_VIDEOS=$(find . -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) | wc -l)
TOTAL_SIZE=$(du -sh . | cut -f1)

echo -e "${YELLOW}Total Videos: $TOTAL_VIDEOS${NC}"
echo -e "${YELLOW}Total Size: $TOTAL_SIZE${NC}"
echo ""
echo -e "${BLUE}All datasets saved to: $BASE_DIR${NC}"
echo ""
