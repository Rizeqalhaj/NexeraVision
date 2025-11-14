#!/bin/bash
# Phase 1: Immediate Free Downloads (20,000+ videos)
# Estimated time: 3-5 days
# Storage required: ~150-200 GB

set -e

echo "=========================================="
echo "PHASE 1: IMMEDIATE FREE DOWNLOADS"
echo "Target: 20,000+ fight videos"
echo "Estimated time: 3-5 days"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p /workspace/datasets/phase1/{kaggle,academic,archive,github}
cd /workspace/datasets/phase1

# Logging
LOG_FILE="logs/phase1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo ""

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle credentials not found!"
    echo "Please create ~/.kaggle/kaggle.json with your API key"
    echo "Get it from: https://www.kaggle.com/settings/account"
    exit 1
fi

echo "✅ Kaggle credentials found"
echo ""

# ============================================
# PART 1: KAGGLE DATASETS (10,000 videos)
# ============================================

echo "=========================================="
echo "PART 1: Kaggle Datasets (10,000 videos)"
echo "Estimated time: 2-4 hours"
echo "=========================================="
echo ""

cd kaggle

# RWF-2000 (2,000 videos) - If not already downloaded
echo "[1/8] Downloading RWF-2000 (2,000 videos)..."
if [ ! -d "rwf2000" ]; then
    kaggle datasets download -d vulamnguyen/rwf2000
    unzip -q rwf2000.zip -d rwf2000
    echo "✅ RWF-2000 complete"
else
    echo "✅ RWF-2000 already exists, skipping"
fi

# Hockey Fights (1,000 videos)
echo "[2/8] Downloading Hockey Fights (1,000 videos)..."
if [ ! -d "hockey-fights" ]; then
    kaggle datasets download -d yassershrief/hockey-fight-vidoes
    unzip -q hockey-fight-vidoes.zip -d hockey-fights
    echo "✅ Hockey Fights complete"
else
    echo "✅ Hockey Fights already exists, skipping"
fi

# Real Life Violence (2,000 videos)
echo "[3/8] Downloading Real Life Violence (2,000 videos)..."
if [ ! -d "real-life-violence" ]; then
    kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset
    unzip -q real-life-violence-situations-dataset.zip -d real-life-violence
    echo "✅ Real Life Violence complete"
else
    echo "✅ Real Life Violence already exists, skipping"
fi

# Movies Fight (1,000 videos)
echo "[4/8] Downloading Movies Fight (1,000 videos)..."
if [ ! -d "movies-fight" ]; then
    kaggle datasets download -d naveenk903/movies-fight-detection-dataset
    unzip -q movies-fight-detection-dataset.zip -d movies-fight
    echo "✅ Movies Fight complete"
else
    echo "✅ Movies Fight already exists, skipping"
fi

# CCTV-Fights (1,000 videos)
echo "[5/8] Downloading CCTV-Fights (1,000 videos)..."
if [ ! -d "cctv-fights" ]; then
    kaggle datasets download -d shreyj1729/cctv-fights-dataset
    unzip -q cctv-fights-dataset.zip -d cctv-fights
    echo "✅ CCTV-Fights complete"
else
    echo "✅ CCTV-Fights already exists, skipping"
fi

# SCVD (3,223 videos)
echo "[6/8] Downloading SCVD Smart City (3,223 videos)..."
if [ ! -d "scvd" ]; then
    kaggle datasets download -d toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
    unzip -q smartcity-cctv-violence-detection-dataset-scvd.zip -d scvd
    echo "✅ SCVD complete"
else
    echo "✅ SCVD already exists, skipping"
fi

# XD-Violence (4,754 videos)
echo "[7/8] Downloading XD-Violence (4,754 videos)..."
if [ ! -d "xd-violence" ]; then
    kaggle datasets download -d nguhaduong/xd-violence-video-dataset
    unzip -q xd-violence-video-dataset.zip -d xd-violence
    echo "✅ XD-Violence complete"
else
    echo "✅ XD-Violence already exists, skipping"
fi

# EAVDD (1,530 videos)
echo "[8/8] Downloading EAVDD (1,530 videos)..."
if [ ! -d "eavdd" ]; then
    kaggle datasets download -d arnab91/eavdd-violence
    unzip -q eavdd-violence.zip -d eavdd
    echo "✅ EAVDD complete"
else
    echo "✅ EAVDD already exists, skipping"
fi

cd ..

KAGGLE_COUNT=$(find kaggle -type f \( -name "*.avi" -o -name "*.mp4" -o -name "*.mkv" \) | wc -l)
echo "✅ Kaggle datasets complete: $KAGGLE_COUNT videos"
echo ""

# ============================================
# PART 2: ACADEMIC FREE-ACCESS (5,000 videos)
# ============================================

echo "=========================================="
echo "PART 2: Academic Free-Access (5,000 videos)"
echo "Estimated time: 1-2 hours"
echo "=========================================="
echo ""

cd academic

# UCF Crime (1,900 videos)
echo "[1/3] Downloading UCF Crime (1,900 videos)..."
if [ ! -d "ucf-crime" ]; then
    mkdir -p ucf-crime
    wget --no-check-certificate -q http://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip -O ucf-crime.zip
    unzip -q ucf-crime.zip -d ucf-crime
    echo "✅ UCF Crime complete"
else
    echo "✅ UCF Crime already exists, skipping"
fi

# VID Dataset (3,020 videos) - Requires manual download from Harvard Dataverse
echo "[2/3] VID Dataset (3,020 videos)..."
echo "⚠️  VID Dataset requires manual download:"
echo "1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N4LNZD"
echo "2. Register (free) and download VID_dataset.zip"
echo "3. Place in /workspace/datasets/phase1/academic/vid/"
echo ""

# UCF-101 (500-700 boxing videos from 13,320 total)
echo "[3/3] Downloading UCF-101 (boxing subset)..."
if [ ! -f "ucf101.rar" ]; then
    wget --no-check-certificate -q https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
    echo "✅ UCF-101 downloaded (extract boxing classes manually)"
    echo "Extract classes: Boxing, BoxingPunchingBag, BoxingSpeedBag"
else
    echo "✅ UCF-101 already downloaded"
fi

cd ..

ACADEMIC_COUNT=$(find academic -type f \( -name "*.avi" -o -name "*.mp4" -o -name "*.mkv" \) | wc -l)
echo "✅ Academic datasets complete: $ACADEMIC_COUNT videos"
echo ""

# ============================================
# PART 3: INTERNET ARCHIVE (5,000+ videos)
# ============================================

echo "=========================================="
echo "PART 3: Internet Archive (5,000+ videos)"
echo "Estimated time: 4-8 hours (size dependent)"
echo "=========================================="
echo ""

cd archive

# Check if internetarchive is installed
if ! command -v ia &> /dev/null; then
    echo "Installing Internet Archive CLI..."
    pip install -q internetarchive
fi

# Download public domain boxing matches
echo "Downloading public domain boxing matches..."
if [ ! -d "boxing" ]; then
    mkdir -p boxing
    ia search "subject:(boxing) AND mediatype:movies AND year:[1900 TO 2000]" \
      --parameters="rows=500" | \
      jq -r '.identifier' | \
      head -n 100 | \
      xargs -I {} -P 4 ia download {} --destdir=boxing/ --glob="*.mp4" 2>/dev/null || true
    echo "✅ Boxing matches downloaded"
else
    echo "✅ Boxing already exists, skipping"
fi

# Download public domain wrestling
echo "Downloading public domain wrestling..."
if [ ! -d "wrestling" ]; then
    mkdir -p wrestling
    ia search "subject:(wrestling) AND mediatype:movies AND year:[1900 TO 2000]" \
      --parameters="rows=500" | \
      jq -r '.identifier' | \
      head -n 50 | \
      xargs -I {} -P 4 ia download {} --destdir=wrestling/ --glob="*.mp4" 2>/dev/null || true
    echo "✅ Wrestling downloaded"
else
    echo "✅ Wrestling already exists, skipping"
fi

# Download martial arts films (public domain)
echo "Downloading public domain martial arts..."
if [ ! -d "martial-arts" ]; then
    mkdir -p martial-arts
    ia search "subject:(martial arts) AND mediatype:movies AND year:[1900 TO 2000]" \
      --parameters="rows=500" | \
      jq -r '.identifier' | \
      head -n 50 | \
      xargs -I {} -P 4 ia download {} --destdir=martial-arts/ --glob="*.mp4" 2>/dev/null || true
    echo "✅ Martial arts downloaded"
else
    echo "✅ Martial arts already exists, skipping"
fi

cd ..

ARCHIVE_COUNT=$(find archive -type f \( -name "*.avi" -o -name "*.mp4" -o -name "*.mkv" \) | wc -l)
echo "✅ Internet Archive complete: $ARCHIVE_COUNT videos"
echo ""

# ============================================
# SUMMARY
# ============================================

echo ""
echo "=========================================="
echo "PHASE 1 DOWNLOAD COMPLETE!"
echo "=========================================="
echo ""

TOTAL=$((KAGGLE_COUNT + ACADEMIC_COUNT + ARCHIVE_COUNT))

echo "📊 FINAL STATISTICS:"
echo "-------------------"
echo "Kaggle datasets:      $KAGGLE_COUNT videos"
echo "Academic datasets:    $ACADEMIC_COUNT videos"
echo "Internet Archive:     $ARCHIVE_COUNT videos"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TOTAL DOWNLOADED:     $TOTAL videos"
echo ""

if [ $TOTAL -ge 20000 ]; then
    echo "🎉 TARGET ACHIEVED: 20,000+ videos!"
elif [ $TOTAL -ge 15000 ]; then
    echo "✅ EXCELLENT: 15,000+ videos downloaded"
elif [ $TOTAL -ge 10000 ]; then
    echo "✅ GOOD: 10,000+ videos downloaded"
else
    echo "⚠️  Downloaded $TOTAL videos (target was 20,000)"
    echo "Check if some downloads failed or need manual intervention"
fi

echo ""
echo "📁 STORAGE USAGE:"
du -sh kaggle/ academic/ archive/ 2>/dev/null || echo "Computing storage..."
echo ""

echo "🔄 NEXT STEPS:"
echo "1. Start Phase 3 (Kinetics-700) download in parallel:"
echo "   bash /home/admin/Desktop/NexaraVision/download_phase3_kinetics.sh"
echo ""
echo "2. While Phase 3 runs, register for academic sources (Phase 2):"
echo "   - IEEE DataPort: https://ieee-dataport.org/"
echo "   - Zenodo: https://zenodo.org/"
echo ""
echo "3. Once all downloads complete, combine datasets:"
echo "   python /home/admin/Desktop/NexaraVision/combine_all_datasets.py"
echo ""
echo "=========================================="
echo "Download log saved to: $LOG_FILE"
echo "=========================================="
