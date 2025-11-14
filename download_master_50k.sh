#!/bin/bash
# MASTER DOWNLOAD SCRIPT - 50K+ Violence Detection Videos
# Run on cloud GPU with 2TB storage
# Estimated time: 3-7 days depending on bandwidth

set -e

echo "=========================================="
echo "VIOLENCE DETECTION DATASET DOWNLOADER"
echo "Target: 50,000+ videos"
echo "Storage Required: 1.5-2 TB"
echo "=========================================="

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle credentials not found!"
    echo "Create ~/.kaggle/kaggle.json with your API key"
    exit 1
fi

# Create directory structure
mkdir -p /workspace/datasets/{tier1,tier2,tier3,raw,processed,logs}
cd /workspace/datasets

# Logging
LOG_FILE="logs/download_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo ""

# ============================================
# TIER 1: PRIORITY DATASETS (18-20K videos)
# ============================================

echo "=========================================="
echo "PHASE 1: Tier 1 Downloads (18-20K videos)"
echo "Estimated time: 4-8 hours"
echo "=========================================="

cd tier1

# XD-Violence (4,754 videos)
echo "[1/9] Downloading XD-Violence (4,754 videos)..."
kaggle datasets download -d nguhaduong/xd-violence-video-dataset
kaggle datasets download -d bhavay192/xd-violence-1005-2004-set
kaggle datasets download -d bhavay192/xd-violence-train-2805-3319-set

# RWF-2000 (2,000 videos)
echo "[2/9] Downloading RWF-2000 (2,000 videos)..."
kaggle datasets download -d vulamnguyen/rwf2000
wget -q https://zenodo.org/records/15687512/files/RWF-2000.zip

# Real Life Violence (2,000 videos)
echo "[3/9] Downloading Real Life Violence Situations (2,000 videos)..."
kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset

# SCVD (3,223 videos)
echo "[4/9] Downloading SCVD Smart-City CCTV (3,223 videos)..."
kaggle datasets download -d toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd

# UCF-Crime (1,900 videos)
echo "[5/9] Downloading UCF-Crime (1,900 videos)..."
kaggle datasets download -d odins0n/ucf-crime-dataset
# Backup: wget http://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip

# EAVDD (1,530 videos)
echo "[6/9] Downloading EAVDD (1,530 videos)..."
kaggle datasets download -d arnab91/eavdd-violence

# Bus Violence (1,400 videos)
echo "[7/9] Downloading Bus Violence (1,400 videos)..."
wget -q https://zenodo.org/records/7044203/files/BusViolence.zip

# VID Dataset (3,020 videos)
echo "[8/9] Downloading VID Dataset (3,020 videos)..."
wget -q https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/N4LNZD -O VID_dataset.zip

# Godseye Fusion (3,350 videos - if not already covered)
echo "[9/9] Checking HuggingFace datasets..."
python3 << 'PYTHON_EOF'
from datasets import load_dataset
try:
    dataset = load_dataset("jherng/xd-violence")
    print("✅ XD-Violence HuggingFace mirror available")
except:
    print("⚠️  XD-Violence HuggingFace download failed, using Kaggle version")
PYTHON_EOF

echo "Extracting Tier 1 datasets..."
for file in *.zip; do
    echo "Extracting $file..."
    unzip -q "$file" -d "${file%.zip}" 2>/dev/null || echo "⚠️  $file extraction had warnings"
done

cd ..

echo "✅ Tier 1 complete!"
TIER1_COUNT=$(find tier1 -type f \( -name "*.avi" -o -name "*.mp4" -o -name "*.mkv" \) | wc -l)
echo "Tier 1 videos: $TIER1_COUNT"

# ============================================
# TIER 2: KINETICS-700 (15-20K videos)
# ============================================

echo ""
echo "=========================================="
echo "PHASE 2: Kinetics-700 Violence Classes"
echo "Target: 23,000 videos (expect ~16,000 success)"
echo "Estimated time: 12-24 hours"
echo "=========================================="

# Install dependencies
pip install -q kinetics-downloader yt-dlp tqdm

cd tier2

# Kinetics-700 violence classes
echo "Downloading Kinetics-700 violence/fighting classes..."
echo "This will take 12-24 hours. Use screen/tmux to keep running."

kinetics-downloader download \
  --version 700 \
  --classes "boxing,wrestling,punching person (boxing),side kick,high kick,drop kicking,arm wrestling,capoeira,fencing (sport),high jump,javelin throw,jumpstyle dancing,kicking field goal,kicking soccer ball,kickboxing,long jump,martial arts,playing kickball,pole vault,spinning poi,stretching arm,triple jump,tai chi" \
  --output-dir kinetics_violence/ \
  --num-workers 8 \
  --trim-format "%06d" \
  --verbose 2>&1 | tee kinetics_download.log

cd ..

echo "✅ Tier 2 complete!"
TIER2_COUNT=$(find tier2 -type f -name "*.mp4" | wc -l)
echo "Tier 2 videos: $TIER2_COUNT"

# ============================================
# TIER 3: SUPPLEMENTARY (8-10K videos)
# ============================================

echo ""
echo "=========================================="
echo "PHASE 3: Supplementary Datasets (8-10K)"
echo "Estimated time: 2-4 hours"
echo "=========================================="

cd tier3

# Hockey Fights (1,000 videos)
echo "[1/8] Downloading Hockey Fights (1,000 videos)..."
kaggle datasets download -d yassershrief/hockey-fight-vidoes
# Backup: aria2c -x 16 https://academictorrents.com/download/38d9ed996a5a75a039b84cf8a137be794e7cee89.torrent

# Movies Fight (1,000 videos)
echo "[2/8] Downloading Movies Fight Dataset (1,000 videos)..."
kaggle datasets download -d naveenk903/movies-fight-detection-dataset

# CCTV-Fights (1,000 videos)
echo "[3/8] Downloading CCTV-Fights (1,000 videos)..."
kaggle datasets download -d shreyj1729/cctv-fights-dataset

# UCF-101 (full dataset - extract violence classes later)
echo "[4/8] Downloading UCF-101 (13,320 total, ~700 violence)..."
wget -q https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
if command -v unrar &> /dev/null; then
    unrar x UCF101.rar
else
    echo "⚠️  unrar not installed, skipping UCF-101 extraction"
fi

# AIRTLab (350 videos)
echo "[5/8] Cloning AIRTLab Dataset (350 videos)..."
git clone --depth 1 https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos airtlab

# Surveillance Camera Fight (300 videos)
echo "[6/8] Cloning Surveillance Fight Dataset (300 videos)..."
git clone --depth 1 https://github.com/seymanurakti/fight-detection-surv-dataset surveillance-fight

# Weapons Detection
echo "[7/8] Downloading Weapons in CCTV..."
kaggle datasets download -d kruthisb999/guns-and-knifes-detection-in-cctv-videos

# Violence-Combined (various)
echo "[8/8] Downloading Violence-Combined..."
kaggle datasets download -d yash07yadav/project-data

echo "Extracting Tier 3 datasets..."
for file in *.zip; do
    echo "Extracting $file..."
    unzip -q "$file" -d "${file%.zip}" 2>/dev/null || echo "⚠️  $file extraction had warnings"
done

cd ..

echo "✅ Tier 3 complete!"
TIER3_COUNT=$(find tier3 -type f \( -name "*.avi" -o -name "*.mp4" -o -name "*.mkv" \) | wc -l)
echo "Tier 3 videos: $TIER3_COUNT"

# ============================================
# SUMMARY & STATISTICS
# ============================================

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE!"
echo "=========================================="

TOTAL=$((TIER1_COUNT + TIER2_COUNT + TIER3_COUNT))

echo ""
echo "📊 FINAL STATISTICS:"
echo "-------------------"
echo "Tier 1 (Priority):      $TIER1_COUNT videos"
echo "Tier 2 (Kinetics):      $TIER2_COUNT videos"
echo "Tier 3 (Supplementary): $TIER3_COUNT videos"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TOTAL DOWNLOADED:       $TOTAL videos"
echo ""

if [ $TOTAL -ge 50000 ]; then
    echo "🎉 TARGET ACHIEVED: 50,000+ videos!"
elif [ $TOTAL -ge 40000 ]; then
    echo "✅ EXCELLENT: 40,000+ videos (close to target)"
    echo "💡 Recommendation: Run augmentation to reach 50K"
elif [ $TOTAL -ge 30000 ]; then
    echo "✅ GOOD: 30,000+ videos"
    echo "💡 Recommendation: Add more sources or augment 2x"
else
    echo "⚠️  Below target. Downloaded: $TOTAL videos"
    echo "💡 Recommendation: Check Kinetics download success rate"
fi

echo ""
echo "📁 STORAGE USAGE:"
du -sh tier1/ tier2/ tier3/ 2>/dev/null || echo "Computing storage..."
echo ""

echo "🔄 NEXT STEPS:"
echo "1. Organize dataset:"
echo "   python /home/admin/Desktop/NexaraVision/combine_all_datasets.py"
echo ""
echo "2. Start ultimate training:"
echo "   python /home/admin/Desktop/NexaraVision/runpod_train_ultimate.py"
echo ""
echo "3. Monitor training:"
echo "   tail -f training_ultimate_*.log"
echo ""
echo "=========================================="
echo "Download log saved to: $LOG_FILE"
echo "=========================================="
