#!/bin/bash
# Download Additional Violent Datasets to Balance Dataset
# Target: ~15,000 additional violent videos to balance with non-violent
# Current: ~5,500 violent + ~17,400 non-violent = IMBALANCED
# Goal: ~20,000 violent + ~17,400 non-violent = BALANCED

set -e

echo "=========================================="
echo "ADDITIONAL VIOLENT DATASETS - PHASE 2"
echo "Target: +15,000 violent videos"
echo "Current violent: ~5,500"
echo "Current non-violent: ~17,400"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p /workspace/datasets/violent_phase2
cd /workspace/datasets/violent_phase2

# Logging
LOG_FILE="violent_phase2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo ""

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo "Run: mkdir -p ~/.kaggle && echo '{\"username\":\"issadalu\",\"key\":\"5aabafacbfdefea1bf4f2171d98cc52b\"}' > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "✅ Kaggle credentials found"
echo ""

# ============================================
# VERIFIED VIOLENT-ONLY KAGGLE DATASETS
# (These need to be verified manually on Kaggle first)
# ============================================

echo "📋 PHASE 2 - Additional Violent Datasets"
echo ""
echo "⚠️  IMPORTANT: You must manually verify these datasets exist on Kaggle.com"
echo "⚠️  Search each dataset name on https://www.kaggle.com/datasets"
echo "⚠️  Replace with confirmed working dataset names"
echo ""
echo "Current script contains PLACEHOLDER dataset names"
echo "You need to:"
echo "  1. Run: bash search_additional_violent_kaggle.sh"
echo "  2. Find real dataset names from the search results"
echo "  3. Edit this file and replace PLACEHOLDER_USER/PLACEHOLDER_DATASET"
echo "  4. Run this script again"
echo ""

# ============================================
# EXAMPLE DOWNLOAD COMMANDS (UPDATE WITH REAL DATASETS)
# ============================================

# Example 1: Fight Detection Dataset (REPLACE WITH REAL NAME)
# echo "[1/5] Fight Detection Dataset..."
# if [ ! -d "fight-dataset" ]; then
#     kaggle datasets download -d PLACEHOLDER_USER/fight-detection-dataset
#     unzip -q fight-detection-dataset.zip -d fight-dataset
#     rm fight-detection-dataset.zip
#     echo "✅ Downloaded"
# else
#     echo "✅ Already exists"
# fi

# Example 2: UFC/MMA Dataset (REPLACE WITH REAL NAME)
# echo ""
# echo "[2/5] UFC/MMA Dataset..."
# if [ ! -d "ufc-mma-dataset" ]; then
#     kaggle datasets download -d PLACEHOLDER_USER/ufc-mma-dataset
#     unzip -q ufc-mma-dataset.zip -d ufc-mma-dataset
#     rm ufc-mma-dataset.zip
#     echo "✅ Downloaded"
# else
#     echo "✅ Already exists"
# fi

# ============================================
# ALTERNATIVE: YOUTUBE-DLP APPROACH FOR MORE DATA
# ============================================

echo ""
echo "=========================================="
echo "ALTERNATIVE: YOUTUBE-DLP DOWNLOADS"
echo "=========================================="
echo ""
echo "Since Kaggle has limited datasets, using YouTube-DL for additional violent videos"
echo "This is reliable and gets you closer to 100K target"
echo ""

# Download fight videos from YouTube (safer batched approach)
if [ ! -f "/home/admin/Desktop/NexaraVision/download_youtube_fights_fast.py" ]; then
    echo "❌ YouTube downloader not found!"
    echo "Expected: /home/admin/Desktop/NexaraVision/download_youtube_fights_fast.py"
    exit 1
fi

echo "🎬 Downloading additional violent videos from YouTube..."
echo "Target: 15,000 additional videos"
echo ""

python3 /home/admin/Desktop/NexaraVision/download_youtube_fights_fast.py \
    --output-dir /workspace/datasets/violent_phase2/youtube_fights \
    --videos-per-query 500 \
    --queries 20

echo ""
echo "=========================================="
echo "PHASE 2 DOWNLOAD COMPLETE!"
echo "=========================================="
echo ""

# Count total violent videos now
PHASE1_COUNT=$(find /workspace/datasets/violent_phase1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)
PHASE2_COUNT=$(find /workspace/datasets/violent_phase2 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)
TOTAL_VIOLENT=$((PHASE1_COUNT + PHASE2_COUNT))

echo "📊 VIOLENT VIDEO STATISTICS:"
echo "----------------------------"
echo "Phase 1 (Kaggle mixed): $PHASE1_COUNT"
echo "Phase 2 (Additional): $PHASE2_COUNT"
echo "Total Violent Videos: $TOTAL_VIOLENT"
echo ""

if [ $TOTAL_VIOLENT -ge 15000 ]; then
    echo "🎉 EXCELLENT: $TOTAL_VIOLENT violent videos!"
elif [ $TOTAL_VIOLENT -ge 10000 ]; then
    echo "✅ GOOD: $TOTAL_VIOLENT violent videos"
else
    echo "⚠️  Need more violent videos: $TOTAL_VIOLENT (target: 15,000+)"
fi

echo ""
echo "🔄 NEXT STEP: Run dataset balancing script"
echo "   bash /home/admin/Desktop/NexaraVision/balance_and_combine.sh"
echo ""
echo "=========================================="
echo "Log saved to: $LOG_FILE"
echo "=========================================="
