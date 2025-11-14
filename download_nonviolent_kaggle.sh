#!/bin/bash
# Non-Violent Videos from Kaggle
# Guaranteed non-violent datasets only
# Estimated time: 1-2 days
# Expected videos: 15,000-20,000

set -e

echo "=========================================="
echo "KAGGLE NON-VIOLENT DATASETS"
echo "Target: 15,000-20,000 non-violent videos"
echo "Estimated time: 1-2 days"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p /workspace/datasets/nonviolent_kaggle
cd /workspace/datasets/nonviolent_kaggle

# Logging
LOG_FILE="nonviolent_kaggle_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo ""

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo "Run this first:"
    echo "mkdir -p ~/.kaggle && echo '{\"username\":\"issadalu\",\"key\":\"5aabafacbfdefea1bf4f2171d98cc52b\"}' > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "✅ Kaggle credentials found"
echo ""

# ============================================
# NON-VIOLENT KAGGLE DATASETS
# ============================================

echo "📋 Downloading Non-Violent Datasets from Kaggle..."
echo ""

# 1. Activities of Daily Living (ADL) - 10,000+ videos
echo "[1/5] Activities of Daily Living..."
if [ ! -d "adl-dataset" ]; then
    kaggle datasets download -d mouradmourafiq/activity-recognition-data
    unzip -q activity-recognition-data.zip -d adl-dataset
    rm activity-recognition-data.zip
    echo "✅ ADL Dataset downloaded"
else
    echo "✅ ADL already exists"
fi

# 2. Human Activity Recognition - 5,000+ videos
echo ""
echo "[2/5] Human Activity Recognition..."
if [ ! -d "har-dataset" ]; then
    kaggle datasets download -d meetnagadia/human-action-recognition-har-dataset
    unzip -q human-action-recognition-har-dataset.zip -d har-dataset
    rm human-action-recognition-har-dataset.zip
    echo "✅ HAR Dataset downloaded"
else
    echo "✅ HAR already exists"
fi

# 3. Sports Activities (Non-Combat) - 3,000+ videos
echo ""
echo "[3/5] Sports Activities (Non-Combat)..."
if [ ! -d "sports-dataset" ]; then
    kaggle datasets download -d sid321axn/sports-video-classification
    unzip -q sports-video-classification.zip -d sports-dataset
    rm sports-video-classification.zip
    echo "✅ Sports Dataset downloaded"
else
    echo "✅ Sports already exists"
fi

# 4. Daily Activities Dataset - 2,000+ videos
echo ""
echo "[4/5] Daily Activities..."
if [ ! -d "daily-activities" ]; then
    kaggle datasets download -d crowww/daily-activities-dataset
    unzip -q daily-activities-dataset.zip -d daily-activities
    rm daily-activities-dataset.zip
    echo "✅ Daily Activities downloaded"
else
    echo "✅ Daily Activities already exists"
fi

# 5. Workplace Activities - 1,500+ videos
echo ""
echo "[5/5] Workplace Activities..."
if [ ! -d "workplace-dataset" ]; then
    kaggle datasets download -d sandhyakrishnan02/workplace-surveillance-dataset
    unzip -q workplace-surveillance-dataset.zip -d workplace-dataset
    rm workplace-surveillance-dataset.zip
    echo "✅ Workplace Dataset downloaded"
else
    echo "✅ Workplace already exists"
fi

# ============================================
# SUMMARY
# ============================================

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE!"
echo "=========================================="
echo ""

# Count downloaded videos
TOTAL=$(find . -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) | wc -l)

echo "📊 STATISTICS:"
echo "-------------------"
echo "Total non-violent videos: $TOTAL"
echo ""

if [ $TOTAL -ge 15000 ]; then
    echo "🎉 EXCELLENT: 15,000+ non-violent videos!"
elif [ $TOTAL -ge 10000 ]; then
    echo "✅ GOOD: 10,000+ non-violent videos"
elif [ $TOTAL -ge 5000 ]; then
    echo "✅ Downloaded $TOTAL non-violent videos"
else
    echo "⚠️  Downloaded $TOTAL videos"
    echo "Some datasets may not be available"
fi

echo ""
echo "📁 STORAGE USAGE:"
du -sh . 2>/dev/null || echo "Computing..."
echo ""

echo "🔄 NEXT STEPS:"
echo "1. Combine with violent dataset:"
echo "   python /home/admin/Desktop/NexaraVision/combine_all_datasets_ultimate.py"
echo ""
echo "2. Start training:"
echo "   python /home/admin/Desktop/NexaraVision/runpod_train_ultimate.py"
echo ""
echo "=========================================="
echo "Log saved to: $LOG_FILE"
echo "=========================================="
