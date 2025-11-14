#!/bin/bash
# Phase 3: Kinetics-700 Fight Classes Download (23,000 videos)
# Using custom Python downloader (kinetics700_downloader.py)
# Estimated time: 12-24 hours
# Storage required: ~100-150 GB

set -e

echo "=========================================="
echo "PHASE 3: KINETICS-700 FIGHT CLASSES"
echo "Target: 23,000+ fight videos"
echo "Estimated time: 12-24 hours"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p /workspace/datasets/phase3/kinetics
cd /workspace/datasets/phase3/kinetics

# Logging
LOG_FILE="../../phase1/logs/phase3_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ../../phase1/logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo ""

# Check Python dependencies
echo "📦 Installing required packages..."
pip install -q pandas requests tqdm yt-dlp

echo ""
echo "=========================================="
echo "KINETICS-700 FIGHT-RELATED CLASSES"
echo "=========================================="
echo ""

# List of all fight/combat-related classes in Kinetics-700
echo "📋 Classes to download:"
echo "1. boxing"
echo "2. wrestling"
echo "3. punching person (boxing)"
echo "4. side kick"
echo "5. high kick"
echo "6. drop kicking"
echo "7. arm wrestling"
echo "8. capoeira"
echo "9. fencing (sport)"
echo "10. kickboxing"
echo "11. martial arts"
echo "12. slapping"
echo "13. headbutting"
echo "14. punching bag"
echo "15. sword fighting"
echo "16. tai chi"
echo "17. training with punching bag"
echo "18. catching or throwing baseball"
echo "19. catching or throwing softball"
echo ""

echo "⚠️  IMPORTANT NOTES:"
echo "- This will download from YouTube using Kinetics-700 video IDs"
echo "- Some videos may be unavailable (deleted, private, region-locked)"
echo "- Expected success rate: 60-70% (14,000-16,000 videos)"
echo "- Download time varies based on bandwidth"
echo "- Uses 8 parallel workers for faster download"
echo ""

read -p "Start Kinetics-700 download? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled"
    exit 0
fi

echo ""
echo "=========================================="
echo "STARTING DOWNLOAD"
echo "=========================================="
echo ""

# Copy custom downloader
cp /home/admin/Desktop/NexaraVision/kinetics700_downloader.py ./

# Run custom Kinetics-700 downloader
python kinetics700_downloader.py \
  --output-dir . \
  --num-workers 8

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE!"
echo "=========================================="
echo ""

# Count downloaded videos
KINETICS_COUNT=$(find . -type f -name "*.mp4" | wc -l)

echo "📊 STATISTICS:"
echo "-------------------"
echo "Videos downloaded: $KINETICS_COUNT"
echo "Target was: 23,000 videos"
if [ $KINETICS_COUNT -gt 0 ]; then
    echo "Success rate: $(echo "scale=1; $KINETICS_COUNT * 100 / 23000" | bc)%"
fi
echo ""

if [ $KINETICS_COUNT -ge 15000 ]; then
    echo "🎉 EXCELLENT: 15,000+ videos downloaded!"
elif [ $KINETICS_COUNT -ge 10000 ]; then
    echo "✅ GOOD: 10,000+ videos downloaded"
elif [ $KINETICS_COUNT -ge 5000 ]; then
    echo "⚠️  Downloaded $KINETICS_COUNT videos (expected 60-70% success rate)"
else
    echo "⚠️  Downloaded $KINETICS_COUNT videos"
    echo "This may be due to:"
    echo "- Network issues"
    echo "- YouTube availability"
    echo "- Region restrictions"
fi

echo ""
echo "📁 STORAGE USAGE:"
du -sh . 2>/dev/null || echo "Computing storage..."
echo ""

echo "🔄 NEXT STEPS:"
echo "1. Combine all datasets (Phase 1 + Phase 3):"
echo "   python /home/admin/Desktop/NexaraVision/combine_all_datasets_ultimate.py"
echo ""
echo "2. Start ultimate training with combined dataset:"
echo "   python /home/admin/Desktop/NexaraVision/runpod_train_ultimate.py"
echo ""
echo "=========================================="
echo "Download log saved to: $LOG_FILE"
echo "=========================================="
