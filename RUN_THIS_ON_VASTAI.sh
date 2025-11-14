#!/bin/bash
# ============================================================================
# RUN THIS ON VAST.AI - Dataset Corruption Scanner
# ============================================================================
#
# This script scans your entire dataset for corrupted videos and provides
# a complete corruption report across train/val/test splits.
#
# USAGE:
# 1. Copy clean_corrupted_videos.py to Vast.ai:
#    scp clean_corrupted_videos.py root@<vastai_ip>:/workspace/
#
# 2. SSH into Vast.ai:
#    ssh root@<vastai_ip>
#
# 3. Run this script:
#    bash RUN_THIS_ON_VASTAI.sh
#
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                  DATASET CORRUPTION SCANNER                                ║"
echo "║                  Running on Vast.ai                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're on Vast.ai (has /workspace directory)
if [ ! -d "/workspace" ]; then
    echo "❌ ERROR: /workspace directory not found"
    echo "   This script must be run on Vast.ai, not your local machine!"
    echo ""
    echo "   Please:"
    echo "   1. Copy this script to Vast.ai: scp RUN_THIS_ON_VASTAI.sh root@<vastai_ip>:/workspace/"
    echo "   2. SSH into Vast.ai: ssh root@<vastai_ip>"
    echo "   3. Run: bash /workspace/RUN_THIS_ON_VASTAI.sh"
    exit 1
fi

# Check if dataset exists
if [ ! -d "/workspace/organized_dataset" ]; then
    echo "❌ ERROR: Dataset not found at /workspace/organized_dataset"
    echo "   Please verify your dataset location and update the path in this script."
    exit 1
fi

# Check if clean_corrupted_videos.py exists
if [ ! -f "/workspace/clean_corrupted_videos.py" ]; then
    echo "❌ ERROR: clean_corrupted_videos.py not found in /workspace/"
    echo ""
    echo "   Please copy it from your local machine:"
    echo "   scp clean_corrupted_videos.py root@<vastai_ip>:/workspace/"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
python3 -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   Installing opencv-python-headless..."
    pip install --break-system-packages opencv-python-headless tqdm
    echo "   ✅ Dependencies installed"
else
    echo "   ✅ Dependencies already installed"
fi
echo ""

# Run the corruption scan
echo "🔍 Starting dataset corruption scan..."
echo "   This will scan: train, val, and test splits"
echo "   Estimated time: 10-30 minutes depending on dataset size"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

python3 /workspace/clean_corrupted_videos.py /workspace/organized_dataset

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ SCAN COMPLETE"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Review the corruption statistics above"
echo ""
echo "2. If you want to REMOVE corrupted videos and move them to a backup folder:"
echo "   python3 /workspace/clean_corrupted_videos.py /workspace/organized_dataset --remove"
echo ""
echo "3. If corruption is < 20%: Proceed with training using clean dataset"
echo "   If corruption is 20-40%: Re-download corrupted videos (recommended)"
echo "   If corruption is > 40%: Consider rebuilding dataset"
echo ""
echo "4. See CRITICAL_DATA_CORRUPTION_ANALYSIS.md for detailed action plan"
echo ""
