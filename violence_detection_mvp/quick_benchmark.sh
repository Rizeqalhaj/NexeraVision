#!/bin/bash

echo "=============================================================================="
echo "🔍 FINDING YOUR VIDEOS"
echo "=============================================================================="

# Find first video file
VIDEO_FILE=$(find /workspace -type f \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | grep -v cache | grep -v checkpoint | head -1)

if [ -z "$VIDEO_FILE" ]; then
    echo "❌ No video files found in /workspace"
    echo ""
    echo "Please run this to find your videos:"
    echo "  find /workspace -name '*.mp4' -type f | head -5"
    exit 1
fi

echo "✅ Found video: $VIDEO_FILE"
echo ""

echo "=============================================================================="
echo "🚀 RUNNING BENCHMARK"
echo "=============================================================================="
echo ""

cd /workspace/violence_detection_mvp
python3 benchmark_video_loading.py "$VIDEO_FILE"
