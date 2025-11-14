#!/bin/bash

echo "================================================================"
echo "FIX FOR TRAINING SLOWDOWN AT 70%"
echo "================================================================"
echo ""
echo "The problem is videos with corrupted h264 encoding that cause"
echo "10x slowdown (from 10 it/s to 1.7 it/s)."
echo ""
echo "This script will:"
echo "  1. Test each video by reading 10 frames with 2s timeout"
echo "  2. Remove videos that take >2s (too slow)"
echo "  3. Keep only fast-processing videos"
echo ""
echo "Expected result: Training at consistent 8-10 it/s speed"
echo "================================================================"
echo ""

cd /home/admin/Desktop/NexaraVision/violence_detection_mvp

# Suppress h264 warnings
export OPENCV_LOG_LEVEL=ERROR
export OPENCV_VIDEOIO_DEBUG=0

python3 remove_slow_videos.py

echo ""
echo "================================================================"
echo "Now you can start training without slowdowns:"
echo "================================================================"
echo ""
echo "cd /workspace/violence_detection_mvp"
echo "python3 train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset"
echo ""
