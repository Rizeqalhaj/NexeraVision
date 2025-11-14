#!/bin/bash

echo "================================================================"
echo "AGGRESSIVE VIDEO CLEANUP - GUARANTEED FIX FOR HANGING"
echo "================================================================"
echo ""
echo "This script uses multiprocessing with forced timeout to catch"
echo "videos that hang even when cv2.VideoCapture timeouts don't work."
echo ""
echo "It will:"
echo "  1. Test EVERY video with 3-second forced timeout"
echo "  2. Move corrupted/hanging videos to backup directory"
echo "  3. Leave only clean videos for training"
echo ""
echo "================================================================"
echo ""

cd /home/admin/Desktop/NexaraVision/violence_detection_mvp

python3 aggressive_video_cleanup.py

echo ""
echo "================================================================"
echo "Cleanup complete! Now you can start training:"
echo "================================================================"
echo ""
echo "cd /workspace/violence_detection_mvp"
echo "python3 train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset"
echo ""
