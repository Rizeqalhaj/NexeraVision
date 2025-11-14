#!/bin/bash
# Complete Training Command for NexaraVision Violence Detection
# Dataset: 31,209 videos (balanced 50/50)
# Target: 93-95% accuracy
# Hardware: Single RTX 5000 Ada GPU (GPU:0)

echo "================================================================"
echo "STARTING TRAINING - SINGLE GPU MODE"
echo "================================================================"
echo ""
echo "✅ Feature extraction already complete (cached)"
echo "✅ Using only GPU:0 (avoids dual-GPU autotuning conflicts)"
echo "✅ Optimized parameters for 31K balanced dataset"
echo ""
echo "================================================================"
echo ""

cd /workspace/violence_detection_mvp

python3 train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset

echo ""
echo "================================================================"
echo "Training complete!"
echo "Model saved to: violence_detection_mvp/models/best_model.h5"
echo "================================================================"
