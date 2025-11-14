#!/bin/bash
# Quick Start Script for 93-97% Accuracy
# Run this on your cloud GPU

set -e  # Exit on error

echo "=========================================="
echo "QUICK START: Path to 93-97% Accuracy"
echo "=========================================="
echo ""

# Change to workspace
cd /workspace

echo "Step 1: Download ALL datasets (2-4 hours)"
echo "This will download ~10,000 videos from 7 datasets"
read -p "Start downloading? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python /home/admin/Desktop/NexaraVision/download_all_violence_datasets.py
fi

echo ""
echo "Step 2: Combine all datasets (30-60 min)"
read -p "Start combining datasets? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python /home/admin/Desktop/NexaraVision/combine_all_datasets.py
fi

echo ""
echo "Step 3: Ultimate Training (12-18 hours)"
echo "Features:"
echo "  - EfficientNetB4 feature extraction"
echo "  - Bidirectional LSTM + Attention"
echo "  - 10× data augmentation"
echo "  - 1000 epochs"
echo "  - Target: 88-93% accuracy"
read -p "Start ultimate training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python /home/admin/Desktop/NexaraVision/runpod_train_ultimate.py
fi

echo ""
echo "Step 4: Ensemble Training (optional, for 93%+)"
echo "This trains 5 different models and averages predictions"
read -p "Start ensemble training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python /home/admin/Desktop/NexaraVision/train_ensemble_models.py
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Monitor training: tail -f training_ultimate_*.log"
echo "2. View TensorBoard: tensorboard --logdir ./models_ultimate/tensorboard_logs"
echo "3. Check progress: tail -1 ./models_ultimate/training_log.csv"
echo ""
echo "Expected timeline:"
echo "  - Download: 2-4 hours"
echo "  - Combine: 30-60 min"
echo "  - Ultimate training: 12-18 hours → 88-93% accuracy"
echo "  - Ensemble training: 20-30 hours → 93-97% accuracy"
echo ""
echo "Total cost (2× RTX 4080 @ $0.89/hr): ~$50"
echo "=========================================="
