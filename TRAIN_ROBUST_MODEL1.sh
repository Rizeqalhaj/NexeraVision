#!/bin/bash

echo "================================================================"
echo "TRAINING ROBUST MODEL 1 - 10x AUGMENTATION"
echo "================================================================"
echo ""
echo "Strategy:"
echo "  - Each training video → 10 augmented versions"
echo "  - Total training samples: 17,678 → 176,780"
echo "  - Aggressive augmentations:"
echo "    * Brightness: 0.6x to 1.4x"
echo "    * Contrast: 0.7x to 1.5x"
echo "    * Rotation: ±15 degrees"
echo "    * Zoom: 0.85x to 1.15x"
echo "    * Noise: 30% probability"
echo "    * Blur: 20% probability"
echo "    * Frame dropout: 15%"
echo ""
echo "Goal: Model that works in REAL scenarios with varying conditions"
echo "================================================================"
echo ""

cd /workspace/violence_detection_mvp

python3 /home/admin/Desktop/NexaraVision/violence_detection_mvp/train_robust_model1.py

echo ""
echo "================================================================"
echo "TRAINING COMPLETE!"
echo "================================================================"
