#!/bin/bash

echo "================================================================"
echo "SIMPLE TEST-TIME AUGMENTATION (3x)"
echo "================================================================"
echo ""
echo "What this does:"
echo "  - Takes each test video"
echo "  - Creates 3 versions:"
echo "    * Original"
echo "    * Horizontal flip"
echo "    * Slight brightness boost (+10%)"
echo "  - Averages predictions → More robust!"
echo ""
echo "Expected boost: +0.3-0.5% accuracy"
echo "Time: ~10-15 minutes for test set"
echo "================================================================"
echo ""

cd /workspace/violence_detection_mvp

echo "Testing Model 1 (VGG19 + BiLSTM) with Simple TTA..."
echo ""

python3 /home/admin/Desktop/NexaraVision/violence_detection_mvp/predict_with_tta_simple.py \
    --model /workspace/ensemble_models/vgg19_bilstm/best_model.h5 \
    --dataset /workspace/organized_dataset

echo ""
echo "================================================================"
echo "SIMPLE TTA COMPLETE!"
echo "================================================================"
