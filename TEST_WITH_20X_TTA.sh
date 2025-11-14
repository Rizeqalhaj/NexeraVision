#!/bin/bash

echo "================================================================"
echo "20X TEST-TIME AUGMENTATION (TTA)"
echo "================================================================"
echo ""
echo "What this does:"
echo "  - Takes each test video"
echo "  - Creates 20 augmented versions:"
echo "    * Original + horizontal flip"
echo "    * 4 brightness levels + flip"
echo "    * 3 contrast variations"
echo "    * 4 small rotations"
echo "  - Predicts on all 20 versions"
echo "  - Averages predictions → More robust!"
echo ""
echo "Expected boost: +1-2% accuracy"
echo "Time: ~30-45 minutes for 4,684 test videos"
echo "================================================================"
echo ""

cd /workspace/violence_detection_mvp

echo "Testing Model 1 (VGG19 + BiLSTM) with 20x TTA..."
echo ""

python3 /home/admin/Desktop/NexaraVision/violence_detection_mvp/predict_with_tta_20x.py \
    --model /workspace/ensemble_models/vgg19_bilstm/best_model.h5 \
    --dataset /workspace/organized_dataset

echo ""
echo "================================================================"
echo "TTA COMPLETE!"
echo "================================================================"
