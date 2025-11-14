#!/bin/bash

echo "================================================================"
echo "ENSEMBLE TRAINING FOR 92-95% ACCURACY"
echo "================================================================"
echo ""
echo "Strategy:"
echo "  1. Train 3 diverse models (VGG19, ResNet50, EfficientNet)"
echo "  2. Each with different architecture (BiLSTM, BiGRU, Attention)"
echo "  3. Data augmentation: flip, brightness, rotation, frame dropout"
echo "  4. Combine via soft voting for final prediction"
echo ""
echo "Expected Results:"
echo "  - Individual models: 90-92% each"
echo "  - Ensemble: 92-95%"
echo ""
echo "Time: ~8-12 hours total (2-4 hours per model)"
echo "================================================================"
echo ""

cd /home/admin/Desktop/NexaraVision/violence_detection_mvp

python3 train_ensemble_ultimate.py

echo ""
echo "================================================================"
echo "ENSEMBLE TRAINING COMPLETE"
echo "================================================================"
echo ""
echo "Next step: Evaluate ensemble"
echo "python3 ensemble_predict.py"
echo ""
