#!/bin/bash

# Quick Start Training Script
# Run this on RunPod L40S or any machine with GPU

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Violence Detection Training - L40S Optimized            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if dataset path is provided
DATASET_PATH=${1:-"/workspace/RWF-2000"}

echo "📊 Configuration:"
echo "   Dataset path: $DATASET_PATH"
echo "   GPU: Checking..."
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not found"
    echo "   Make sure you're on a GPU instance"
    echo ""
fi

# Check dataset
if [ ! -d "$DATASET_PATH/train" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_PATH"
    echo ""
    echo "Usage: ./start_training.sh /path/to/RWF-2000"
    echo ""
    echo "Your dataset should have this structure:"
    echo "  $DATASET_PATH/"
    echo "  ├── train/"
    echo "  │   ├── Fight/"
    echo "  │   └── NonFight/"
    echo "  └── val/"
    echo "      ├── Fight/"
    echo "      └── NonFight/"
    echo ""
    exit 1
fi

echo "✅ Dataset found"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import tensorflow; import cv2; import numpy; import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip install -q tensorflow opencv-python-headless scikit-learn numpy
    echo "✅ Dependencies installed"
else
    echo "✅ All dependencies available"
fi
echo ""

# Create directories
mkdir -p models logs data/processed/features_l40s

echo "🚀 Starting training..."
echo "   Expected time: 1-2 hours"
echo "   Expected cost: ~$2-3 on L40S"
echo "   Target accuracy: 95-98%"
echo ""
echo "Press Ctrl+C to cancel, or wait 10 seconds to start..."
sleep 10

# Start training
python3 runpod_train_l40s.py \
    --dataset-path "$DATASET_PATH" \
    --batch-size 256 \
    --epochs 100 \
    --learning-rate 0.0001

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  Training Complete!                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Output files:"
echo "   - models/violence_detector_l40s_best.h5"
echo "   - logs/training_l40s_*.csv"
echo "   - logs/history_l40s_*.json"
echo ""
echo "Next steps:"
echo "   1. Download the model file"
echo "   2. Test with: python test_model.py"
echo "   3. Deploy for production use"
echo ""
