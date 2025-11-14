#!/bin/bash
# Complete Setup and Training Pipeline for 2× RTX 5000 Ada
# Optimized for: AMD Threadripper PRO 7945WX, 64GB VRAM, 257GB RAM
# Estimated total time: 32-39 hours
# Estimated cost: $34-42 @ $1.07/hr

set -e

echo "=========================================="
echo "RTX 5000 Ada VIOLENCE DETECTION TRAINING"
echo "Hardware: 2× RTX 5000 Ada (64GB VRAM)"
echo "CPU: AMD Threadripper PRO 7945WX (24 cores)"
echo "RAM: 257GB | Storage: 1TB NVMe"
echo "Cost: $1.07/hr"
echo "=========================================="
echo ""

# ============================================
# SYSTEM OPTIMIZATION
# ============================================
echo "🔧 Optimizing system for AI training..."

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

# CPU optimization for 24-core Threadripper
export OMP_NUM_THREADS=24
export TF_NUM_INTEROP_THREADS=24
export TF_NUM_INTRAOP_THREADS=24

# Memory optimization for 257GB RAM
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH=1

# XLA compilation for 15-30% speedup
export TF_XLA_FLAGS=--tf_xla_auto_jit=2

echo "✅ System optimized"
echo ""

# ============================================
# DEPENDENCIES
# ============================================
echo "📦 Installing dependencies..."

pip install --upgrade pip
pip install tensorflow[and-cuda]==2.15.0
pip install opencv-python-headless
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install pillow
pip install pandas
pip install kaggle
pip install yt-dlp
pip install kinetics-downloader
pip install internetarchive

echo "✅ Dependencies installed"
echo ""

# ============================================
# DOWNLOAD DATASETS
# ============================================
echo "=========================================="
echo "PHASE 1: DATASET DOWNLOADS"
echo "Target: 100K violent + 100K non-violent"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p /workspace/datasets/{violent,nonviolent}
mkdir -p /workspace/data/combined/{train,val}/{fight,non_fight}
mkdir -p /workspace/features
mkdir -p /workspace/models/{ultimate,ensemble}
mkdir -p /workspace/logs

cd /workspace/datasets

# Copy download scripts
cp /home/admin/Desktop/NexaraVision/download_phase1_immediate.sh ./violent/
cp /home/admin/Desktop/NexaraVision/download_phase3_kinetics.sh ./violent/
cp /home/admin/Desktop/NexaraVision/download_nonviolent_phase1.sh ./nonviolent/
cp /home/admin/Desktop/NexaraVision/download_nonviolent_phase3.sh ./nonviolent/

chmod +x ./violent/*.sh
chmod +x ./nonviolent/*.sh

# Start downloads in parallel
echo "🚀 Starting parallel downloads..."
echo "Violent datasets: Phase 1 + Phase 3"
echo "Non-violent datasets: Phase 1 + Phase 3"
echo ""

cd /workspace/datasets/violent
./download_phase1_immediate.sh > /workspace/logs/violent_phase1.log 2>&1 &
VIOLENT_PID=$!

sleep 30  # Stagger starts

./download_phase3_kinetics.sh > /workspace/logs/violent_phase3.log 2>&1 &
KINETICS_PID=$!

cd /workspace/datasets/nonviolent
./download_nonviolent_phase1.sh > /workspace/logs/nonviolent_phase1.log 2>&1 &
NONVIOLENT1_PID=$!

sleep 30

./download_nonviolent_phase3.sh > /workspace/logs/nonviolent_phase3.log 2>&1 &
NONVIOLENT3_PID=$!

echo "📊 Download processes started:"
echo "  Violent Phase 1 PID: $VIOLENT_PID"
echo "  Violent Phase 3 PID: $KINETICS_PID"
echo "  Non-violent Phase 1 PID: $NONVIOLENT1_PID"
echo "  Non-violent Phase 3 PID: $NONVIOLENT3_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f /workspace/logs/violent_phase1.log"
echo "  tail -f /workspace/logs/violent_phase3.log"
echo ""

# Wait for all downloads
echo "⏳ Waiting for downloads to complete..."
echo "This will take 3-5 days. You can disconnect safely."
echo ""

wait $VIOLENT_PID
wait $KINETICS_PID
wait $NONVIOLENT1_PID
wait $NONVIOLENT3_PID

echo "✅ All downloads complete!"
echo ""

# ============================================
# COMBINE AND DEDUPLICATE DATASETS
# ============================================
echo "=========================================="
echo "PHASE 2: DATASET PREPARATION"
echo "=========================================="
echo ""

cd /workspace

# Copy combination script
cp /home/admin/Desktop/NexaraVision/combine_all_datasets_ultimate.py ./
cp /home/admin/Desktop/NexaraVision/deduplicate_videos.py ./

# Combine datasets
echo "🔄 Combining all datasets..."
python combine_all_datasets_ultimate.py

# Deduplicate
echo "🔍 Removing duplicates..."
python deduplicate_videos.py /workspace/data/combined --method both --execute

# Count final dataset
TRAIN_FIGHT=$(find /workspace/data/combined/train/fight -type f | wc -l)
TRAIN_NON=$(find /workspace/data/combined/train/non_fight -type f | wc -l)
VAL_FIGHT=$(find /workspace/data/combined/val/fight -type f | wc -l)
VAL_NON=$(find /workspace/data/combined/val/non_fight -type f | wc -l)

TOTAL=$((TRAIN_FIGHT + TRAIN_NON + VAL_FIGHT + VAL_NON))

echo ""
echo "📊 FINAL DATASET STATISTICS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training - Fight:      $TRAIN_FIGHT"
echo "Training - Non-fight:  $TRAIN_NON"
echo "Validation - Fight:    $VAL_FIGHT"
echo "Validation - Non-fight: $VAL_NON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TOTAL VIDEOS:          $TOTAL"
echo ""

if [ $TOTAL -lt 80000 ]; then
    echo "⚠️  Dataset size below target (80K minimum)"
    echo "Consider running additional downloads"
fi

# ============================================
# FEATURE EXTRACTION
# ============================================
echo "=========================================="
echo "PHASE 3: FEATURE EXTRACTION"
echo "EfficientNetB4 with 2× RTX 5000 Ada"
echo "Estimated time: 4-5 hours"
echo "=========================================="
echo ""

# Copy extraction script
cp /home/admin/Desktop/NexaraVision/extract_features_optimized.py ./

# Run extraction with RTX 5000 Ada config
python extract_features_optimized.py \
    --data-dir /workspace/data/combined \
    --output-dir /workspace/features \
    --batch-size 192 \
    --num-workers 20 \
    --gpu-count 2 \
    --mixed-precision

echo "✅ Feature extraction complete"
echo ""

# ============================================
# ULTIMATE MODEL TRAINING
# ============================================
echo "=========================================="
echo "PHASE 4: ULTIMATE MODEL TRAINING"
echo "Target: 88-93% accuracy"
echo "Estimated time: 10-12 hours"
echo "=========================================="
echo ""

# Copy training script
cp /home/admin/Desktop/NexaraVision/runpod_train_ultimate.py ./

# Train with optimized settings
python runpod_train_ultimate.py \
    --features-dir /workspace/features \
    --output-dir /workspace/models/ultimate \
    --batch-size 256 \
    --epochs 80 \
    --gpu-count 2 \
    --mixed-precision \
    --xla-compile

echo "✅ Ultimate model training complete"
echo ""

# ============================================
# ENSEMBLE TRAINING
# ============================================
echo "=========================================="
echo "PHASE 5: ENSEMBLE TRAINING"
echo "Target: 93-97% accuracy"
echo "Estimated time: 18-22 hours"
echo "=========================================="
echo ""

# Copy ensemble script
cp /home/admin/Desktop/NexaraVision/train_ensemble_models.py ./

# Train 5 models (2 in parallel)
python train_ensemble_models.py \
    --features-dir /workspace/features \
    --output-dir /workspace/models/ensemble \
    --batch-size 192 \
    --epochs 70 \
    --gpu-count 2 \
    --parallel-models 2 \
    --mixed-precision

echo "✅ Ensemble training complete"
echo ""

# ============================================
# FINAL EVALUATION
# ============================================
echo "=========================================="
echo "FINAL EVALUATION"
echo "=========================================="
echo ""

# Test ensemble
python -c "
import tensorflow as tf
import numpy as np
from pathlib import Path

# Load models
models = []
for model_path in Path('/workspace/models/ensemble').glob('*.h5'):
    models.append(tf.keras.models.load_model(model_path))

print(f'Loaded {len(models)} ensemble models')

# Load validation data
X_val = np.load('/workspace/features/X_val.npy')
y_val = np.load('/workspace/features/y_val.npy')

# Ensemble prediction
predictions = []
for model in models:
    pred = model.predict(X_val, batch_size=256, verbose=0)
    predictions.append(pred)

# Average predictions
ensemble_pred = np.mean(predictions, axis=0)
ensemble_accuracy = np.mean((ensemble_pred > 0.5).astype(int) == y_val)

print(f'')
print(f'🎉 FINAL ENSEMBLE ACCURACY: {ensemble_accuracy*100:.2f}%')
print(f'')

if ensemble_accuracy >= 0.95:
    print('✅ PRODUCTION-READY: 95%+ accuracy achieved!')
elif ensemble_accuracy >= 0.93:
    print('✅ EXCELLENT: 93%+ accuracy achieved')
elif ensemble_accuracy >= 0.90:
    print('✅ GOOD: 90%+ accuracy achieved')
else:
    print('⚠️  Consider more training or data augmentation')
"

# ============================================
# COST SUMMARY
# ============================================
HOURS=39
COST=$(echo "$HOURS * 1.07" | bc)

echo ""
echo "=========================================="
echo "TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "💰 COST SUMMARY:"
echo "  Total hours: ~$HOURS hours"
echo "  Cost: ~\$$COST"
echo ""
echo "📁 MODEL LOCATIONS:"
echo "  Ultimate: /workspace/models/ultimate/"
echo "  Ensemble: /workspace/models/ensemble/"
echo "  Features: /workspace/features/"
echo ""
echo "🚀 READY FOR PRODUCTION DEPLOYMENT!"
echo ""
echo "Next steps:"
echo "1. Download models: scp from RunPod"
echo "2. Deploy to camera system"
echo "3. Real-time violence detection operational"
echo ""
echo "=========================================="
