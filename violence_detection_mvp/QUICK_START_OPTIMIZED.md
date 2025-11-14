# Quick Start Guide - Optimized Training

## Prerequisites

### System Requirements
- 2× NVIDIA RTX 5000 Ada Generation (or similar GPUs with 24GB+ VRAM each)
- CUDA 11.8+ and cuDNN 8.6+
- TensorFlow 2.12+
- 64GB+ system RAM (recommended)
- 500GB+ storage for dataset and features

### Python Dependencies
```bash
pip install tensorflow-gpu>=2.12.0
pip install opencv-python>=4.7.0
pip install tqdm scikit-learn
pip install tensorboard
```

---

## Step 1: Prepare Dataset

Organize your dataset in the following structure:
```
organized_dataset/
├── train/
│   ├── violent/
│   │   ├── video001.mp4
│   │   ├── video002.mp4
│   │   └── ...
│   └── nonviolent/
│       ├── video001.mp4
│       ├── video002.mp4
│       └── ...
├── val/
│   ├── violent/
│   └── nonviolent/
└── test/
    ├── violent/
    └── nonviolent/
```

### Expected Distribution
- **Training:** 70-80% of data
- **Validation:** 10-15% of data
- **Testing:** 10-15% of data

---

## Step 2: First Training Run

### Basic Command (Recommended Settings)
```bash
python train_rtx5000_dual_optimized.py \
    --dataset-path /path/to/organized_dataset \
    --epochs 100 \
    --batch-size 64 \
    --cache-dir ./feature_cache \
    --checkpoint-dir ./checkpoints
```

### What Happens:
1. **GPU Detection:** Verifies 2 GPUs available
2. **Mixed Precision:** Enables FP16 training automatically
3. **Feature Extraction:** Extracts VGG19 features (slow first time, ~30-60 min)
4. **Feature Caching:** Saves features to disk (next runs are 10× faster)
5. **Training:** Trains for 100 epochs with all optimizations
6. **Checkpointing:** Saves best model and periodic checkpoints
7. **Evaluation:** Tests on test set and generates metrics

### Expected Runtime:
- **First run:** 5-6 hours (includes feature extraction)
- **Subsequent runs:** 4-5 hours (loads cached features)

---

## Step 3: Monitor Training

### TensorBoard (Real-time Monitoring)
```bash
# In a separate terminal
tensorboard --logdir ./checkpoints/tensorboard --port 6006

# Open browser: http://localhost:6006
```

**Key Metrics to Watch:**
- **Loss:** Should decrease smoothly
- **Accuracy:** Should increase and plateau around 93-95%
- **Val Loss vs Train Loss:** Gap indicates overfitting
- **Learning Rate:** Should follow warmup + cosine decay curve

### GPU Monitoring
```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Target: >90% GPU utilization on both GPUs
```

### Training Logs
```bash
# View live training progress
tail -f checkpoints/training_history.csv
```

---

## Step 4: After Training

### Check Results
```bash
# View training results
cat checkpoints/training_results.json
```

**Expected Results:**
```json
{
  "test_metrics": {
    "accuracy": 0.93,
    "precision": 0.92,
    "recall": 0.94,
    "auc": 0.97
  }
}
```

### Load Best Model
```python
import tensorflow as tf
from src.model_architecture import AttentionLayer

model = tf.keras.models.load_model(
    'checkpoints/best_model.h5',
    custom_objects={'AttentionLayer': AttentionLayer}
)
```

---

## Advanced Usage

### Maximum Accuracy Configuration
```bash
python train_rtx5000_dual_optimized.py \
    --dataset-path /path/to/organized_dataset \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --warmup-epochs 10 \
    --label-smoothing 0.15 \
    --use-focal-loss \
    --use-class-weights
```

### Fast Experimentation
```bash
python train_rtx5000_dual_optimized.py \
    --dataset-path /path/to/organized_dataset \
    --epochs 30 \
    --batch-size 96 \
    --warmup-epochs 3
```

### Debugging Mode (No Mixed Precision)
```bash
python train_rtx5000_dual_optimized.py \
    --dataset-path /path/to/organized_dataset \
    --epochs 10 \
    --batch-size 32 \
    --no-mixed-precision
```

---

## Command-Line Arguments

### Required
- `--dataset-path`: Path to organized dataset folder

### Training Parameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Total batch size across all GPUs (default: 64)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--warmup-epochs`: Warmup epochs (default: 5)

### Optimization Flags
- `--mixed-precision`: Enable FP16 training (default: True)
- `--no-mixed-precision`: Disable FP16 training
- `--xla`: Enable XLA compilation (default: True)
- `--use-focal-loss`: Use focal loss (default: True)
- `--use-class-weights`: Use class weights (default: True)

### Regularization
- `--label-smoothing`: Label smoothing factor (default: 0.1)

### Directories
- `--cache-dir`: Feature cache directory (default: ./feature_cache)
- `--checkpoint-dir`: Checkpoint directory (default: ./checkpoints)

---

## Troubleshooting

### Out of Memory Error
**Solution 1:** Reduce batch size
```bash
python train_rtx5000_dual_optimized.py ... --batch-size 32
```

**Solution 2:** Ensure mixed precision is enabled
```bash
# Verify in logs: "Mixed precision training enabled (FP16)"
```

### Training Too Slow
**Check GPU Utilization:**
```bash
nvidia-smi
# Should show >90% GPU utilization
# Should show "FP16" in processes
```

**Check Feature Cache:**
```bash
ls -lh feature_cache/
# Should see: train_features.npy, val_features.npy, test_features.npy
# First run extracts features (slow), subsequent runs load cache (fast)
```

### Poor Accuracy on Minority Class
**Increase focal loss gamma:**
```bash
# Edit TrainingConfig in the script:
focal_loss_gamma: float = 3.0  # Increase from 2.0
```

**Or adjust class weights manually:**
```python
# In calculate_class_weights(), multiply weights:
weights[0] *= 1.5  # Boost non-violent class
```

### Training Unstable (Loss Spikes)
**Reduce learning rate:**
```bash
python train_rtx5000_dual_optimized.py ... --learning-rate 0.0005
```

**Increase warmup:**
```bash
python train_rtx5000_dual_optimized.py ... --warmup-epochs 10
```

---

## File Structure After Training

```
project/
├── train_rtx5000_dual_optimized.py
├── feature_cache/
│   ├── train_features.npy (VGG19 features cached)
│   ├── train_labels.npy
│   ├── val_features.npy
│   ├── val_labels.npy
│   ├── test_features.npy
│   └── test_labels.npy
└── checkpoints/
    ├── best_model.h5 (best validation accuracy)
    ├── checkpoint_epoch_005.h5
    ├── checkpoint_epoch_010.h5
    ├── ...
    ├── training_config.json (hyperparameters used)
    ├── training_results.json (final metrics)
    ├── training_history.csv (epoch-by-epoch metrics)
    └── tensorboard/
        └── events.out.tfevents...
```

---

## Performance Expectations

### Training Speed
- **Per epoch:** 2-3 minutes
- **100 epochs:** 4-5 hours
- **GPU utilization:** 95%+

### Accuracy
- **Overall:** 93-95%
- **Non-violent class:** 88-92%
- **Violent class:** 94-96%
- **AUC:** 0.96-0.98

### Memory Usage
- **Per GPU:** 6-8 GB / 32 GB (19-25% utilization)
- **System RAM:** 16-24 GB

---

## Next Steps

### 1. Analyze Results
```bash
# View confusion matrix and per-class metrics
python -c "
import json
with open('checkpoints/training_results.json') as f:
    results = json.load(f)
    print(json.dumps(results, indent=2))
"
```

### 2. Test on New Videos
```python
import numpy as np
import tensorflow as tf
from src.model_architecture import AttentionLayer

# Load model
model = tf.keras.models.load_model(
    'checkpoints/best_model.h5',
    custom_objects={'AttentionLayer': AttentionLayer}
)

# Extract features from new video
features = extract_vgg19_features(['new_video.mp4'])

# Predict
prediction = model.predict(features)
print(f"Violent probability: {prediction[0][1]:.2%}")
```

### 3. Deploy Model
See `OPTIMIZATION_REPORT.md` section 10 for deployment options:
- TensorFlow Serving (API endpoint)
- TensorFlow Lite (mobile deployment)
- TensorRT (optimized inference)

---

## Support

### Documentation
- **Full optimization details:** `OPTIMIZATION_REPORT.md`
- **Model architecture:** `src/model_architecture.py`
- **Configuration:** `src/config.py`

### Common Issues
1. **CUDA out of memory:** Reduce `--batch-size`
2. **Feature extraction slow:** Normal on first run, uses cache after
3. **Low GPU utilization:** Check mixed precision is enabled
4. **Training diverges:** Reduce learning rate or increase warmup

---

## Example Training Session

```bash
# 1. Start training
python train_rtx5000_dual_optimized.py \
    --dataset-path ~/datasets/violence_detection \
    --epochs 100 \
    --batch-size 64

# Output:
# ================================================================================
# GPU CONFIGURATION
# ================================================================================
# Found 2 GPU(s)
#   GPU 0: NVIDIA RTX 5000 Ada Generation (32.0 GB)
#   GPU 1: NVIDIA RTX 5000 Ada Generation (32.0 GB)
# Total VRAM: 64.0 GB
# Mixed precision training enabled (FP16)
# MirroredStrategy created with 2 devices
#
# ================================================================================
# LOADING DATASET STRUCTURE
# ================================================================================
# TRAIN: 10,000 videos (Violent: 7,800 [78.0%], Non-violent: 2,200 [22.0%])
# VAL  :  1,500 videos (Violent: 1,170 [78.0%], Non-violent:   330 [22.0%])
# TEST :  1,500 videos (Violent: 1,170 [78.0%], Non-violent:   330 [22.0%])
#
# ================================================================================
# EXTRACTING VGG19 FEATURES - TRAIN
# ================================================================================
# Extracting features for 10,000 videos...
# [Progress bar shows ~45 minutes for first run]
# Caching features to ./feature_cache/train_features.npy
#
# [Similar for validation and test sets]
#
# ================================================================================
# TRAINING
# ================================================================================
# Epochs: 100
# Steps per epoch: 157
# Epoch 1/100: 157/157 [====] - 180s - loss: 0.6234 - accuracy: 0.6892
# Epoch 2/100: 157/157 [====] - 150s - loss: 0.4123 - accuracy: 0.8145
# ...
# Epoch 100/100: 157/157 [====] - 150s - loss: 0.0892 - accuracy: 0.9745
#
# ================================================================================
# EVALUATION ON TEST SET
# ================================================================================
# Test Results Summary:
#   Overall Accuracy: 94.23%
#   Precision: 0.9312
#   Recall: 0.9487
#   AUC: 0.9712
#
# ================================================================================
# TRAINING COMPLETE
# ================================================================================
# Best model saved: checkpoints/best_model.h5
# Results saved: checkpoints/training_results.json
```

Success! Your model is ready for production use.
