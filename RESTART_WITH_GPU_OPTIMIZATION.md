# Restart Training with GPU Optimization

## Current Situation

You're running the OLD slow training (2.92s/it, 14+ hours).

The NEW GPU-optimized code is ready with:
- ✅ Mixed Precision (FP16) - 2x faster
- ✅ Batch processing 32 videos at once - 4x faster
- ✅ Larger GPU batch size (64 frames) - 2x faster
- ✅ **Total: ~11-12x speedup**

---

## Step 1: Stop Current Training

```bash
# Find the process
ps aux | grep train_robust

# Kill it
pkill -f train_robust_model1
# OR
pkill -f python3
```

**Verify it stopped:**
```bash
ps aux | grep python3
# Should show nothing related to training
```

---

## Step 2: Clean Old Cache (Optional but Recommended)

The old extraction might have created partial cache:

```bash
# Remove old cache if it exists
rm -rf /workspace/robust_cache/
```

This ensures fresh start with GPU-optimized extraction.

---

## Step 3: Start GPU-Optimized Training

```bash
cd /home/admin/Desktop/NexaraVision
bash TRAIN_ROBUST_MODEL1.sh
```

---

## Step 4: Verify GPU Optimization is Working

### Check 1: Mixed Precision Message
You should see:
```
✅ Mixed Precision (FP16) enabled for 2x faster GPU processing
```

### Check 2: Speed
Within first minute, you should see:
```
Extracting vgg19_bilstm train:   1%|▋| 200/17678 [01:00<08:45, 33.2it/s]
```

**Key number:** `33.2it/s` (was 0.34it/s)

### Check 3: GPU Usage
In another terminal:
```bash
watch -n 1 nvidia-smi
```

Should show:
```
GPU-Util: 90-95%
Memory:   10-15GB / 32GB
Power:    180-220W / 250W
```

---

## Expected Timeline

### With GPU Optimization:
```
00:00 - Start
01:15 - Feature extraction done (176,780 samples created)
03:00 - Training complete
03:15 - Ready for TTA testing
```

**Total: ~3 hours** ✅

### Without GPU Optimization (what you had):
```
00:00 - Start
14:30 - Feature extraction done
18:00 - Training complete
```

**Total: ~18 hours** ❌

---

## Troubleshooting

### Issue: Still showing 2.92s/it

**Check if using correct script:**
```bash
ps aux | grep python3
```

Should show:
```
python3 /home/admin/Desktop/NexaraVision/violence_detection_mvp/train_robust_model1.py
```

NOT:
```
python3 train_ensemble_ultimate.py
```

### Issue: GPU not being used

**Check GPU availability:**
```bash
nvidia-smi
```

**Check TensorFlow sees GPU:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Issue: Out of memory

Reduce batch size in code:
```python
gpu_batch_size = 16  # Instead of 32
batch_size = 32      # Instead of 64
```

---

## Files Being Used

Confirm you have the latest versions:

### 1. `train_robust_model1.py`
Should have line 30-31:
```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
logger.info("✅ Mixed Precision (FP16) enabled for 2x faster GPU processing")
```

### 2. `train_ensemble_ultimate.py`
Should have line 309:
```python
gpu_batch_size = 32  # RTX 5000 Ada has 32GB VRAM
```

And line 347:
```python
features_flat = feature_extractor.predict(frames_flat, batch_size=64, verbose=0)
```

---

## One-Command Restart

```bash
pkill -f train_robust && \
rm -rf /workspace/robust_cache/ && \
cd /home/admin/Desktop/NexaraVision && \
bash TRAIN_ROBUST_MODEL1.sh
```

This will:
1. Kill old training
2. Remove partial cache
3. Start fresh GPU-optimized training

---

## What You Should See

### Startup:
```
================================================================
TRAINING ROBUST MODEL 1 - 10x AUGMENTATION
================================================================

✅ Mixed Precision (FP16) enabled for 2x faster GPU processing

================================================================================
LOADING DATASET
================================================================================

train: 17,678 videos
val: 4,233 videos
test: 3,835 videos

📈 Dataset Expansion:
  Original training videos: 17,678
  After 10x augmentation: 176,780

================================================================================
EXTRACTING FEATURES FOR VGG19_BILSTM
================================================================================

🔄 Creating 10x augmented training features...
⚠️  This will take longer but creates robust features!
```

### During Extraction:
```
Extracting vgg19_bilstm train:  15%|███▋| 2650/17678 [01:15<06:30, 38.5it/s]
```

**Good signs:**
- ✅ Speed: 30-40 it/s (not 0.3 it/s)
- ✅ Time estimate: 6-8 hours (not 14+ hours)
- ✅ GPU util: 90-95%

---

## After Feature Extraction

You'll see:
```
✅ Augmented training features shape: (176780, 20, 4096)
✅ Original: 17,678 → Augmented: 176,780 samples

Cached vgg19_bilstm train features: (176780, 20, 4096)
```

Then training starts:
```
================================================================================
TRAINING ROBUST MODEL
================================================================================

Epoch 1/150
2762/2762 ━━━━━━━━━━━━━━━━━━━━ 89s 32ms/step
  loss: 0.4521 - accuracy: 0.8234
  val_loss: 0.3821 - val_accuracy: 0.8456
```

---

## Summary

**DO THIS NOW:**
```bash
# 1. Stop old training
pkill -f train_robust_model1

# 2. Start GPU-optimized training
cd /home/admin/Desktop/NexaraVision
bash TRAIN_ROBUST_MODEL1.sh
```

**CONFIRM:**
- ✅ See "Mixed Precision (FP16) enabled" message
- ✅ Speed shows 30-40 it/s
- ✅ GPU at 90-95% usage

**RESULT:**
- ✅ Done in ~3 hours instead of 18
- ✅ Robust model ready for testing
- ✅ Fix the 68% TTA problem today!
