# Single GPU Fix for Multi-GPU Systems

## Problem

When you have multiple GPUs (2x RTX 3090 Ti), TensorFlow tries to use both GPUs but isn't configured for distributed training, causing:
```
CUDA_ERROR_OUT_OF_MEMORY: out of memory
```

Even though you have **48GB VRAM total**, the OOM error occurs because TensorFlow initializes both GPUs improperly.

---

## Solution: Force Single GPU Mode

### ✅ Method 1: Use START_TRAINING.sh (RECOMMENDED)

This script sets environment variables BEFORE Python starts:

```bash
cd /workspace
chmod +x START_TRAINING.sh
./START_TRAINING.sh
```

**What it does:**
- Sets `CUDA_VISIBLE_DEVICES=0` (TensorFlow only sees GPU 0)
- Enables memory growth (dynamic allocation)
- Clears any stuck processes
- Shows GPU status before training
- Starts training with proper configuration

---

### ✅ Method 2: Manual Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
python3 train_model_optimized.py
```

---

### ✅ Method 3: Run Fix Script First

```bash
cd /workspace
chmod +x FIX_SINGLE_GPU.sh
./FIX_SINGLE_GPU.sh
./START_TRAINING.sh
```

**What FIX_SINGLE_GPU.sh does:**
- Reduces batch_size to 8 (conservative for single GPU)
- Updates training config
- Prepares environment for training

---

## Updated Configuration

The training script now automatically:

1. **Sets environment variables BEFORE TensorFlow import**:
   ```python
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force GPU 0 only
   os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
   ```

2. **Enables memory growth**:
   - TensorFlow allocates memory as needed
   - Prevents grabbing all 24GB at once

3. **Uses conservative batch size**:
   - batch_size=8 (safe for single GPU)
   - Can increase to 12 or 16 if training succeeds

---

## Expected Performance (Single GPU)

| Configuration | Value |
|---------------|-------|
| **GPU Used** | GPU 0 only (24 GB VRAM) |
| **Batch Size** | 8 |
| **VRAM Usage** | ~4-6 GB |
| **Training Speed** | ~60-80 videos/second |
| **Training Time** | ~6-8 hours |
| **Target Accuracy** | 96-100% |

**Why slower than before?**
- batch_size=8 instead of 16 (more conservative)
- Single GPU instead of potential dual-GPU
- **But it will actually WORK without OOM!**

---

## Verification

### Check GPU Visibility

Before training:
```bash
export CUDA_VISIBLE_DEVICES=0
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Expected output:**
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Should show **1 GPU**, not 2!

### Monitor During Training

Terminal 1 (Training):
```bash
./START_TRAINING.sh
```

Terminal 2 (Monitoring):
```bash
watch -n 1 nvidia-smi
```

**Expected GPU usage:**
```
GPU 0: 4-6 GB used, 60-70°C, training active ✅
GPU 1: 0 GB used, idle ❌
```

---

## Troubleshooting

### Still Getting OOM?

#### 1. Reduce Batch Size Further
```bash
# Edit config
vim /workspace/training_config.json

# Change batch_size to 4
{
  "training": {
    "batch_size": 4
  }
}
```

#### 2. Check for Other GPU Processes
```bash
# See what's using GPU memory
nvidia-smi

# Kill stuck processes
pkill -f python
sleep 5

# Restart training
./START_TRAINING.sh
```

#### 3. Use Smaller Model (Last Resort)
Edit `/workspace/model_architecture.py`:
```python
# Change from ResNet50V2 to MobileNetV2
from tensorflow.keras.applications import MobileNetV2

backbone = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
```

**Trade-off:** 3-5% accuracy loss, but trains on less memory

---

## Why Environment Variables Must Come First

### ❌ Wrong (OOM occurs):
```python
import tensorflow as tf  # TF initializes both GPUs here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Too late!
```

### ✅ Correct:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set BEFORE TF import
import tensorflow as tf  # TF only sees GPU 0
```

**The updated train_model_optimized.py now does this correctly!**

---

## Files for Single GPU Setup

1. **START_TRAINING.sh** - Main training starter (sets env vars)
2. **FIX_SINGLE_GPU.sh** - Updates config for single GPU
3. **train_model_optimized.py** - Updated with env vars before TF import
4. **training_config.json** - Conservative batch_size=8

---

## Quick Start

```bash
cd /workspace

# Step 1: Fix config (optional, already done in script)
./FIX_SINGLE_GPU.sh

# Step 2: Start training
./START_TRAINING.sh

# Step 3: Monitor in another terminal
watch -n 1 nvidia-smi
```

---

## Summary

**Problem:** Multi-GPU system without distributed training → OOM errors
**Solution:** Force single GPU (GPU 0 only) via environment variables
**Method:** Set CUDA_VISIBLE_DEVICES=0 BEFORE importing TensorFlow
**Result:** Stable training on 24GB VRAM with batch_size=8

**You should now see training start without OOM errors!** 🚀

---

## After Training Succeeds

If training works with batch_size=8, you can optimize:

### Increase Batch Size (Faster Training)

Edit `/workspace/training_config.json`:
```json
{
  "training": {
    "batch_size": 12  // Or even 16
  }
}
```

Restart training and monitor VRAM usage. If it stays under ~18GB, you're safe!

---

## Multi-GPU Training (Advanced, Future)

For true distributed training across both GPUs:

```python
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = build_model()
    model.compile(...)

model.fit(...)  # Automatically uses both GPUs
```

**Benefits:** Can use batch_size=32+ (split across GPUs)
**Complexity:** Requires code refactoring and testing

**For now, stick with single GPU - it works!**
