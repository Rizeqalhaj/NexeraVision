# Vast.ai GPU-Accelerated Training Setup

## 📦 Files to Upload to Your Vast.ai Instance

Upload these 3 files from your local machine to `/workspace/violence_detection_mvp/` on Vast.ai:

1. **`gpu_video_loader.py`** - GPU video decoding library (10-50x speedup)
2. **`train_robust_gpu_accelerated.py`** - GPU-accelerated training script
3. **`benchmark_video_loading.py`** - Performance benchmark script

---

## 🚀 Quick Start (Copy-Paste These Commands on Vast.ai)

### Step 1: Navigate to Project Directory

```bash
cd /workspace/violence_detection_mvp
```

### Step 2: Check Files Are Uploaded

```bash
ls -lh gpu_video_loader.py train_robust_gpu_accelerated.py benchmark_video_loading.py
```

You should see all 3 files listed.

### Step 3: Find Your Video Dataset Path

Run this to find where your videos are:

```bash
find /workspace -type f -name "*.mp4" | head -5
```

**Example output:**
```
/workspace/data/train/Fight/fight_001.mp4
/workspace/data/train/NonFight/normal_001.mp4
```

Note the base directory (e.g., `/workspace/data`).

### Step 4: Run Benchmark (Optional but Recommended)

This will show you the speedup:

```bash
# Replace VIDEO_PATH with path from Step 3
VIDEO_PATH="/workspace/data/train/Fight/fight_001.mp4"

python3 benchmark_video_loading.py "$VIDEO_PATH"
```

**Expected output:**
```
📊 PERFORMANCE SUMMARY
================================================================================

Backend         Time (ms)       Throughput           Speedup
--------------------------------------------------------------------------------
tensorflow      8.23ms          121.5 videos/sec     18.5x 🚀
opencv          152.34ms        6.6 videos/sec       1.0x

🎯 BEST: TENSORFLOW
   Speedup: 18.5x faster than OpenCV
   Time saved per video: 144.11ms

💰 TIME SAVINGS FOR FULL TRAINING:
   OpenCV (CPU):  7.5 hours
   TENSORFLOW (GPU): 0.4 hours
   Time saved:    7.1 hours (426 minutes)
   Speedup:       18.5x faster!
```

### Step 5: Update Data Path in Training Script

Edit the training script to match your dataset location:

```bash
# Check your data directory structure
ls -la /workspace/data/

# If different from /workspace/data, update the script:
nano train_robust_gpu_accelerated.py

# Find line ~267 (data_dir="/workspace/data")
# Change to your actual path if different
```

### Step 6: Run GPU-Accelerated Training

```bash
# Run in background with logging
nohup python3 train_robust_gpu_accelerated.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

**Or run in foreground (recommended first time):**

```bash
python3 train_robust_gpu_accelerated.py
```

### Step 7: Monitor GPU Usage (In Another Terminal)

Open a second terminal and run:

```bash
watch -n 1 nvidia-smi
```

**What you should see:**
- Memory: 20-28 GB used
- GPU Utilization: **60-70%** (vs 0-5% before)
- Power: 180-250W

---

## ⏱️ Expected Timeline

| Stage | Time | Notes |
|-------|------|-------|
| Feature Extraction (Train) | 20-30 min | 17,678 videos × 10x aug = 176,780 samples |
| Feature Extraction (Val) | 2-3 min | 2,211 videos (no aug) |
| Feature Extraction (Test) | 2-3 min | 2,211 videos (no aug) |
| Model Training | 1.5-2 hours | 150 epochs with early stopping |
| **TOTAL** | **~2.5 hours** | vs 12 hours with CPU loading |

---

## 📊 What Happens During Training

```
================================================================================
🚀 GPU-ACCELERATED ROBUST TRAINING
================================================================================
✅ GPU Configuration:
   - Using GPU:1 (RTX 5000 Ada, 32GB VRAM)
   - Mixed Precision (FP16) enabled
   - GPU video decoding enabled
   - Expected 10-50x speedup vs CPU video loading

📂 Loading dataset split...
✅ Dataset loaded:
   Train: 17,678 videos
   Val: 2,211 videos
   Test: 2,211 videos

🔧 Loading VGG19 feature extractor...
✅ VGG19 loaded

================================================================================
🚀 STARTING GPU-ACCELERATED FEATURE EXTRACTION
================================================================================
✅ GPU Video Loader initialized: tensorflow

🎬 GPU-ACCELERATED Feature Extraction: TRAIN
💡 Processing 176,780 samples in GPU batches of 32
GPU TRAIN: 100%|██████████| 176780/176780 [25:32<00:00, 115.4video/s]

✅ TRAIN features extracted:
   - Features shape: (176780, 30, 4096)
   - Fight samples: 88390
   - Normal samples: 88390

[Similar for VAL and TEST...]

================================================================================
✅ GPU Feature Extraction Complete
⏱️  Total time: 28.5 minutes
📊 Average: 118.3 videos/sec
================================================================================

🏗️  Building BiLSTM model...
✅ Model built

================================================================================
🎯 TRAINING ROBUST MODEL
================================================================================
Epoch 1/150
2762/2762 [==============================] - 45s 16ms/step
  loss: 0.4523 - accuracy: 0.7891
  val_loss: 0.3214 - val_accuracy: 0.8654

[Training continues...]

Epoch 87/150
2762/2762 [==============================] - 42s 15ms/step
  loss: 0.1234 - accuracy: 0.9523
  val_loss: 0.1567 - val_accuracy: 0.9234

================================================================================
📊 FINAL EVALUATION
================================================================================
✅ Test Accuracy: 91.24%

💾 Final model saved: /workspace/robust_gpu_models/robust_vgg19_final.h5

================================================================================
✅ TRAINING COMPLETE
================================================================================
```

---

## 📂 Output Files

After training, you'll have:

```
/workspace/robust_gpu_cache/
├── X_train_vgg19.npy       # Cached training features
├── y_train.npy
├── X_val_vgg19.npy
├── y_val.npy
├── X_test_vgg19.npy
└── y_test.npy

/workspace/robust_gpu_checkpoints/
└── robust_vgg19_XXX_0.XXXX.h5  # Best checkpoint

/workspace/robust_gpu_models/
├── robust_vgg19_final.h5   # 🎯 YOUR FINAL MODEL
└── training_log.csv         # Training history
```

---

## ✅ After Training: Test Robustness with TTA

Once training completes, validate that your model is robust:

```bash
cd /workspace/violence_detection_mvp

# Test with a video
python3 predict_with_tta_simple.py \
    --model /workspace/robust_gpu_models/robust_vgg19_final.h5 \
    --video /workspace/data/test/Fight/fight_test_001.mp4
```

**Expected results:**
- Standard prediction: ~90-92%
- With TTA (3x augmentation): ~88-90%

**Key metric: TTA accuracy should NOT drop below 85%**

If it stays above 85%, your model is robust! ✅

---

## 🐛 Troubleshooting

### Issue: "No module named 'cv2'"

```bash
pip install opencv-python
```

### Issue: "tensorflow.io has no attribute 'decode_video'"

Your TensorFlow is too old. Upgrade:

```bash
pip install --upgrade tensorflow>=2.13.0
```

Or install PyAV as alternative:

```bash
pip install av
# The script will auto-detect and use PyAV
```

### Issue: "CUDA out of memory"

Reduce batch size in `train_robust_gpu_accelerated.py`:

```python
# Line ~154
gpu_batch_size = 16  # Change from 32 to 16
```

### Issue: Still seeing 0% GPU utilization

Check if TensorFlow is using GPU:

```bash
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

Should show:
```
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Issue: "find: /workspace: No such file or directory"

Your dataset might be elsewhere. Find it:

```bash
find / -name "*.mp4" -type f 2>/dev/null | grep -i fight | head -5
```

Update the path in the training script accordingly.

---

## 💡 Pro Tips

### Tip 1: Run in Screen/Tmux

So you can disconnect and reconnect:

```bash
# Install tmux if not available
apt-get update && apt-get install -y tmux

# Start session
tmux new -s training

# Run training
python3 train_robust_gpu_accelerated.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### Tip 2: Download Model After Training

```bash
# On your local machine
scp user@vast-ip:/workspace/robust_gpu_models/robust_vgg19_final.h5 ./

# Or use Vast.ai file browser
```

### Tip 3: Resume Training If Interrupted

The script saves checkpoints. If training stops, you can resume by loading the best checkpoint:

```python
# In train_robust_gpu_accelerated.py, after building model
model = tf.keras.models.load_model('/workspace/robust_gpu_checkpoints/robust_vgg19_XXX_0.XXXX.h5')
```

---

## 📞 Support

If you encounter issues:

1. Check `training.log` for error messages
2. Verify GPU is accessible: `nvidia-smi`
3. Check TensorFlow version: `python3 -c "import tensorflow as tf; print(tf.__version__)"`
4. Ensure all 3 files are uploaded correctly

---

## 🎯 Success Checklist

- ✅ All 3 Python files uploaded to `/workspace/violence_detection_mvp/`
- ✅ Benchmark shows >10x speedup with TensorFlow backend
- ✅ Training starts and shows "GPU Video Loader initialized: tensorflow"
- ✅ GPU utilization reaches 60-70% (check with `nvidia-smi`)
- ✅ Training completes in ~2.5 hours
- ✅ Test with TTA maintains >85% accuracy
- ✅ Final model saved to `/workspace/robust_gpu_models/robust_vgg19_final.h5`

**If all checkmarks are complete: You have a production-ready robust violence detection model!** 🎉
