# Quick Reference Card - Vast.ai GPU Training

## 📦 Files to Upload

Upload these 3 files to `/workspace/violence_detection_mvp/`:

```
✅ gpu_video_loader.py
✅ train_robust_gpu_accelerated.py
✅ benchmark_video_loading.py
```

---

## 🚀 Commands (Copy-Paste)

### 1. Find Your Videos

```bash
find /workspace -name "*.mp4" -type f | head -5
```

### 2. Run Benchmark

```bash
cd /workspace/violence_detection_mvp

# Replace VIDEO_PATH with path from step 1
python3 benchmark_video_loading.py /workspace/data/train/Fight/fight_001.mp4
```

### 3. Start Training

```bash
cd /workspace/violence_detection_mvp
python3 train_robust_gpu_accelerated.py
```

### 4. Monitor GPU (In Another Terminal)

```bash
watch -n 1 nvidia-smi
```

---

## 📊 Expected Performance

| Metric | Old (OpenCV) | New (TensorFlow GPU) |
|--------|--------------|----------------------|
| Speed | 5-7 vid/sec | 100-300 vid/sec |
| GPU Util | 0-5% | 60-70% |
| Time | 12 hours | 2.5 hours |

---

## ✅ Success Indicators

During training, you should see:

```
✅ GPU Video Loader initialized: tensorflow
✅ GPU TRAIN: 100%|██████████| 176780/176780 [25:32<00:00, 115.4video/s]
✅ GPU Utilization: 60-70% (check nvidia-smi)
✅ Total time: ~2.5 hours
```

---

## 🐛 Quick Fixes

### TensorFlow too old

```bash
pip install --upgrade tensorflow>=2.13.0
```

### Missing OpenCV

```bash
pip install opencv-python
```

### Out of memory

Edit `train_robust_gpu_accelerated.py` line ~154:
```python
gpu_batch_size = 16  # Reduce from 32
```

---

## 📂 Output Location

Final model will be saved to:
```
/workspace/robust_gpu_models/robust_vgg19_final.h5
```

---

## 📞 Troubleshooting

Check TensorFlow GPU support:
```bash
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

Check video decode support:
```bash
python3 -c "import tensorflow as tf; print('Video decode:', hasattr(tf.io, 'decode_video'))"
```

---

## 🎯 After Training

Test robustness with TTA:
```bash
python3 predict_with_tta_simple.py \
    --model /workspace/robust_gpu_models/robust_vgg19_final.h5 \
    --video /workspace/data/test/Fight/test_video.mp4
```

**Expected:** TTA accuracy stays > 85% (proves robustness)

---

For detailed guides, see:
- `GPU_ACCELERATED_TRAINING_GUIDE.md`
- `VASTAI_SETUP_INSTRUCTIONS.md`
- `SOLUTION_SUMMARY.md`
