# NexaraVision A100 GPU Installation Package

Complete automated installation system for NVIDIA A100 SXM4 80GB training environment.

---

## Quick Start (3 Steps)

```bash
# 1. Install (5-15 min)
bash COMPLETE_A100_INSTALL.sh

# 2. Verify (1 min)
python3 /tmp/test_training_setup.py

# 3. Train
python3 scripts/train_model.py --batch-size 32 --epochs 50
```

**Expected result:** Model training in ~2 hours with mixed precision on 10K dataset.

---

## Package Contents

| File | Size | Purpose |
|------|------|---------|
| **COMPLETE_A100_INSTALL.sh** | 19KB | Main installation script (recommended) |
| **A100_SETUP_README.md** | 15KB | Complete documentation and guides |
| **QUICK_START_A100.md** | 2.4KB | 5-minute quick reference |
| **A100_INSTALLATION_SUMMARY.txt** | 18KB | Comprehensive overview |
| **A100_CHECKLIST.txt** | 4KB | Step-by-step checklist |
| **INSTALLATION_GUIDE.md** | 7KB | Multi-method installation guide |

---

## Documentation Levels

Choose based on your needs:

### Level 1: Quick Start (5 minutes)
**File:** `QUICK_START_A100.md`
- Essential commands only
- Copy-paste ready
- Performance tips

**Best for:** Experienced users who need fast setup

### Level 2: Checklist (10 minutes)
**File:** `A100_CHECKLIST.txt`
- Step-by-step checklist
- Verification steps
- Troubleshooting quick fixes

**Best for:** Systematic installation tracking

### Level 3: Summary (15 minutes)
**File:** `A100_INSTALLATION_SUMMARY.txt`
- High-level overview
- Performance benchmarks
- Configuration examples

**Best for:** Understanding scope and expectations

### Level 4: Complete Guide (30 minutes)
**File:** `A100_SETUP_README.md`
- Detailed documentation
- Advanced optimization
- Comprehensive troubleshooting

**Best for:** First-time setup, deep optimization

---

## Installation Script Features

### Auto-Detection
- Python version (3.8-3.12)
- CUDA toolkit (12.0+, 13.0)
- GPU availability (A100 SXM4 80GB)
- Compatible TensorFlow version

### Installation Process (8 Steps)
1. System information detection
2. TensorFlow version determination
3. Virtual environment check
4. Core dependencies installation
5. TensorFlow verification
6. Dependency verification
7. GPU configuration test
8. Training environment test

### Error Handling
- Comprehensive validation
- Clear error messages
- Graceful failure recovery
- Installation logging

### Idempotent Design
- Safe to run multiple times
- Updates outdated packages
- Preserves existing installations

---

## What Gets Installed

### Core Framework
- TensorFlow 2.15+ (GPU support, CUDA 12.x, cuDNN 8.9)

### Computer Vision
- OpenCV 4.8+ (core + contrib)

### Scientific Computing
- NumPy 1.24+
- Pandas 2.0+
- scikit-learn 1.3+

### Visualization
- Matplotlib 3.7+
- Seaborn 0.12+

### Utilities
- tqdm 4.66+ (progress bars)
- h5py 3.9+ (model I/O)
- Pillow 10.0+ (image processing)
- psutil (system monitoring)

### GPU Monitoring
- gpustat 1.1+ (GPU stats)
- pynvml 11.5+ (NVIDIA library)

---

## Performance Expectations

### Training Speed (A100 80GB, 10K dataset)

| Configuration | Batch | Speed | Memory | Time (50 epochs) |
|---------------|-------|-------|--------|------------------|
| Baseline | 16 | 120 samp/s | 25GB | 6.5 hours |
| Recommended | 32 | 200 samp/s | 45GB | 4 hours |
| Optimal | 48 | 280 samp/s | 65GB | 3 hours |
| Mixed Precision | 32 | 400 samp/s | 25GB | 2.5 hours |
| **Mixed + XLA** | **32** | **500 samp/s** | **25GB** | **2 hours** ⭐ |

### Optimization Impact
- Mixed Precision: **2-3x faster**
- XLA Compilation: **+15-30%**
- Batch Size 48: **+40%**
- **Combined: ~4x faster than baseline**

---

## Recommended Configuration

Add to your training script for optimal performance:

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

# 1. Enable memory growth (prevent OOM)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 2. Enable mixed precision (2-3x speedup)
mixed_precision.set_global_policy('mixed_float16')

# 3. Use recommended batch size
BATCH_SIZE = 32  # Conservative, increase to 48 for optimal

# 4. Enable XLA compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
    jit_compile=True  # +15-30% speedup
)
```

---

## Batch Size Guide

| Batch Size | Memory | Stability | Performance | Recommendation |
|------------|--------|-----------|-------------|----------------|
| 16 | ~25GB | High | Baseline | Conservative |
| 32 | ~45GB | High | Good | **Recommended** ⭐ |
| 48 | ~65GB | Medium | Better | Optimal |
| 64 | ~75GB | Low | Best | Aggressive |

**Strategy:** Start with 32, monitor `nvidia-smi`, increase if GPU < 90% utilized

---

## Verification Tests

The installation script automatically verifies:

1. **TensorFlow + GPU**
   - TensorFlow version
   - GPU detection (1 GPU expected)
   - CUDA/cuDNN versions

2. **Model Architecture**
   - ResNet50V2 loading (23.5M params)
   - BiGRU layer functionality

3. **OpenCV**
   - Version check (4.8+)
   - Image operations

4. **Data Pipeline**
   - Train/validation split
   - Batch processing

5. **Memory Estimation**
   - GPU memory detection (80GB)
   - Recommended batch size

6. **All Dependencies**
   - Import verification
   - Version compatibility

---

## Troubleshooting

### GPU Not Detected

```bash
# Check driver
nvidia-smi

# Check TensorFlow GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reinstall if needed
bash COMPLETE_A100_INSTALL.sh
```

### Out of Memory

```bash
# Reduce batch size
python3 scripts/train_model.py --batch-size 16
```

### Slow Training

```python
# Enable optimizations
mixed_precision.set_global_policy('mixed_float16')  # 2-3x
model.compile(..., jit_compile=True)  # +15-30%
BATCH_SIZE = 48  # +40%
```

---

## Monitoring Commands

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU stats
gpustat -i 1

# Training logs
tail -f logs/training.log

# TensorBoard
tensorboard --logdir=logs --port=6006
```

---

## Success Criteria

Installation successful when:

- ✅ All 6 verification tests pass
- ✅ GPU detected by TensorFlow
- ✅ `nvidia-smi` shows A100 80GB
- ✅ Training runs without errors
- ✅ GPU utilization > 90%
- ✅ No OOM errors with batch size 32

---

## System Requirements

### Minimum
- Python 3.8+
- CUDA 11.2+
- cuDNN 8.6+
- 16GB RAM
- 100GB disk space

### Recommended (A100)
- Python 3.10+
- CUDA 12.0+ (13.0 for A100)
- A100 SXM4 80GB
- 128GB+ RAM
- 500GB+ SSD

---

## Complete Workflow

1. **Install** (5-15 min)
   ```bash
   bash COMPLETE_A100_INSTALL.sh
   ```

2. **Verify** (1 min)
   ```bash
   python3 /tmp/test_training_setup.py
   ```

3. **Prepare Dataset** (10-30 min)
   ```bash
   python3 scripts/preprocess_videos.py --input dataset/ --output processed/
   ```

4. **Configure Training**
   - Edit `training_config.json`
   - Add GPU optimizations to training script

5. **Train** (2-6 hours)
   ```bash
   python3 scripts/train_model.py --batch-size 32 --epochs 50
   ```

6. **Monitor**
   - GPU: `watch -n 1 nvidia-smi`
   - Logs: `tail -f logs/training.log`
   - TensorBoard: `tensorboard --logdir=logs`

---

## Support Resources

### Documentation
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [NVIDIA A100 Datasheet](https://www.nvidia.com/en-us/data-center/a100/)
- [Mixed Precision Training](https://www.tensorflow.org/guide/mixed_precision)

### Project Files
- **Quick Start**: `QUICK_START_A100.md`
- **Checklist**: `A100_CHECKLIST.txt`
- **Complete Guide**: `A100_SETUP_README.md`
- **Summary**: `A100_INSTALLATION_SUMMARY.txt`

---

## Version Info

- **Created:** 2025-11-15
- **Version:** 1.0
- **Target GPU:** NVIDIA A100 SXM4 80GB
- **Supported Python:** 3.8, 3.9, 3.10, 3.11, 3.12
- **TensorFlow:** 2.15+

---

## License

NexaraVision Violence Detection Project

---

**Ready to install? Run:** `bash COMPLETE_A100_INSTALL.sh`
