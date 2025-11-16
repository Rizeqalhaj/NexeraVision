# A100 GPU Training Environment Setup

Complete installation guide for NVIDIA A100 SXM4 80GB GPU training environment for violence detection model.

---

## Quick Start (TL;DR)

```bash
# 1. Run the installation script
bash COMPLETE_A100_INSTALL.sh

# 2. Verify installation
python3 /tmp/test_training_setup.py

# 3. Monitor GPU
watch -n 1 nvidia-smi

# 4. Start training
python3 scripts/train_model.py --batch-size 32 --epochs 50
```

---

## System Information

### Hardware Configuration
- **GPU**: NVIDIA A100 SXM4 80GB
- **CUDA Version**: 13.0
- **Memory**: 80GB GPU RAM
- **Compute Capability**: 8.0
- **TensorCores**: 432
- **Peak Performance**: 312 TFLOPS (FP16)

### Software Stack
- **Python**: 3.8+ (auto-detected)
- **TensorFlow**: 2.15+ with GPU support
- **CUDA**: 12.0+ (bundled with TensorFlow)
- **cuDNN**: 8.6+ (bundled with TensorFlow)

---

## Installation Process

### Prerequisites

Before running the installation script, ensure:

1. **Python 3.8+ is installed**
   ```bash
   python3 --version
   # Should show: Python 3.8.x or higher
   ```

2. **NVIDIA drivers are installed**
   ```bash
   nvidia-smi
   # Should show: A100 SXM4 80GB
   ```

3. **Sufficient disk space**
   ```bash
   df -h
   # Need: 10GB+ for dependencies, 50GB+ for training data
   ```

### Step 1: Download and Run Script

```bash
# Navigate to project directory
cd /home/admin/Desktop/NexaraVision

# Make script executable
chmod +x COMPLETE_A100_INSTALL.sh

# Run installation
bash COMPLETE_A100_INSTALL.sh
```

### Step 2: Installation Progress

The script will proceed through 8 steps:

```
[Step 1/8] Detecting System Information
  → Python version detection
  → CUDA toolkit detection
  → GPU detection
  → pip verification

[Step 2/8] Determining Compatible TensorFlow Version
  → Python 3.8-3.11: TensorFlow 2.15+
  → Python 3.12+: TensorFlow 2.16+

[Step 3/8] Virtual Environment Check
  → Recommends venv (optional but recommended)

[Step 4/8] Installing Core Dependencies
  → TensorFlow (GPU support)
  → OpenCV 4.8+
  → NumPy, Pandas, scikit-learn
  → Visualization tools
  → GPU monitoring tools

[Step 5/8] Verifying TensorFlow Installation
  → Import test
  → GPU detection test
  → CUDA/cuDNN version check

[Step 6/8] Verifying All Dependencies
  → Tests all package imports
  → Checks version compatibility

[Step 7/8] Testing GPU Detection and Configuration
  → GPU computation test
  → Memory growth configuration
  → Performance test

[Step 8/8] Creating Training Verification Script
  → Creates test_training_setup.py
  → Runs environment validation
```

**Total Time**: 5-15 minutes (depending on internet speed)

### Step 3: Verify Installation

After installation completes, run the verification script:

```bash
python3 /tmp/test_training_setup.py
```

**Expected Output**:
```
═══════════════════════════════════════════════════════════
NexaraVision Training Environment Test
═══════════════════════════════════════════════════════════

[1] TensorFlow + GPU:
    TensorFlow version: 2.15.x
    GPUs available: 1
      GPU 0: /physical_device:GPU:0

[2] ResNet50V2 Model:
    ✅ ResNet50V2 loaded (23,564,800 params)

[3] BiGRU Layer:
    ✅ BiGRU functional (output shape: (1, 256))

[4] OpenCV:
    OpenCV version: 4.8.x
    ✅ Image operations working

[5] Data Pipeline:
    ✅ Train: (80, 10, 224, 224, 3), Val: (20, 10, 224, 224, 3)

[6] Memory Estimation:
    GPU Memory: 80.00 GB total
    Available: 78.50 GB
    Recommended batch size: 32-48

═══════════════════════════════════════════════════════════
✅ ALL TESTS PASSED - Environment ready for training
═══════════════════════════════════════════════════════════
```

---

## Installed Dependencies

The script automatically installs:

### Core ML Framework
- **tensorflow[and-cuda]** ≥2.15.0
  - Includes CUDA 12.x and cuDNN 8.9
  - GPU support enabled by default

### Computer Vision
- **opencv-python** ≥4.8.0 - Core OpenCV
- **opencv-contrib-python** ≥4.8.0 - Additional modules

### Numerical Computing
- **numpy** ≥1.24.0, <2.0.0 - Arrays and matrices
- **pandas** ≥2.0.0 - Data manipulation

### Machine Learning
- **scikit-learn** ≥1.3.0 - ML utilities and metrics

### Visualization
- **matplotlib** ≥3.7.0 - Plotting
- **seaborn** ≥0.12.0 - Statistical visualization

### Utilities
- **tqdm** ≥4.66.0 - Progress bars
- **h5py** ≥3.9.0 - Model saving/loading
- **Pillow** ≥10.0.0 - Image processing
- **psutil** ≥5.9.0 - System monitoring

### GPU Monitoring
- **gpustat** ≥1.1.0 - GPU status tool
- **pynvml** ≥11.5.0 - NVIDIA Management Library

### Additional
- **pyyaml** ≥6.0.0 - Configuration files
- **requests** ≥2.31.0 - HTTP library

---

## GPU Configuration

### Memory Growth (Recommended for A100)

The installation script automatically configures memory growth. To use in your training script:

```python
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
```

### Why Memory Growth?

- **Without**: TensorFlow allocates ALL 80GB upfront
- **With**: TensorFlow allocates memory as needed
- **Benefit**: Allows multiple experiments, prevents OOM errors

### Batch Size Recommendations

Based on A100 80GB memory:

| Model Component | Batch Size | Memory Usage | Throughput |
|-----------------|------------|--------------|------------|
| ResNet50V2 + BiGRU | 16 | ~25GB | Conservative |
| ResNet50V2 + BiGRU | 32 | ~45GB | **Recommended** |
| ResNet50V2 + BiGRU | 48 | ~65GB | Optimal |
| ResNet50V2 + BiGRU | 64 | ~75GB | Aggressive |

**Start with batch size 32**, then increase while monitoring `nvidia-smi`.

---

## Performance Optimization

### 1. Mixed Precision Training

Enable for 2-3x speedup on A100:

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print(f"Compute dtype: {policy.compute_dtype}")  # float16
print(f"Variable dtype: {policy.variable_dtype}")  # float32
```

**Benefits**:
- 2-3x faster training
- 50% less memory usage
- Same accuracy with proper loss scaling

### 2. XLA Compilation

Enable XLA for additional 15-30% speedup:

```python
# In model.compile()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
    jit_compile=True  # Enable XLA
)
```

### 3. Data Pipeline Optimization

```python
AUTOTUNE = tf.data.AUTOTUNE

dataset = dataset.cache()  # Cache in memory
dataset = dataset.prefetch(buffer_size=AUTOTUNE)  # Prefetch next batch
dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)  # Parallel preprocessing
```

### 4. Multi-GPU Strategy (if available)

```python
# For multiple A100s
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = build_model()
    model.compile(...)
```

---

## Monitoring and Debugging

### GPU Monitoring Commands

```bash
# Real-time GPU monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Detailed GPU stats with gpustat
gpustat -i 1

# Log GPU utilization to file
nvidia-smi dmon -s u > gpu_usage.log &

# Check GPU temperature and power
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv --loop=1

# Kill background monitoring
pkill nvidia-smi
```

### TensorFlow GPU Debugging

```python
# Check GPU visibility
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Check GPU memory usage
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Enable GPU memory logging
tf.debugging.set_log_device_placement(True)
```

### Common Issues and Solutions

#### Issue 1: GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not working, check driver installation
lsmod | grep nvidia

# Verify CUDA is accessible
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Solution**: Reinstall NVIDIA drivers or check CUDA installation.

#### Issue 2: Out of Memory (OOM)

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions**:
1. Reduce batch size: `--batch-size 16`
2. Enable memory growth (already configured)
3. Use gradient accumulation
4. Reduce sequence length

#### Issue 3: Slow Training

**Check**:
```bash
# GPU utilization should be 90%+
nvidia-smi

# If low utilization:
# 1. Enable mixed precision
# 2. Enable XLA compilation
# 3. Optimize data pipeline
# 4. Increase batch size
```

---

## Training Workflow

### Step 1: Prepare Dataset

```bash
# Organize videos
dataset/
├── violence/
│   ├── video001.mp4
│   └── video002.mp4
└── non_violence/
    ├── video001.mp4
    └── video002.mp4
```

### Step 2: Extract Frames (if needed)

```bash
python3 scripts/preprocess_videos.py \
    --input dataset/ \
    --output processed/ \
    --frames-per-video 20 \
    --img-size 224
```

### Step 3: Configure Training

Create `training_config.json`:
```json
{
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "frames_per_video": 20,
    "img_height": 224,
    "img_width": 224,
    "sequence_model": "Bidirectional-GRU",
    "gru_units": 128,
    "mixed_precision": true,
    "xla_compile": true
}
```

### Step 4: Start Training

```bash
# Terminal 1: Run training
python3 scripts/train_model.py \
    --config training_config.json \
    --data processed/ \
    --output models/

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: Monitor logs
tail -f logs/training.log
```

### Step 5: Monitor Progress

```bash
# Check training metrics
tensorboard --logdir=logs/tensorboard --port=6006

# View in browser
# http://localhost:6006
```

---

## Benchmark Expectations

### Training Performance (A100 80GB)

| Configuration | Batch Size | Speed | Memory | Accuracy (50 epochs) |
|---------------|------------|-------|--------|---------------------|
| **Baseline** | 16 | ~120 samples/s | 25GB | 85-90% |
| **Recommended** | 32 | ~200 samples/s | 45GB | 85-90% |
| **Optimal** | 48 | ~280 samples/s | 65GB | 85-90% |
| **Mixed Precision** | 32 | ~400 samples/s | 25GB | 85-90% |
| **Mixed + XLA** | 32 | ~500 samples/s | 25GB | 85-90% |

### Expected Training Times

Dataset: 10,000 video sequences (20 frames each)

| Configuration | Time per Epoch | Total Time (50 epochs) |
|---------------|----------------|------------------------|
| Baseline (BS=16) | ~8 min | ~6.5 hours |
| Recommended (BS=32) | ~5 min | ~4 hours |
| Optimal (BS=48) | ~3.5 min | ~3 hours |
| Mixed Precision (BS=32) | ~2.5 min | **~2 hours** |

---

## Troubleshooting

### Re-run Installation

The script is idempotent (safe to run multiple times):

```bash
bash COMPLETE_A100_INSTALL.sh
```

### Force Reinstall Dependencies

```bash
pip3 install --force-reinstall tensorflow[and-cuda]>=2.15.0
```

### Check Installation Log

```bash
# Find latest installation log
ls -lt /tmp/nexaravision_install_*.log | head -1

# View log
cat /tmp/nexaravision_install_*.log
```

### Clean Installation

```bash
# Remove all installed packages
pip3 uninstall -y tensorflow opencv-python numpy pandas scikit-learn \
    matplotlib seaborn tqdm h5py pillow psutil gpustat pynvml

# Re-run installation
bash COMPLETE_A100_INSTALL.sh
```

---

## Advanced Configuration

### Custom TensorFlow Build

For maximum performance, build TensorFlow from source:

```bash
# Install Bazel
wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-installer-linux-x86_64.sh
bash bazel-6.1.0-installer-linux-x86_64.sh --user

# Clone TensorFlow
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

# Configure for A100
./configure

# Build with optimizations
bazel build --config=cuda --config=opt //tensorflow/tools/pip_package:build_pip_package

# Create wheel
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Install
pip3 install /tmp/tensorflow_pkg/tensorflow-*.whl
```

### CUDA Environment Variables

Add to `~/.bashrc` for optimal performance:

```bash
# CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# TensorFlow optimizations
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1

# XLA optimizations
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
```

---

## Resources

### Documentation
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [NVIDIA A100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)

### Optimization Guides
- [TensorFlow Performance Guide](https://www.tensorflow.org/guide/profiler)
- [Mixed Precision Training](https://www.tensorflow.org/guide/mixed_precision)
- [XLA Compilation](https://www.tensorflow.org/xla)

### Community
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## Summary Checklist

Before starting training, verify:

- [ ] Installation script completed successfully
- [ ] Test script shows GPU detected
- [ ] `nvidia-smi` shows A100 GPU
- [ ] TensorFlow imports without errors
- [ ] Virtual environment activated (recommended)
- [ ] Dataset prepared and accessible
- [ ] Training script configured
- [ ] Monitoring tools ready (nvidia-smi, tensorboard)
- [ ] Sufficient disk space (50GB+)
- [ ] Backup strategy in place

---

## Quick Reference Card

```bash
# Installation
bash COMPLETE_A100_INSTALL.sh

# Verification
python3 /tmp/test_training_setup.py

# GPU Status
nvidia-smi
gpustat -i 1

# Training
python3 scripts/train_model.py --batch-size 32 --epochs 50

# Monitoring
watch -n 1 nvidia-smi
tail -f logs/training.log
tensorboard --logdir=logs

# Troubleshooting
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

**Last Updated**: 2025-11-15
**Script Version**: 1.0
**Supported GPUs**: NVIDIA A100 SXM4 80GB
**Supported CUDA**: 12.0+, 13.0
**Supported Python**: 3.8, 3.9, 3.10, 3.11, 3.12
