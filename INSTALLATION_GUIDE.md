# Installation Guide - NexaraVision Training Environment

## 🎯 Target Hardware

### A100 GPU Environment (Primary)
- **GPU**: NVIDIA A100 SXM4 80GB
- **CUDA**: 13.0
- **Python**: 3.8+
- **TensorFlow**: 2.15+ (with GPU support)

### Alternative: RTX 6000 Ada (Previous Setup)
- **GPU**: 2x RTX 6000 Ada (48GB each, 96GB total)
- **RAM**: 257.9 GB
- **Storage**: 500 GB SSD

---

## 📦 Installation Methods (Choose One)

### ⭐ Method 1: A100 Complete Install Script (Recommended for A100)

```bash
bash COMPLETE_A100_INSTALL.sh
```

**What it does**:
- Auto-detects Python version and system configuration
- Installs compatible TensorFlow version (2.15+)
- Configures GPU memory growth for A100
- Verifies all dependencies and GPU detection
- Runs comprehensive environment tests
- Provides detailed summary and recommendations

**Time**: 5-15 minutes
**Best for**: A100 GPU environments, first-time setup

---

### Method 2: One-Line Quick Install (Fastest)

```bash
bash QUICK_INSTALL.sh
```

**What it does**: Installs all dependencies in one command
**Time**: 2-3 minutes

---

### Method 3: Comprehensive Install Script (RTX 6000 Ada)

```bash
chmod +x INSTALL_TRAINING_DEPENDENCIES.sh
bash INSTALL_TRAINING_DEPENDENCIES.sh
```

**What it does**:
- Installs all dependencies with progress feedback
- Verifies TensorFlow GPU support
- Shows detailed installation summary

**Time**: 3-5 minutes

---

### Method 4: pip requirements.txt

```bash
pip3 install -r requirements_training.txt
```

**What it does**: Standard pip installation from requirements file
**Time**: 2-3 minutes

---

### Method 5: Python Setup Script

```bash
python3 setup_training_env.py
```

**What it does**:
- Installs dependencies
- Checks datasets
- Verifies GPU
- Creates workspace structure
- Generates training config

**Time**: 5-10 minutes

---

## 📋 Complete Installation Checklist

### Step 1: Connect to Vast.ai

```bash
ssh root@195.142.145.66
```

### Step 2: Update System (Optional)

```bash
apt-get update
apt-get install -y wget curl git
```

### Step 3: Install Dependencies

**Choose your preferred method above**, then verify:

```bash
# Verify TensorFlow
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Verify GPU
python3 -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# Verify OpenCV
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

**Expected Output**:
```
TensorFlow: 2.15.0
GPUs: 2
OpenCV: 4.8.1
```

### Step 4: Upload Training Files

Upload to `/workspace/`:
```
train_model_RESUME.py
model_architecture.py
data_preprocessing.py
training_config.json
```

### Step 5: Verify Data

```bash
# Check pre-extracted frames
find /workspace/processed/frames/ -name "*.npy" | wc -l
# Should show: 10738

# Check splits file
cat /workspace/processed/splits.json | head -10

# Check disk space
df -h /workspace
```

### Step 6: Start Training

```bash
cd /workspace
python3 train_model_RESUME.py
```

---

## 🔧 Dependency Details

### Core Packages

| Package | Version | Purpose |
|---------|---------|---------|
| **tensorflow** | 2.15.0 | Deep learning framework (GPU support) |
| **opencv-python** | 4.8.1.78 | Video processing and frame extraction |
| **numpy** | 1.24.3 | Numerical computing |
| **pandas** | 2.0.3 | Data manipulation |
| **scikit-learn** | 1.3.0 | ML utilities and metrics |
| **matplotlib** | 3.7.3 | Visualization |
| **tqdm** | 4.66.1 | Progress bars |
| **h5py** | 3.9.0 | Model saving/loading |

### GPU Monitoring Tools

```bash
# Install if not included
pip3 install gpustat pynvml

# Monitor GPUs
gpustat -i 1  # Updates every 1 second
```

---

## ✅ Verification Tests

### Test 1: TensorFlow GPU

```python
python3 << EOF
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("CUDA Available:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')
print(f"\nDetected {len(gpus)} GPU(s):")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

# Test GPU computation
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("\n✅ GPU computation test passed!")
    print("Result:", c.numpy())
EOF
```

**Expected Output**:
```
TensorFlow Version: 2.15.0
CUDA Available: True

Detected 2 GPU(s):
  GPU 0: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
  GPU 1: PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')

✅ GPU computation test passed!
```

### Test 2: OpenCV Video Processing

```python
python3 << EOF
import cv2
import numpy as np

print("OpenCV Version:", cv2.__version__)

# Create test video
test_frame = np.zeros((224, 224, 3), dtype=np.uint8)
print("✅ OpenCV working - can process frames")
EOF
```

### Test 3: Load Model Architecture

```python
python3 << EOF
from model_architecture import ViolenceDetectionModel

model_builder = ViolenceDetectionModel(
    frames_per_video=20,
    sequence_model='Bidirectional-GRU',
    gru_units=128
)

model = model_builder.build_model(trainable_backbone=False)
print(f"✅ Model created successfully!")
print(f"Parameters: {sum(p.numel() for p in model.trainable_weights) / 1e6:.1f}M")
EOF
```

---

## 🆘 Troubleshooting

### Problem: "No module named 'tensorflow'"

**Solution**:
```bash
pip3 install tensorflow==2.15.0
```

### Problem: "ImportError: libGL.so.1"

**Solution**:
```bash
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Problem: "No GPU detected"

**Check**:
```bash
nvidia-smi  # Should show 2x RTX 6000 Ada

# If not working:
apt-get install -y nvidia-driver-535 nvidia-utils-535
reboot
```

### Problem: TensorFlow not using GPU

**Solution**:
```bash
# Check CUDA
ls /usr/local/cuda*/lib64/libcudnn*

# Reinstall TensorFlow
pip3 uninstall tensorflow
pip3 install tensorflow==2.15.0
```

### Problem: Out of disk space

**Check**:
```bash
df -h /workspace

# Clean up if needed
rm -rf /workspace/logs/training/old_*
```

---

## 📊 System Requirements

### Minimum Requirements
- Python 3.9+
- CUDA 11.8+
- cuDNN 8.6+
- 16 GB RAM
- 100 GB free disk space

### Recommended (Your Setup)
- ✅ Python 3.10
- ✅ CUDA 12.4
- ✅ 2x RTX 6000 Ada (96GB VRAM)
- ✅ 257.9 GB RAM
- ✅ 500 GB SSD

---

## 🚀 Quick Reference Commands

```bash
# Install all dependencies
bash INSTALL_TRAINING_DEPENDENCIES.sh

# Verify installation
python3 -c "import tensorflow as tf; print(f'TF: {tf.__version__}, GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# Check data
find /workspace/processed/frames/ -name "*.npy" | wc -l

# Monitor GPU
watch -n 1 nvidia-smi

# Start training
python3 /workspace/train_model_RESUME.py

# Monitor progress
tail -f /workspace/logs/training/initial_training_log.csv
```

---

## 📞 Support

**Installation Issues**: Check error logs
**GPU Issues**: Run `nvidia-smi` and verify drivers
**Dependency Issues**: Try `pip3 install --force-reinstall <package>`

---

**Installation Time**: 5-10 minutes total
**Next**: Upload training scripts and start training!
