#!/bin/bash
# Fix TensorFlow GPU Detection

echo "================================================================================"
echo "GPU FIX SCRIPT - TensorFlow CUDA Setup"
echo "================================================================================"

# Check hardware GPUs
echo ""
echo "Step 1: Checking GPU hardware..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check CUDA installation
echo ""
echo "Step 2: Checking CUDA libraries..."
ls -lh /usr/local/cuda*/lib64/libcudnn* 2>/dev/null || echo "⚠️  cuDNN not found"
ls -lh /usr/local/cuda*/lib64/libcublas* 2>/dev/null | head -3

# Uninstall old TensorFlow
echo ""
echo "Step 3: Removing old TensorFlow..."
pip3 uninstall -y tensorflow tensorflow-gpu tf-nightly

# Install TensorFlow with CUDA
echo ""
echo "Step 4: Installing TensorFlow with CUDA support..."
pip3 install --upgrade pip
pip3 install tensorflow[and-cuda]==2.15.0

# Verify installation
echo ""
echo "Step 5: Verifying GPU detection..."
python3 << 'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"\n{'=' * 80}")
if len(gpus) > 0:
    print(f"✅ SUCCESS! Detected {len(gpus)} GPU(s)")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
    print(f"{'=' * 80}")
else:
    print("❌ Still no GPUs detected")
    print("   This may require container restart")
    print(f"{'=' * 80}")
EOF

echo ""
echo "================================================================================"
echo "If GPUs still not detected, run: exit + reconnect to vast.ai"
echo "================================================================================"
