#!/bin/bash
# Quick TensorFlow Installation Fix

echo "================================================================================"
echo "TENSORFLOW INSTALLATION FIX"
echo "================================================================================"

# Check Python version
echo ""
echo "Python version:"
python3 --version
which python3

# Upgrade pip
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install TensorFlow
echo ""
echo "Installing TensorFlow 2.15.0..."
python3 -m pip install tensorflow==2.15.0

# Install other required packages
echo ""
echo "Installing other packages..."
python3 -m pip install opencv-python==4.8.1.78 numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 tqdm==4.66.1 h5py==3.9.0 Pillow==10.0.1

# Verify
echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")

try:
    import tensorflow as tf
    print(f"\n✅ TensorFlow: {tf.__version__}")
    print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")

    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n✅ GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")

    if len(gpus) > 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n✅ Memory growth enabled")
        print("\n🚀 READY TO TRAIN!")
    else:
        print("\n⚠️  No GPUs detected")

except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("   TensorFlow installation failed")

try:
    import cv2
    print(f"\n✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("\n❌ OpenCV not found")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError:
    print("❌ NumPy not found")
EOF

echo ""
echo "================================================================================"
echo "Done!"
echo "================================================================================"
