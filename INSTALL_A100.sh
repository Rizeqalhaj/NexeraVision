#!/bin/bash
# A100 Installation - TensorFlow 2.20.0

echo "================================================================================"
echo "A100 SETUP - TensorFlow 2.20.0 Installation"
echo "================================================================================"

# Install TensorFlow 2.20.0 (latest)
echo ""
echo "Installing TensorFlow 2.20.0..."
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow==2.20.0

# Install other packages
echo ""
echo "Installing dependencies..."
python3 -m pip install opencv-python numpy pandas scikit-learn tqdm h5py Pillow matplotlib seaborn

# Verify
echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"

python3 << 'EOF'
import tensorflow as tf
import cv2
import numpy as np

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ OpenCV: {cv2.__version__}")
print(f"✅ NumPy: {np.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\n🎮 GPUs detected: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
        # Enable memory growth
        tf.config.experimental.set_memory_growth(gpu, True)
    print("\n✅ Memory growth enabled")
    print("🚀 READY TO TRAIN!")
else:
    print("⚠️  No GPUs detected")
EOF

echo ""
echo "================================================================================"
echo "✅ Installation Complete!"
echo "================================================================================"
