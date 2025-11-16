#!/bin/bash
# Simple GPU Fix - Install TensorFlow without TensorRT issues

echo "================================================================================"
echo "SIMPLE GPU FIX"
echo "================================================================================"

# Check CUDA is available
echo ""
echo "Checking CUDA installation..."
nvcc --version 2>/dev/null || echo "CUDA compiler not in PATH (this is OK)"
ldconfig -p | grep cuda | head -5

# Install TensorFlow (CUDA libraries should already be in the container)
echo ""
echo "Installing TensorFlow 2.15.0..."
pip3 uninstall -y tensorflow tensorflow-gpu tf-nightly tensorrt
pip3 install --upgrade pip setuptools wheel
pip3 install tensorflow==2.15.0

# Test
echo ""
echo "Testing GPU detection..."
python3 << 'EOF'
import tensorflow as tf
print("\nTensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs detected: {len(gpus)}")

if gpus:
    print("✅ SUCCESS!")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

    # Enable memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("\n✅ Memory growth enabled")
else:
    print("\n❌ No GPUs detected")
    print("Vast.ai container may need restart")
EOF

echo ""
echo "================================================================================"
echo "Done! If successful, run: python3 /workspace/FAST_RESUME.py"
echo "================================================================================"
