#!/bin/bash
################################################################################
# NexaraVision Training Dependencies Installation
# For: 2x RTX 6000 Ada (96GB VRAM)
# Model: ResNet50V2 + Bidirectional GRU
################################################################################

echo "================================================================================"
echo "NexaraVision - Installing Training Dependencies"
echo "================================================================================"
echo "Target: 2x RTX 6000 Ada (96GB VRAM)"
echo "Model: ResNet50V2 + Bidirectional GRU"
echo "Python: $(python3 --version)"
echo "================================================================================"
echo ""

# Update pip first
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip
echo "✅ pip updated"
echo ""

# Core Deep Learning Framework
echo "================================================================================"
echo "🧠 Installing TensorFlow (GPU support)"
echo "================================================================================"
pip3 install tensorflow==2.15.0
echo "✅ TensorFlow 2.15.0 installed"
echo ""

# Computer Vision
echo "================================================================================"
echo "📷 Installing Computer Vision Libraries"
echo "================================================================================"
pip3 install opencv-python==4.8.1.78
pip3 install opencv-contrib-python==4.8.1.78
pip3 install Pillow==10.0.1
echo "✅ OpenCV and Pillow installed"
echo ""

# Scientific Computing
echo "================================================================================"
echo "🔬 Installing Scientific Libraries"
echo "================================================================================"
pip3 install numpy==1.24.3
pip3 install pandas==2.0.3
pip3 install scikit-learn==1.3.0
pip3 install scipy==1.11.3
echo "✅ NumPy, Pandas, scikit-learn installed"
echo ""

# Visualization
echo "================================================================================"
echo "📊 Installing Visualization Libraries"
echo "================================================================================"
pip3 install matplotlib==3.7.3
pip3 install seaborn==0.12.2
echo "✅ Matplotlib and Seaborn installed"
echo ""

# Progress Bars and Utilities
echo "================================================================================"
echo "🛠️  Installing Utilities"
echo "================================================================================"
pip3 install tqdm==4.66.1
pip3 install h5py==3.9.0
pip3 install psutil==5.9.5
echo "✅ Utilities installed"
echo ""

# CUDA/GPU Monitoring (Optional but useful)
echo "================================================================================"
echo "🎮 Installing GPU Monitoring Tools"
echo "================================================================================"
pip3 install gpustat==1.1.1
pip3 install pynvml==11.5.0
echo "✅ GPU monitoring tools installed"
echo ""

# Verify TensorFlow GPU
echo "================================================================================"
echo "🔍 Verifying TensorFlow GPU Support"
echo "================================================================================"
python3 << 'EOF'
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
print(f"CUDA Available: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs Detected: {len(gpus)}")

for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu.name}")

if len(gpus) > 0:
    print("\n✅ GPU support is working!")
else:
    print("\n⚠️  No GPUs detected - training will be slow!")
EOF

echo ""
echo "================================================================================"
echo "📋 Installation Summary"
echo "================================================================================"
echo "Installed packages:"
echo "  ✅ TensorFlow 2.15.0 (GPU)"
echo "  ✅ OpenCV 4.8.1.78"
echo "  ✅ NumPy 1.24.3"
echo "  ✅ Pandas 2.0.3"
echo "  ✅ scikit-learn 1.3.0"
echo "  ✅ Matplotlib 3.7.3"
echo "  ✅ Seaborn 0.12.2"
echo "  ✅ tqdm 4.66.1"
echo "  ✅ h5py 3.9.0"
echo "  ✅ Pillow 10.0.1"
echo "  ✅ GPU monitoring tools"
echo ""
echo "================================================================================"
echo "✅ INSTALLATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Upload training scripts to /workspace/"
echo "  2. Verify data: find /workspace/processed/frames/ -name '*.npy' | wc -l"
echo "  3. Start training: python3 /workspace/train_model_RESUME.py"
echo ""
echo "Quick tests:"
echo "  - Check GPU: nvidia-smi"
echo "  - Monitor GPU: gpustat -i 1"
echo "  - Test TensorFlow: python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo ""
echo "================================================================================"
