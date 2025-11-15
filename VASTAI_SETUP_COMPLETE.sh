#!/bin/bash
# NexaraVision Vast.ai Complete Setup Script
# Run this script on the Vast.ai instance

set -e  # Exit on error

echo "========================================"
echo "NexaraVision Vast.ai Setup"
echo "========================================"
echo ""

# Update system
echo "📦 Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# Install essential packages
echo "🔧 Installing essential packages..."
apt-get install -y \
    python3.10 python3-pip \
    git wget curl unzip zip \
    ffmpeg libsm6 libxext6 \
    aria2 screen htop nvtop \
    tree tmux vim

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip3 install --upgrade pip --quiet

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --quiet \
    tensorflow==2.13.0 \
    opencv-python==4.8.1.78 \
    opencv-contrib-python==4.8.1.78 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    h5py==3.9.0 \
    Pillow==10.0.0 \
    tqdm==4.66.1 \
    kaggle==1.5.16 \
    yt-dlp==2023.10.13

# Create directory structure
echo "📁 Creating workspace directory structure..."
mkdir -p /workspace/datasets/{tier1,tier2,tier3,processed,cache}
mkdir -p /workspace/{models,logs,checkpoints,scripts}
mkdir -p /workspace/temp

# Configure Kaggle API
echo "🔑 Configuring Kaggle API..."
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"issadalu","key":"5aabafacbfdefea1bf4f2171d98cc52b"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Test GPU
echo ""
echo "========================================"
echo "🎮 GPU Information"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader

echo ""
python3 -c "import tensorflow as tf; print(f'✅ TensorFlow GPUs Available: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# Test OpenCV
python3 -c "import cv2; print(f'✅ OpenCV Version: {cv2.__version__}')"

# Display directory structure
echo ""
echo "========================================"
echo "📂 Workspace Structure"
echo "========================================"
tree -L 2 /workspace/

echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Transfer dataset download scripts"
echo "2. Start downloading datasets"
echo ""
