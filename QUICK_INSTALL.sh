#!/bin/bash
# Quick one-liner installation for NexaraVision training

echo "🚀 NexaraVision Quick Install"

pip3 install --upgrade pip && \
pip3 install tensorflow==2.15.0 opencv-python==4.8.1.78 numpy==1.24.3 pandas==2.0.3 \
scikit-learn==1.3.0 matplotlib==3.7.3 seaborn==0.12.2 tqdm==4.66.1 h5py==3.9.0 \
Pillow==10.0.1 psutil==5.9.5 gpustat==1.1.1 && \
python3 -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__} - GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"

echo ""
echo "✅ Installation complete!"
echo "Run: python3 /workspace/train_model_RESUME.py"
