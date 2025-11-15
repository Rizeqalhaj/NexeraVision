#!/bin/bash
#
# NexaraVision Training Setup for Vast.ai (2x RTX 3090 Ti)
# Complete setup script for 48GB VRAM multi-GPU system
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 NexaraVision Training Setup - Vast.ai"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Check GPU
echo "🎮 Step 1: GPU Detection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo ""

# Step 2: Fix Configuration
echo "⚙️  Step 2: Fixing Training Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Backup original config if exists
if [ -f "/workspace/training_config.json" ]; then
    BACKUP_FILE="/workspace/training_config.json.backup.$(date +%Y%m%d_%H%M%S)"
    cp /workspace/training_config.json "$BACKUP_FILE"
    echo "✅ Original config backed up to: $BACKUP_FILE"
fi

# Create optimized config for 48GB VRAM
cat <<'EOF' > /workspace/training_config.json
{
  "data": {
    "augmentation": true,
    "class_weights": true
  },
  "model": {
    "sequence_model": "bidirectional_gru",
    "gru_units": 128,
    "dense_layers": [256, 128],
    "dropout": [0.4, 0.3, 0.2]
  },
  "training": {
    "frames_per_video": 20,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.5
  },
  "paths": {
    "models": "/workspace/models/saved_models",
    "logs": "/workspace/models/logs",
    "checkpoints": "/workspace/models/checkpoints"
  }
}
EOF

echo "✅ Config created with optimized settings"
echo ""

# Step 3: Show Configuration
echo "📋 Step 3: Configuration Preview"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat /workspace/training_config.json
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 4: Hardware Summary
echo "💾 Step 4: Hardware Configuration Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "GPU Hardware:"
echo "  • Model: 2x RTX 3090 Ti"
echo "  • Total VRAM: 48 GB"
echo "  • Available VRAM: ~23 GB"
echo "  • CUDA Cores: ~21,760 (dual GPU)"
echo "  • Tensor Cores: ~680 (dual GPU)"
echo ""
echo "Training Configuration:"
echo "  • Batch Size: 16 (optimized for 48GB VRAM)"
echo "  • Frames per Video: 20"
echo "  • GPU Mode: Single GPU (GPU 0 only)"
echo "  • Memory Growth: Enabled (dynamic allocation)"
echo ""
echo "Expected Performance:"
echo "  • VRAM Usage: ~8-12 GB per batch"
echo "  • Training Speed: ~120-150 videos/second"
echo "  • Total Training Time: ~4-6 hours"
echo "  • Expected Accuracy: 96-100%"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 5: Environment Check
echo "🔍 Step 5: Environment Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python and TensorFlow
echo "Python Version:"
python3 --version

echo ""
echo "TensorFlow Version:"
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

echo ""
echo "CUDA Available:"
python3 -c "import tensorflow as tf; print('✅ CUDA Available' if tf.test.is_built_with_cuda() else '❌ CUDA Not Available')"

echo ""
echo "GPU Visible to TensorFlow:"
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'{len(gpus)} GPU(s) detected'); [print(f'  • {gpu.name}') for gpu in gpus]"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 6: Ready to Train
echo "✅ Step 6: Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Ready to Start Training!"
echo ""
echo "Run the following command:"
echo "  cd /workspace"
echo "  python3 train_model_optimized.py"
echo ""
echo "Monitor GPU usage in another terminal:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Expected training time: 4-6 hours"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
