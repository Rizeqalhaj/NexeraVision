#!/bin/bash
#
# Emergency Fix: Force Single GPU with Reduced Batch Size
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Single GPU Fix - Reducing Memory Usage"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup current config
if [ -f "/workspace/training_config.json" ]; then
    BACKUP="/workspace/training_config.json.backup.$(date +%Y%m%d_%H%M%S)"
    cp /workspace/training_config.json "$BACKUP"
    echo "✅ Backed up config to: $BACKUP"
fi

# Create conservative config for single GPU
cat <<'CONFIGEOF' > /workspace/training_config.json
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
    "batch_size": 8,
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
CONFIGEOF

echo ""
echo "✅ Updated training config"
echo ""
echo "📊 Changes Made:"
echo "   batch_size: 16 → 8 (more conservative)"
echo "   Single GPU mode: GPU 0 only"
echo ""
echo "Expected Performance:"
echo "   VRAM Usage: ~4-6 GB"
echo "   Training Speed: ~60-80 videos/sec"
echo "   Training Time: ~6-8 hours"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Ready to train!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Run:"
echo "  ./START_TRAINING.sh"
echo ""
echo "Or manually:"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo "  python3 train_model_optimized.py"
echo ""
