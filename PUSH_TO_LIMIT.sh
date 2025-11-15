#!/bin/bash
#
# Push GPU to 95-100% Utilization (batch_size=40)
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 PUSHING GPU 1 TO THE LIMIT - 95-100% UTILIZATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup current config
cp /workspace/training_config.json "/workspace/training_config.json.backup.$(date +%Y%m%d_%H%M%S)"

# Maximum possible batch size
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
    "batch_size": 40,
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

echo "✅ Config updated: batch_size=40"
echo ""
echo "📊 MAXIMUM Performance:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  batch_size=40: 215 steps × ~1500ms = 5.4 min/epoch = 2.7 hours total"
echo ""
echo "  VRAM Usage: ~22-24 GB / 24 GB (95-100% utilized) 🔥"
echo "  GPU Util: 95-100%"
echo "  Training Speed: ~200-250 videos/second"
echo ""
echo "⚠️  WARNING: This pushes GPU to the limit!"
echo "   - If OOM occurs, reduce to batch_size=32 or 36"
echo "   - Monitor nvidia-smi closely"
echo "   - May hit 24GB ceiling and fail"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Ready to start training at MAXIMUM capacity"
echo ""
echo "Run: ./START_TRAINING.sh"
echo ""
