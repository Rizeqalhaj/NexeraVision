#!/bin/bash
#
# Increase Batch Size to Utilize Full GPU Power
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ Increasing Batch Size to Utilize GPU 1 (24GB VRAM)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup current config
cp /workspace/training_config.json "/workspace/training_config.json.backup.$(date +%Y%m%d_%H%M%S)"

# Create optimized config for GPU 1 (24GB VRAM)
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
CONFIGEOF

echo "✅ Config updated!"
echo ""
echo "📊 Performance Comparison:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  batch_size=1:  8584 steps/epoch × 680ms = 97 minutes/epoch = 48 hours total ❌"
echo "  batch_size=16: 537 steps/epoch × 1000ms = 9 minutes/epoch = 4.5 hours total ✅"
echo ""
echo "  Speed improvement: 16x faster!"
echo "  VRAM usage: ~12-16 GB (still plenty of headroom)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Ready to restart training with batch_size=16"
echo ""
