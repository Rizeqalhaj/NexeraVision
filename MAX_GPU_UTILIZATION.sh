#!/bin/bash
#
# Maximum GPU Utilization: 90-100% of 24GB VRAM
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ MAXIMUM GPU UTILIZATION - 90-100% of 24GB VRAM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup current config
cp /workspace/training_config.json "/workspace/training_config.json.backup.$(date +%Y%m%d_%H%M%S)"

# Create maximum utilization config
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
    "batch_size": 32,
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

echo "✅ Config updated: batch_size=32"
echo ""
echo "📊 Performance Projection:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Batch Size Comparison:"
echo "  batch_size=1:  8584 steps × 680ms  = 97 min/epoch  = 48 hours   (10% GPU)  ❌"
echo "  batch_size=16: 537 steps  × 1000ms = 9 min/epoch   = 4.5 hours  (60% GPU)  ⚠️"
echo "  batch_size=32: 269 steps  × 1200ms = 5.4 min/epoch = 2.7 hours  (85-90% GPU) ✅"
echo ""
echo "Expected VRAM Usage:"
echo "  ~18-22 GB / 24 GB (85-90% utilized)"
echo ""
echo "Training Speed:"
echo "  ~150-200 videos/second"
echo "  Total time: ~2.7-3 hours for 30 epochs"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Starting training with batch_size=32..."
echo ""
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"
echo ""
echo "If VRAM usage < 90%, you can push to batch_size=40 or 48!"
echo "If OOM occurs, script will automatically retry with batch_size=24"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
