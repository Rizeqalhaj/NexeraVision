#!/bin/bash
# Emergency fix for GPU OOM error
# Run this on Vast.ai to reduce batch size

echo "🔧 Fixing GPU Out of Memory Error..."
echo "=================================="

# Backup original config
cp /workspace/training_config.json /workspace/training_config.json.backup
echo "✅ Original config backed up"

# Reduce batch size to 4 (from 32 or 16)
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
    "batch_size": 4,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "loss": "binary_crossentropy"
  },
  "paths": {
    "models": "/workspace/models/saved_models",
    "logs": "/workspace/models/logs",
    "checkpoints": "/workspace/models/checkpoints"
  }
}
EOF

echo "✅ Batch size reduced: 32 → 4"
echo "✅ This will use less GPU memory"
echo ""
echo "=================================="
echo "🚀 Now restart training:"
echo "  python3 train_model_optimized.py"
echo "=================================="
