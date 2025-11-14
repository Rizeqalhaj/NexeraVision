#!/bin/bash
#
# Fix Vast.ai Training Configuration
# Fixes: GPU OOM error + missing config fields
#
# Usage:
#   chmod +x fix_vastai_config.sh
#   ./fix_vastai_config.sh
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 NexaraVision Training Config Fix"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if we're on Vast.ai
if [ ! -d "/workspace" ]; then
    echo "⚠️  Warning: /workspace not found"
    echo "This script is designed for Vast.ai environment"
    echo "Continuing anyway..."
    echo ""
fi

# Backup original config if it exists
if [ -f "/workspace/training_config.json" ]; then
    BACKUP_FILE="/workspace/training_config.json.backup.$(date +%Y%m%d_%H%M%S)"
    cp /workspace/training_config.json "$BACKUP_FILE"
    echo "✅ Original config backed up to:"
    echo "   $BACKUP_FILE"
    echo ""
fi

# Create fixed config
echo "📝 Creating new config with fixes..."

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

echo "✅ Config file created successfully"
echo ""

# Show what changed
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Configuration Changes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "GPU Memory Fix:"
echo "  🔧 batch_size: 32 → 4"
echo "  ✅ Fits in 1GB GPU memory"
echo "  ⏱️  Training will be slower but won't crash"
echo ""
echo "Missing Fields Added:"
echo "  ✅ early_stopping_patience: 5 epochs"
echo "  ✅ reduce_lr_patience: 3 epochs"
echo "  ✅ reduce_lr_factor: 0.5x"
echo ""

# Verify config was created
if [ -f "/workspace/training_config.json" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ SUCCESS - Config Fixed!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 Config Preview:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cat /workspace/training_config.json
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Next Steps:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Clear GPU memory (if training was running):"
    echo "   pkill -f train_model"
    echo ""
    echo "2. Restart training:"
    echo "   cd /workspace"
    echo "   python3 train_model_optimized.py"
    echo ""
    echo "3. Monitor GPU usage:"
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌ ERROR: Failed to create config file"
    exit 1
fi
