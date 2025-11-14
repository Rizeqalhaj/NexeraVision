#!/bin/bash
#
# Quick Fix: Use GPU 1 (GPU 0 is occupied)
# Just run this and training starts!
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎮 Using GPU 1 for Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check GPU 1 status
echo "📊 GPU 1 Status:"
nvidia-smi --id=1 --query-gpu=name,memory.total,memory.used,memory.free --format=table
echo ""

GPU1_MEMORY_USED=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)

if [ "$GPU1_MEMORY_USED" -lt 1000 ]; then
    echo "✅ GPU 1 is available! ($GPU1_MEMORY_USED MB used)"
    echo ""
    echo "🚀 Starting training on GPU 1 with batch_size=1..."
    echo ""
    
    # Set batch_size=1 (ultra-conservative)
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
    "batch_size": 1,
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
    
    echo "✅ Config set: batch_size=1"
    echo ""
    
    # Export GPU 1
    export CUDA_VISIBLE_DEVICES=1
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_GPU_ALLOCATOR=cuda_malloc_async
    
    # Start training
    cd /workspace
    python3 train_model_optimized.py
    
else
    echo "⚠️  WARNING: GPU 1 has $GPU1_MEMORY_USED MB used"
    echo ""
    echo "Both GPUs occupied? Run:"
    echo "  ./CLEAR_GPU_MEMORY.sh"
    echo ""
    exit 1
fi
