#!/bin/bash
#
# NexaraVision Training Starter Script
# Forces GPU 1 (GPU 0 is occupied)
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 NexaraVision Training - Using GPU 1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Force TensorFlow to use ONLY GPU 1 (GPU 0 is already occupied)
export CUDA_VISIBLE_DEVICES=1

# Prevent TensorFlow from allocating all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Better memory management
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "🎮 GPU Configuration:"
echo "   CUDA_VISIBLE_DEVICES=1 (using GPU 1, GPU 0 is occupied)"
echo "   TF_FORCE_GPU_ALLOW_GROWTH=true"
echo "   TF_GPU_ALLOCATOR=cuda_malloc_async"
echo ""

# Show which GPU will be used
echo "📊 GPU Status Before Training:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --id=1 --query-gpu=name,memory.total,memory.used,memory.free --format=table
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Clear any existing TensorFlow processes on GPU 1
echo "🧹 Cleaning up any existing processes on GPU 1..."
pkill -f train_model 2>/dev/null || true
sleep 2

# Start training
echo "🚀 Starting Training on GPU 1..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /workspace
python3 train_model_optimized.py

# Show final status
EXIT_CODE=$?
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully on GPU 1!"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check GPU 1 memory: nvidia-smi --id=1"
    echo "  2. Reduce batch size in /workspace/training_config.json"
    echo "  3. Check logs: /workspace/models/logs/training/"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT_CODE
