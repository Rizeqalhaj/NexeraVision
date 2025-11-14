#!/bin/bash
#
# Clear GPU Memory - Kill all processes using GPU
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧹 Clearing GPU Memory"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📊 Current GPU Status:"
nvidia-smi
echo ""

echo "🔍 Finding processes using GPU..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

if [ -z "$GPU_PIDS" ]; then
    echo "✅ No processes found using GPU"
else
    echo "⚠️  Found processes using GPU:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo ""
    
    echo "🛑 Killing GPU processes..."
    for PID in $GPU_PIDS; do
        echo "  Killing PID $PID..."
        kill -9 $PID 2>/dev/null || true
    done
fi

echo ""
echo "🧹 Killing any Python/TensorFlow processes..."
pkill -9 -f python 2>/dev/null || true
pkill -9 -f tensorflow 2>/dev/null || true
pkill -9 -f train_model 2>/dev/null || true

echo ""
echo "⏳ Waiting for GPU to clear (5 seconds)..."
sleep 5

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 GPU Status After Cleanup:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi
echo ""

# Check if GPUs are clear
GPU_MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0)

if [ "$GPU_MEMORY_USED" -lt 100 ]; then
    echo "✅ GPU 0 is clear! ($GPU_MEMORY_USED MB used)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Ready to train!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Run:"
    echo "  ./START_TRAINING.sh"
else
    echo "⚠️  WARNING: GPU 0 still has $GPU_MEMORY_USED MB used"
    echo ""
    echo "If this persists, try:"
    echo "  sudo nvidia-smi --gpu-reset -i 0"
    echo "  (requires root access)"
fi
echo ""
