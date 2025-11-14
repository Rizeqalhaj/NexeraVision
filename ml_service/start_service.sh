#!/bin/bash
# NexaraVision ML Service Startup Script

set -e

echo "=========================================="
echo "NexaraVision ML Service Startup"
echo "=========================================="

# Check if model exists
MODEL_PATH="${MODEL_PATH:-/app/models/ultimate_best_model.h5}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found at $MODEL_PATH"
    echo "Please ensure the trained model is available before starting the service."
    exit 1
fi

echo "✓ Model file found: $MODEL_PATH"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check TensorFlow and GPU
echo ""
echo "Checking TensorFlow installation..."
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

echo ""
echo "Checking GPU availability..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs detected: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus]"

# Start service
echo ""
echo "=========================================="
echo "Starting FastAPI service..."
echo "=========================================="

exec uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers "${MAX_WORKERS:-4}" \
    --log-level info
