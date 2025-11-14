# NexaraVision ML Service - Quick Start Guide

Get the violence detection ML service running in 5 minutes.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (optional, will use CPU if unavailable)
- Trained model file: `ultimate_best_model.h5`

## Step 1: Copy Trained Model

```bash
cd /home/admin/Desktop/NexaraVision/ml_service

# Create models directory
mkdir -p models

# Copy trained model
cp ../downloaded_models/ultimate_best_model.h5 models/
```

## Step 2: Setup Environment

### Option A: Using venv (Recommended for development)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using Docker (Recommended for production)

```bash
# Build Docker image
docker-compose build

# Start service
docker-compose up -d

# View logs
docker-compose logs -f ml-service
```

## Step 3: Start Service

### Development Mode (Hot Reload)

```bash
# Activate venv if not already
source venv/bin/activate

# Start with auto-reload
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Using startup script
./start_service.sh

# Or manually with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Step 4: Test Service

### Test Health Endpoint

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {...}
}
```

### Test File Upload Detection

```bash
# Upload a video file
curl -X POST "http://localhost:8000/api/detect" \
  -F "video=@/path/to/test_video.mp4"
```

Expected response:
```json
{
  "violence_probability": 0.92,
  "confidence": "High",
  "prediction": "violence",
  "per_class_scores": {
    "non_violence": 0.08,
    "violence": 0.92
  },
  "video_metadata": {
    "filename": "test_video.mp4",
    "duration_seconds": 30.5,
    "fps": 30.0,
    "resolution": "1920x1080",
    "total_frames": 915
  }
}
```

### Test Live Detection

```bash
# Send 20 base64 encoded frames
curl -X POST "http://localhost:8000/api/detect_live" \
  -H "Content-Type: application/json" \
  -d '{
    "frames": ["base64_frame_1", ..., "base64_frame_20"]
  }'
```

## Step 5: Explore API Documentation

Open your browser and go to:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Common Issues

### Issue: Model file not found

```bash
# Check if model exists
ls -lh models/ultimate_best_model.h5

# If missing, copy from downloaded_models
cp ../downloaded_models/ultimate_best_model.h5 models/
```

### Issue: GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check TensorFlow GPU support
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issue: CUDA out of memory

Edit `.env` and reduce GPU memory allocation:

```bash
GPU_MEMORY_FRACTION=0.6  # Use 60% instead of 80%
BATCH_SIZE=16            # Reduce batch size
```

### Issue: Port 8000 already in use

```bash
# Use different port
export PORT=8001
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Next Steps

1. **Integration with NestJS Backend**: Configure NestJS to call ML service endpoints
2. **Performance Tuning**: Optimize batch size and GPU memory for your hardware
3. **Monitoring**: Setup Prometheus metrics and Grafana dashboards
4. **Production Deployment**: Deploy with Kubernetes or Docker Swarm

## Performance Tips

- **GPU Memory**: Adjust `GPU_MEMORY_FRACTION` based on available VRAM
- **Batch Size**: Larger batches = better GPU utilization (up to memory limit)
- **Workers**: Set `MAX_WORKERS` to number of CPU cores for best concurrency
- **TensorFlow XLA**: Enable with `TF_XLA_FLAGS=--tf_xla_auto_jit=2` for 10-20% speedup

## Support

For issues or questions:
- Check logs: `docker-compose logs ml-service`
- Review API docs: http://localhost:8000/docs
- Test endpoints: Use Swagger UI for interactive testing
