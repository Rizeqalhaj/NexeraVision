# NexaraVision ML Service - Installation Guide

## Prerequisites

- Python 3.10+ (Python 3.14 detected ✓)
- NVIDIA GPU with CUDA 11.8+ (optional)
- 4GB+ RAM
- Trained model file

## Installation Steps

### 1. Copy Trained Model

```bash
cd /home/admin/Desktop/NexaraVision/ml_service

# Create models directory
mkdir -p models

# Copy trained model (already available at ../downloaded_models/)
cp ../downloaded_models/ultimate_best_model.h5 models/
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify activation
which python  # Should show: .../ml_service/venv/bin/python
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will install:
# - fastapi, uvicorn (API framework)
# - tensorflow (ML inference)
# - opencv-python (video processing)
# - numpy, pillow (image processing)
# - pytest (testing)
```

### 4. Verify Installation

```bash
# Run verification script
python verify_setup.py

# Expected output:
# ✓ All checks passed! Ready to start service.
```

### 5. Start Service

```bash
# Development mode (with hot reload)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use startup script
./start_service.sh

# Production mode (4 workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. Test Service

```bash
# In another terminal
curl http://localhost:8000/api/health

# Should return:
# {"status": "healthy", "model_loaded": true, ...}
```

## Docker Installation (Alternative)

If you prefer Docker:

```bash
# Copy model
mkdir -p models
cp ../downloaded_models/ultimate_best_model.h5 models/

# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f ml-service

# Test
curl http://localhost:8000/api/health
```

## GPU Support

### Check GPU Availability

```bash
# Check NVIDIA driver
nvidia-smi

# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### GPU Not Available?

The service will work on CPU, but inference will be slower:
- With GPU: ~180ms per inference
- With CPU: ~2-5s per inference

## Troubleshooting

### Issue: Module not found

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Port already in use

```bash
# Use different port
python -m uvicorn app.main:app --reload --port 8001
```

### Issue: Model not found

```bash
# Check model exists
ls -lh ../downloaded_models/ultimate_best_model.h5

# Copy to models directory
cp ../downloaded_models/ultimate_best_model.h5 models/
```

## Next Steps

1. Access API documentation: http://localhost:8000/docs
2. Test file upload endpoint with sample video
3. Integrate with NestJS backend
4. Configure production deployment

## Quick Reference

```bash
# Activate venv
source venv/bin/activate

# Start service
./start_service.sh

# Run tests
pytest tests/ -v

# Check logs
tail -f logs/ml_service.log

# Stop service
Ctrl+C (or docker-compose down for Docker)
```
