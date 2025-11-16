# NexaraVision Model Integration Guide

## Summary

Successfully integrated the trained violence detection model (`initial_best_model.keras` - 118MB) into the NexaraVision ML service. The model now powers:

- **File Upload Detection**: Upload CCTV footage for violence analysis
- **Live Camera Monitoring**: Real-time webcam violence detection
- **Multi-Camera Grid**: Monitor multiple feeds simultaneously

## Architecture

```
Frontend (Next.js)          Backend (NestJS)           ML Service (FastAPI)
   Port 8001          →        Port 8002          →        Port 8003
   ┌─────────┐               ┌─────────────┐            ┌──────────────┐
   │  /live  │  HTTP/WS      │  Proxy API  │   HTTP     │ TensorFlow   │
   │  Upload │ ──────────►   │  + WebSocket│ ────────►  │ Model        │
   │  Camera │               │  Gateway    │            │ Inference    │
   └─────────┘               └─────────────┘            └──────────────┘
```

## Model Specifications

- **File**: `ml_service/models/initial_best_model.keras`
- **Size**: 118MB
- **Format**: Keras 3 native format
- **Input Shape**: `(batch, 20, 224, 224, 3)` - 20 RGB frames at 224x224
- **Output Shape**: `(batch, 2)` - [non_violence_prob, violence_prob]
- **Custom Layer**: AttentionLayer for frame importance weighting

## Quick Start

### 1. Test Model Loading
```bash
cd /home/admin/Desktop/NexaraVision
python3 TEST_MODEL_INTEGRATION.py
```

### 2. Start All Services
```bash
./START_NEXARA_SERVICES.sh
```

This will:
- Start ML Service on port 8003 (loads model, ~30-60s)
- Start Backend on port 8002
- Start Frontend on port 8001
- Monitor all services

### 3. Access Application
- **Live Detection**: http://localhost:8001/live
- **API Docs**: http://localhost:8003/docs
- **Backend Health**: http://localhost:8002/api

## Features

### File Upload Detection
1. Navigate to http://localhost:8001/live
2. Select "File Upload" tab
3. Drag & drop or select video file (MP4, AVI, MOV, MKV up to 500MB)
4. Wait for analysis (extracts 20 frames, runs inference)
5. View results with violence probability, confidence, and per-class scores

### Live Camera Monitoring
1. Navigate to http://localhost:8001/live
2. Select "Live Camera" tab
3. Click "Start Live Detection"
4. Grant camera permissions
5. Real-time violence probability overlay
6. Automatic alerts when probability > 85%

### Multi-Camera Grid
1. Navigate to http://localhost:8001/live
2. Select "Multi-Camera Grid" tab
3. Monitor multiple CCTV feeds with screen recording segmentation

## API Endpoints

### ML Service (Port 8003)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | POST | Upload video file for detection |
| `/api/detect_live` | POST | Send 20 frames for live detection |
| `/api/detect_live_batch` | POST | Batch process up to 32 requests |
| `/api/ws/live` | WebSocket | Real-time streaming detection |
| `/api/health` | GET | Service health check |
| `/api/info` | GET | Model and device information |

### Detection Response Format
```json
{
  "violence_probability": 0.92,
  "confidence": "High",
  "prediction": "violence",
  "per_class_scores": {
    "non_violence": 0.08,
    "violence": 0.92
  }
}
```

## Configuration Files Updated

1. **ML Service** (`ml_service/app/core/config.py`)
   - Port: 8003
   - Model path discovery prioritizes `.keras` format

2. **Backend** (`web_app_backend/.env`)
   - Port: 8002
   - ML_SERVICE_URL: http://localhost:8003/api

3. **Frontend** (`web_app_nextjs/.env.local`)
   - API_URL: http://localhost:8002/api
   - WS_URL: ws://localhost:8002/ws/live

## Troubleshooting

### Model Loading Issues
```bash
# Check logs
tail -f /home/admin/Desktop/NexaraVision/logs/ml_service.log

# Verify model file
ls -lh ml_service/models/initial_best_model.keras

# Test model directly
python3 TEST_MODEL_INTEGRATION.py
```

### Service Connection Issues
```bash
# Check if services are running
lsof -i :8001  # Frontend
lsof -i :8002  # Backend
lsof -i :8003  # ML Service

# Kill stuck processes
fuser -k 8003/tcp
```

### TensorFlow/GPU Issues
```bash
# Check TensorFlow installation
pip show tensorflow

# Verify GPU availability (if applicable)
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Performance Expectations

- **Model Loading**: 30-60 seconds (CPU), 10-20 seconds (GPU)
- **Single Video Analysis**: 2-5 seconds
- **Live Detection Latency**: 50-200ms per batch of 20 frames
- **Memory Usage**: ~2-4GB RAM (model + TensorFlow)

## Next Steps

1. **GPU Acceleration**: Install CUDA/cuDNN for faster inference
2. **Model Optimization**: Consider TensorFlow Lite or TensorRT for production
3. **Horizontal Scaling**: Deploy multiple ML service instances
4. **Monitoring**: Add Prometheus metrics and Grafana dashboards
5. **Security**: Add API authentication and rate limiting

## Files Created/Modified

**Created:**
- `START_NEXARA_SERVICES.sh` - Service orchestration script
- `TEST_MODEL_INTEGRATION.py` - Model validation test
- `MODEL_INTEGRATION_GUIDE.md` - This guide

**Modified:**
- `ml_service/app/core/config.py` - Port 8000 → 8003
- `web_app_backend/.env` - ML_SERVICE_URL and PORT updates
- `web_app_backend/.env.example` - Updated defaults
- `web_app_backend/src/config/configuration.ts` - Updated defaults
- `web_app_nextjs/.env.local` - API URL configuration

**Copied:**
- Model file from `/home/admin/Downloads/initial_best_model.keras` to `ml_service/models/`

---

**Note**: The model expects exactly 20 frames per prediction. For live camera monitoring, frames are captured at 30fps with 50% overlap between batches for smooth detection.
