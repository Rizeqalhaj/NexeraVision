# ML Model Update - Keras 3 Format Migration

**Date**: 2025-11-15
**Status**: ✅ Ready for Deployment

## Problem
- Production ML service failing with "batch_shape parameter not supported" error
- Old model (best_model.h5, 34MB) saved with Keras 2.3 - incompatible with TensorFlow 2.15.0 + tf-keras 2.15.0

## Solution
- Migrated to newer Keras 3 native format model
- Model: `initial_best_model.keras` (118MB, Nov 15 2025)
- Source: `/home/admin/Downloads/models/saved_models/checkpoints/`

## Changes Made

### 1. New Model File
- **File**: `ml_service/models/initial_best_model.keras`
- **Size**: 118MB
- **Format**: Keras 3 native (.keras)
- **Created**: Nov 15, 2025

### 2. Config Updates  
**File**: `ml_service/app/core/config.py`
- Updated model search paths to prioritize .keras format
- Added `initial_best_model.keras` as first priority

### 3. Model Loader Updates
**File**: `ml_service/app/models/violence_detector.py`
- Removed `TF_USE_LEGACY_KERAS` environment variable
- Added format detection (.keras vs .h5)
- Keras 3 models load without `safe_mode=False`
- Legacy .h5 models still supported with `safe_mode=False`

### 4. Dependencies
**File**: `ml_service/requirements.txt`
- Added `tf-keras==2.15.0` for Keras 2 compatibility layer
- Updated `opencv-python==4.8.1.78` (fixed version)

## Deployment Instructions

### Via GitHub CI/CD (Recommended)
```bash
git add ml_service/
git commit -m "fix: migrate to Keras 3 model format for production compatibility"
git push nexera development
```

### Manual Docker Deployment
```bash
cd ml_service
docker build -t nexara-ml-service:latest .
docker run -d --name nexara-ml-service \
  -p 8003:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/initial_best_model.keras \
  --restart unless-stopped \
  nexara-ml-service:latest
```

## Verification

### Check Model Loading
```bash
docker logs nexara-ml-service
# Expected: "Found model at: /app/models/initial_best_model.keras"
# Expected: "Model loaded successfully"
```

### Test Endpoints
```bash
# Health check
curl https://vision.nexaratech.io/upload

# WebSocket test  
wscat -c wss://vision.nexaratech.io/ws/live
```

## Compatibility

### TensorFlow Stack
- TensorFlow: 2.15.0
- tf-keras: 2.15.0 (Keras 2 compatibility)
- Python: 3.10 (Docker)

### Model Formats Supported
- ✅ Keras 3 (.keras) - **Preferred**
- ✅ Legacy Keras 2 (.h5) - Fallback with safe_mode=False

### Hardware
- ✅ CPU-only (production server)
- ✅ GPU (if available)

## Rollback Plan

If issues occur, revert to previous model:
```bash
docker run -d --name nexara-ml-service \
  -p 8003:8000 \
  -e MODEL_PATH=/app/models/ultimate_best_model.h5 \
  nexara-ml-service:latest
```

## Next Steps
1. Push changes to GitHub
2. Monitor CI/CD deployment
3. Verify production service starts successfully
4. Test live detection and file upload endpoints
5. Monitor logs for any runtime errors

---
**Updated**: 2025-11-15 21:45 UTC
**Author**: Claude Code
**Branch**: development
