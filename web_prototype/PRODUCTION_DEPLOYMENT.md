# Nexara Vision Enterprise - Production Deployment Guide

## Quick Deployment

### Automated Deployment (Recommended)
```bash
cd /home/admin/Desktop/NexaraVision/web_prototype
chmod +x deploy_production.sh
./deploy_production.sh
```

This script will:
1. Build production Docker image
2. Copy image to production server
3. Copy all 5 models to server
4. Stop old container
5. Start new container with proper volume mounts
6. Verify deployment

---

## Manual Deployment

### Step 1: Build Docker Image
```bash
cd /home/admin/Desktop/NexaraVision/web_prototype
docker build -f Dockerfile.production -t nexara-vision:production-v2 .
```

### Step 2: Copy Models to Server
```bash
# Create models directory on server
ssh root@31.57.166.18 "mkdir -p /root/nexara_models"

# Copy all model files
scp /home/admin/Desktop/NexaraVision/downloaded_models/*.h5 root@31.57.166.18:/root/nexara_models/
```

### Step 3: Transfer Docker Image
```bash
# Save image
docker save nexara-vision:production-v2 | gzip > /tmp/nexara_vision.tar.gz

# Copy to server
scp /tmp/nexara_vision.tar.gz root@31.57.166.18:/tmp/

# Load on server
ssh root@31.57.166.18 "docker load < /tmp/nexara_vision.tar.gz"
```

### Step 4: Deploy Container
```bash
ssh root@31.57.166.18 << 'EOF'
# Stop old container
docker stop nexara-vision-detection 2>/dev/null || true
docker rm nexara-vision-detection 2>/dev/null || true

# Start new container
docker run -d \
  --name nexara-vision-detection \
  --restart unless-stopped \
  -p 8005:8000 \
  -v /root/nexara_models:/app/models:ro \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  nexara-vision:production-v2

# Check status
docker ps | grep nexara-vision-detection
docker logs nexara-vision-detection --tail 50
EOF
```

---

## Verification

### 1. Check Container Status
```bash
ssh root@31.57.166.18 "docker ps | grep nexara-vision-detection"
```

### 2. Check Logs
```bash
ssh root@31.57.166.18 "docker logs -f nexara-vision-detection"
```

### 3. Test Endpoints

#### Health Check
```bash
curl http://31.57.166.18:8005/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "system": "Nexara Vision Enterprise",
  "models_loaded": 1,
  "available_models": [
    "best_model",
    "ultimate_best_model",
    "ensemble_m1_best",
    "ensemble_m2_best",
    "ensemble_m3_best"
  ],
  "feature_extractor_loaded": true,
  "active_sessions": 0,
  "timestamp": "2025-11-06T...",
  "version": "2.0.0"
}
```

#### Models Info
```bash
curl http://31.57.166.18:8005/api/models
```

#### API Info
```bash
curl http://31.57.166.18:8005/api/info
```

### 4. Test Video Upload

#### Using curl (REST API)
```bash
curl -X POST http://31.57.166.18:8005/upload \
  -F "video=@test_video.mp4" \
  -F "model_name=ultimate_best_model"
```

Expected response:
```json
{
  "violence_probability": 12.34,
  "non_violence_probability": 87.66,
  "classification": "NON-VIOLENT",
  "processing_time": 3.45,
  "model_used": "ultimate_best_model",
  "frames_analyzed": 20,
  "confidence": 87.66,
  "timestamp": "2025-11-06T..."
}
```

---

## Model Information

### Available Models (5 total)

1. **best_model** - Primary trained model
   - File: `best_model.h5` (29 MB)
   - Recommended for: General use

2. **ultimate_best_model** - Enhanced model (DEFAULT)
   - File: `ultimate_best_model.h5` (29 MB)
   - Recommended for: Production use

3. **ensemble_m1_best** - Ensemble model 1
   - File: `ensemble_m1_best.h5` (29 MB)
   - Recommended for: High accuracy scenarios

4. **ensemble_m2_best** - Ensemble model 2
   - File: `ensemble_m2_best.h5` (29 MB)
   - Recommended for: Balanced performance

5. **ensemble_m3_best** - Ensemble model 3
   - File: `ensemble_m3_best.h5` (29 MB)
   - Recommended for: Alternative validation

### Model Selection in Requests

#### REST API
```bash
# Default model (ultimate_best_model)
curl -X POST http://31.57.166.18:8005/upload -F "video=@test.mp4"

# Specific model
curl -X POST http://31.57.166.18:8005/upload \
  -F "video=@test.mp4" \
  -F "model_name=ensemble_m1_best"
```

#### WebSocket
```javascript
ws.send(JSON.stringify({
    filename: 'test.mp4',
    file_data: base64Data,
    model: 'ensemble_m1_best'  // Optional, defaults to ultimate_best_model
}));
```

---

## Architecture Details

### Container Configuration
- **Image**: nexara-vision:production-v2
- **Container**: nexara-vision-detection
- **Port**: 8005 (host) → 8000 (container)
- **Models**: /root/nexara_models (host) → /app/models (container, read-only)
- **Restart Policy**: unless-stopped
- **Health Check**: Every 30s via /api/health

### System Requirements
- Docker installed
- 4GB+ RAM recommended
- 10GB disk space for models
- CPU: 4+ cores recommended
- GPU: Optional (uses CPU by default)

### API Endpoints

#### REST Endpoints
- `GET /` - Landing page
- `GET /api/health` - Health check
- `GET /api/models` - Models information
- `GET /api/info` - API information
- `GET /api/stats` - System statistics
- `GET /api/docs` - Swagger documentation
- `GET /api/redoc` - ReDoc documentation
- `POST /upload` - Video upload and analysis

#### WebSocket Endpoints
- `WS /ws/analyze` - Video upload with progress
- `WS /ws/live` - Live camera detection

---

## Troubleshooting

### Container won't start
```bash
# Check logs
ssh root@31.57.166.18 "docker logs nexara-vision-detection"

# Check if models exist
ssh root@31.57.166.18 "ls -lh /root/nexara_models/"

# Verify models are readable
ssh root@31.57.166.18 "docker exec nexara-vision-detection ls -lh /app/models/"
```

### Models not loading
```bash
# Check if models directory is mounted correctly
ssh root@31.57.166.18 "docker inspect nexara-vision-detection | grep -A 5 Mounts"

# Verify model files
ssh root@31.57.166.18 "docker exec nexara-vision-detection ls -lh /app/models/"

# Check application logs
ssh root@31.57.166.18 "docker logs nexara-vision-detection 2>&1 | grep -i model"
```

### Port already in use
```bash
# Check what's using port 8005
ssh root@31.57.166.18 "netstat -tulpn | grep 8005"

# Stop conflicting service
ssh root@31.57.166.18 "docker stop <container_name>"
```

### High memory usage
```bash
# Check container stats
ssh root@31.57.166.18 "docker stats nexara-vision-detection --no-stream"

# Restart container to clear cache
ssh root@31.57.166.18 "docker restart nexara-vision-detection"
```

---

## Maintenance

### View Logs
```bash
# Real-time logs
ssh root@31.57.166.18 "docker logs -f nexara-vision-detection"

# Last 100 lines
ssh root@31.57.166.18 "docker logs nexara-vision-detection --tail 100"
```

### Restart Container
```bash
ssh root@31.57.166.18 "docker restart nexara-vision-detection"
```

### Update Deployment
```bash
# Re-run deployment script
cd /home/admin/Desktop/NexaraVision/web_prototype
./deploy_production.sh
```

### Backup Models
```bash
ssh root@31.57.166.18 "tar -czf /tmp/nexara_models_backup.tar.gz /root/nexara_models/"
scp root@31.57.166.18:/tmp/nexara_models_backup.tar.gz ./models_backup.tar.gz
```

---

## Integration with Frontend

The frontend at **vision.nexaratech.io** should connect to:

```
Backend API: http://31.57.166.18:8005
```

### Frontend Configuration
Update your Next.js environment variables:
```env
NEXT_PUBLIC_API_URL=http://31.57.166.18:8005
NEXT_PUBLIC_WS_URL=ws://31.57.166.18:8005
```

### Nginx Proxy (Optional)
If using nginx proxy, add:
```nginx
location /api/ {
    proxy_pass http://31.57.166.18:8005/api/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
}

location /upload {
    proxy_pass http://31.57.166.18:8005/upload;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    client_max_body_size 100M;
}

location /ws/ {
    proxy_pass http://31.57.166.18:8005/ws/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "Upgrade";
    proxy_set_header Host $host;
}
```

---

## Performance Tuning

### For Production Load
```bash
# Run with multiple workers (if you have GPU)
ssh root@31.57.166.18 << 'EOF'
docker run -d \
  --name nexara-vision-detection \
  --restart unless-stopped \
  -p 8005:8000 \
  -v /root/nexara_models:/app/models:ro \
  -e WORKERS=2 \
  nexara-vision:production-v2 \
  python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 2
EOF
```

### Memory Limits
```bash
# Set memory limit
docker run -d \
  --name nexara-vision-detection \
  --restart unless-stopped \
  --memory="4g" \
  --memory-swap="4g" \
  -p 8005:8000 \
  -v /root/nexara_models:/app/models:ro \
  nexara-vision:production-v2
```

---

## Security Considerations

1. **API Access**: Consider adding authentication for production
2. **HTTPS**: Use SSL/TLS in production
3. **Rate Limiting**: Implement rate limiting for public API
4. **CORS**: Configure CORS for specific origins only
5. **Model Files**: Keep models secure and version-controlled

---

## Support

For issues or questions:
- Check logs: `docker logs nexara-vision-detection`
- Test health: `curl http://31.57.166.18:8005/api/health`
- View docs: http://31.57.166.18:8005/api/docs
