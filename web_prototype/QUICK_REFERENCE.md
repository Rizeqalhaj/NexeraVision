# Nexara Vision Enterprise - Quick Reference

## Deployment Commands

### Deploy to Production (Complete)
```bash
cd /home/admin/Desktop/NexaraVision/web_prototype
./deploy_production.sh
```

### Test Locally First
```bash
cd /home/admin/Desktop/NexaraVision/web_prototype
./test_local.sh
```

### Test Production API
```bash
./test_api.py
```

---

## Quick Tests

### Health Check
```bash
curl http://31.57.166.18:8005/api/health
```

### Check Available Models
```bash
curl http://31.57.166.18:8005/api/models
```

### Upload Video (Default Model)
```bash
curl -X POST http://31.57.166.18:8005/upload -F "video=@test.mp4"
```

### Upload Video (Specific Model)
```bash
curl -X POST http://31.57.166.18:8005/upload \
  -F "video=@test.mp4" \
  -F "model_name=ensemble_m1_best"
```

---

## Available Models

1. **best_model** - General purpose
2. **ultimate_best_model** - DEFAULT (Production)
3. **ensemble_m1_best** - High accuracy
4. **ensemble_m2_best** - Balanced
5. **ensemble_m3_best** - Alternative

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check |
| `/api/models` | GET | List available models |
| `/api/info` | GET | System information |
| `/api/stats` | GET | Live statistics |
| `/upload` | POST | Upload video for analysis |
| `/api/docs` | GET | Swagger documentation |
| `/ws/analyze` | WS | Video upload with progress |
| `/ws/live` | WS | Live camera detection |

---

## Production Details

- **Server**: 31.57.166.18
- **Backend Port**: 8005
- **Container**: nexara-vision-detection
- **Models Path**: /root/nexara_models
- **Frontend**: vision.nexaratech.io

---

## Monitoring Commands

### View Logs
```bash
ssh root@31.57.166.18 "docker logs -f nexara-vision-detection"
```

### Check Container
```bash
ssh root@31.57.166.18 "docker ps | grep nexara-vision"
```

### Restart Service
```bash
ssh root@31.57.166.18 "docker restart nexara-vision-detection"
```

### Check Resources
```bash
ssh root@31.57.166.18 "docker stats nexara-vision-detection --no-stream"
```

---

## Expected Response

```json
{
  "violence_probability": 12.34,
  "non_violence_probability": 87.66,
  "classification": "NON-VIOLENT",
  "processing_time": 3.45,
  "model_used": "ultimate_best_model",
  "frames_analyzed": 20,
  "confidence": 87.66,
  "timestamp": "2025-11-06T12:34:56.789Z"
}
```

---

## Troubleshooting

### Models Not Loading
```bash
ssh root@31.57.166.18 "docker exec nexara-vision-detection ls -lh /app/models/"
```

### Container Won't Start
```bash
ssh root@31.57.166.18 "docker logs nexara-vision-detection"
```

### Port Conflict
```bash
ssh root@31.57.166.18 "netstat -tulpn | grep 8005"
```

---

## Documentation

- **Complete Guide**: README_ENTERPRISE.md
- **Deployment Guide**: PRODUCTION_DEPLOYMENT.md
- **Summary**: DEPLOYMENT_SUMMARY.md
- **This File**: QUICK_REFERENCE.md
