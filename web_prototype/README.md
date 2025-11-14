# 🎥 Violence Detection Web Application

AI-powered real-time violence detection system with web interface.

## 🚀 Features

- **Real-time Video Analysis**: Upload and analyze videos for violence detection
- **Deep Learning Model**: VGG19 + BiLSTM with attention mechanism
- **Modern Web Interface**: Drag & drop, responsive design
- **REST API**: Easy integration with other systems
- **Docker Support**: Containerized deployment
- **Production Ready**: Optimized for deployment on production servers

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           User Browser                      │
│  (Drag & Drop Video Upload)                 │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         FastAPI Backend                     │
│  • Video upload handling                    │
│  • Frame extraction (20 frames)             │
│  • VGG19 feature extraction                 │
│  • BiLSTM inference                         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      Deep Learning Model                    │
│  VGG19 → BiLSTM → Attention → Dense         │
│  Output: Violence / Non-Violence            │
└─────────────────────────────────────────────┘
```

## 📋 Requirements

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB for models and dependencies
- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: Optional (faster inference with CUDA support)

### Software Requirements
- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- nginx (for reverse proxy)

## 🛠️ Installation

### Option 1: Local Development

1. **Clone and navigate to project**:
   ```bash
   cd /home/admin/Desktop/NexaraVision/web_prototype
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model is in place**:
   ```bash
   ls -lh /home/admin/Downloads/best_model.h5
   # Should show ~34MB file
   ```

4. **Run locally**:
   ```bash
   ./deploy.sh local
   ```

5. **Access application**:
   - Open browser: `http://localhost:8000`

### Option 2: Production Deployment with Docker

1. **Build and deploy**:
   ```bash
   ./deploy.sh production
   ```

2. **Verify deployment**:
   ```bash
   docker ps | grep violence-detection
   docker logs -f violence-detection-app
   ```

3. **Access application**:
   - Public URL: `https://vision.nexaratech.io`
   - Local: `http://localhost:8000`

## 🌐 API Documentation

### Health Check
```bash
GET /api/health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "feature_extractor_loaded": true,
  "timestamp": "2025-11-06T..."
}
```

### Predict Violence
```bash
POST /api/predict

Content-Type: multipart/form-data
Body: file (video file)

Response:
{
  "is_violent": false,
  "violence_probability": 0.15,
  "non_violence_probability": 0.85,
  "confidence": 0.85,
  "classification": "NON-VIOLENT",
  "processing_time_seconds": 2.34,
  "frames_analyzed": 20,
  "filename": "test_video.mp4",
  "file_size_mb": 5.2,
  "timestamp": "2025-11-06T..."
}
```

### API Info
```bash
GET /api/info

Response:
{
  "model_info": {
    "architecture": "VGG19 + BiLSTM with Attention",
    "input_frames": 20,
    "frame_size": [224, 224]
  },
  "limits": {
    "max_file_size_mb": 100,
    "allowed_formats": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
  },
  "version": "1.0.0"
}
```

## 🔧 Configuration

Edit `backend/app.py` to modify:

```python
CONFIG = {
    'model_path': '/home/admin/Downloads/best_model.h5',
    'num_frames': 20,
    'frame_size': (224, 224),
    'max_video_size_mb': 100,
    'allowed_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
}
```

## 📊 Usage Examples

### Using the Web Interface

1. **Open browser**: `https://vision.nexaratech.io`
2. **Upload video**: Drag & drop or click to browse
3. **Analyze**: Click "Analyze Video" button
4. **View results**: See violence detection results with confidence scores

### Using cURL

```bash
# Test with video file
curl -X POST https://vision.nexaratech.io/api/predict \
  -F "file=@test_video.mp4"

# Health check
curl https://vision.nexaratech.io/api/health
```

### Using Python

```python
import requests

# Upload and analyze video
url = "https://vision.nexaratech.io/api/predict"
files = {"file": open("test_video.mp4", "rb")}
response = requests.post(url, files=files)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence'] * 100:.1f}%")
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('file', videoFile);

fetch('https://vision.nexaratech.io/api/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Classification:', data.classification);
  console.log('Confidence:', (data.confidence * 100).toFixed(1) + '%');
});
```

## 🚢 Deployment to Production Server

### Prerequisites on Server (31.57.166.18)

1. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

2. **Install nginx**:
   ```bash
   sudo apt update
   sudo apt install nginx -y
   ```

3. **Upload project to server**:
   ```bash
   # From local machine
   scp -r web_prototype admin@31.57.166.18:/home/admin/

   # Or use git
   git clone <repository> /home/admin/web_prototype
   ```

### Deploy

```bash
cd /home/admin/web_prototype
./deploy.sh production
```

### Configure Domain (vision.nexaratech.io)

The deployment script automatically configures nginx, but you can manually verify:

```bash
# Check nginx config
sudo cat /etc/nginx/sites-available/violence-detection

# Test nginx config
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### SSL Certificate (HTTPS)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d vision.nexaratech.io

# Auto-renewal is configured automatically
```

## 🔍 Monitoring & Logs

### View container logs
```bash
docker logs -f violence-detection-app
```

### View nginx logs
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Container status
```bash
docker ps
docker stats violence-detection-app
```

### Restart application
```bash
docker restart violence-detection-app
```

## 🐛 Troubleshooting

### Model not loading

```bash
# Check model file exists
ls -lh /home/admin/Downloads/best_model.h5

# Check container can access model
docker exec violence-detection-app ls -lh /home/admin/Downloads/best_model.h5
```

### Port already in use

```bash
# Find process using port 8000
sudo lsof -i :8000

# Stop existing container
docker stop violence-detection-app
docker rm violence-detection-app
```

### Out of memory

```bash
# Check memory usage
free -h
docker stats

# Restart with memory limit
docker run -d --memory="4g" ...
```

### Nginx not working

```bash
# Check nginx status
sudo systemctl status nginx

# Check config syntax
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

## 📈 Performance Optimization

### GPU Acceleration

To enable GPU support (if available):

1. **Install NVIDIA Docker runtime**:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Uncomment GPU section in docker-compose.yml**:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

3. **Rebuild and redeploy**:
   ```bash
   ./deploy.sh production
   ```

### Batch Processing

For processing multiple videos, use the API in parallel:

```python
import concurrent.futures
import requests

def process_video(video_path):
    url = "https://vision.nexaratech.io/api/predict"
    files = {"file": open(video_path, "rb")}
    return requests.post(url, files=files).json()

video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_video, video_files))
```

## 🔐 Security Considerations

- **File Size Limits**: Max 100MB per upload (configurable)
- **File Type Validation**: Only video formats allowed
- **Temporary Storage**: Uploaded videos are deleted after processing
- **HTTPS**: Use SSL certificate for production (via certbot)
- **Rate Limiting**: Consider adding nginx rate limiting for public deployments

### Add Rate Limiting (nginx)

```nginx
# In /etc/nginx/sites-available/violence-detection
limit_req_zone $binary_remote_addr zone=videoupload:10m rate=5r/m;

location / {
    limit_req zone=videoupload burst=2;
    # ... rest of config
}
```

## 📦 Project Structure

```
web_prototype/
├── backend/
│   └── app.py              # FastAPI application
├── frontend/
│   └── index.html          # Web interface
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── deploy.sh               # Deployment script
└── README.md              # This file
```

## 🧪 Testing

### Test with sample videos

```bash
# Download test videos
wget https://example.com/test_violence.mp4
wget https://example.com/test_normal.mp4

# Test with curl
curl -X POST https://vision.nexaratech.io/api/predict \
  -F "file=@test_violence.mp4"

curl -X POST https://vision.nexaratech.io/api/predict \
  -F "file=@test_normal.mp4"
```

### Unit tests (future enhancement)

```bash
# Run tests
pytest tests/

# Test coverage
pytest --cov=backend tests/
```

## 🎯 Future Enhancements

- [ ] User authentication and API keys
- [ ] Video history and analytics dashboard
- [ ] Real-time CCTV stream processing
- [ ] Webhook notifications for violence detection
- [ ] Multi-model ensemble prediction
- [ ] Mobile app (iOS/Android)
- [ ] Database for storing results
- [ ] Admin dashboard for monitoring
- [ ] Batch video processing interface
- [ ] Support for multiple languages

## 📞 Support

- **Company**: NexaraTech
- **Website**: https://nexaratech.io
- **Application**: https://vision.nexaratech.io

## 📄 License

Proprietary - NexaraTech © 2025

## 🙏 Acknowledgments

- VGG19 architecture from Oxford Visual Geometry Group
- BiLSTM implementation using TensorFlow/Keras
- FastAPI framework for modern Python APIs
