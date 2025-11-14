# 🚀 Quick Start Guide

Get your violence detection system running in 5 minutes!

## ⚡ Fastest Way (Local Testing)

```bash
# 1. Navigate to project
cd /home/admin/Desktop/NexaraVision/web_prototype

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
./deploy.sh local

# 4. Open browser
# → http://localhost:8000
```

**That's it!** Upload a video and see results.

---

## 🌐 Production Deployment

### On Your Server (31.57.166.18)

```bash
# 1. Upload project to server
scp -r web_prototype admin@31.57.166.18:/home/admin/

# 2. SSH to server
ssh admin@31.57.166.18

# 3. Deploy with Docker
cd /home/admin/web_prototype
./deploy.sh production

# 4. Configure domain DNS
# Point vision.nexaratech.io → 31.57.166.18

# 5. Setup SSL (optional but recommended)
sudo certbot --nginx -d vision.nexaratech.io

# 6. Access your app
# → https://vision.nexaratech.io
```

---

## 🧪 Test the API

```bash
# Health check
curl https://vision.nexaratech.io/api/health

# Analyze video
curl -X POST https://vision.nexaratech.io/api/predict \
  -F "file=@your_video.mp4"
```

---

## 🎯 What You Get

**Public URL**: `https://vision.nexaratech.io`

**Features**:
- ✅ Drag & drop video upload
- ✅ Real-time violence detection
- ✅ Confidence scores
- ✅ Processing time metrics
- ✅ REST API for integrations

**Supported Videos**:
- Formats: MP4, AVI, MOV, MKV, FLV, WMV
- Max size: 100MB
- Any resolution (auto-resized)

---

## 🔧 Common Commands

```bash
# View logs
docker logs -f violence-detection-app

# Restart
docker restart violence-detection-app

# Stop
docker stop violence-detection-app

# Rebuild after changes
docker build -t violence-detection .
docker stop violence-detection-app
docker rm violence-detection-app
./deploy.sh production
```

---

## 📞 Need Help?

Check **README.md** for detailed documentation.

**Server**: 31.57.166.18
**Domain**: vision.nexaratech.io
**Model**: /home/admin/Downloads/best_model.h5
