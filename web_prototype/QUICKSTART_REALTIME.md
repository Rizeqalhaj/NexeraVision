# 🚀 Nexara Vision Prototype - Quick Start

## ⚡ Test Locally (5 minutes)

```bash
cd /home/admin/Desktop/NexaraVision/web_prototype

# Install dependencies
pip install -r requirements.txt

# Start server
./deploy.sh local

# Open browser
# → http://localhost:8000
```

## 🎬 Using the Interface

### Upload Video
1. **Drag & drop** video file onto upload area
2. Or **click** upload area to browse for file
3. Supported: MP4, AVI, MOV, MKV, FLV, WMV (max 100MB)

### Watch Real-time Analysis
4. Click **"Start Real-time Analysis"** button
5. Watch the magic happen:
   - ✅ Connection indicator shows "● Connected"
   - 📊 Progress bar advances through stages
   - 📈 Live chart updates frame-by-frame
   - 🔴 Violence probability updates in real-time
   - 🟢 Non-violence probability updates live
   - 💬 Status messages show current operation

### View Results
6. Final results card appears with:
   - ⚠️ "VIOLENCE DETECTED" or ✅ "NON-VIOLENT"
   - Confidence percentage
   - Processing time
   - Frames analyzed (always 20)

### Analyze Another
7. Click **"Analyze Another Video"** to reset

## 📊 What You'll See

### Real-time Chart
```
Violence Probability (Red Line)
Non-Violence Probability (Green Line)
    ▲
100%│     ╱─╲
    │    ╱   ╲╱╲
 50%│   ╱        ╲
    │  ╱          ╲
  0%└──────────────────► 
    F1  F5  F10  F15  F20
```

### Live Metrics
```
┌─────────────────────┐ ┌──────────────────────┐
│ Violence Prob       │ │ Non-Violence Prob    │
│                     │ │                      │
│      23.4%          │ │      76.6%           │
│                     │ │                      │
└─────────────────────┘ └──────────────────────┘
```

### Processing Stages
1. ⏳ **Uploading** → "Connecting to server..."
2. 📹 **Frame Extraction** → "Extracting frame 5/20..."
3. 🧠 **Feature Extraction** → "Extracting VGG19 features..."
4. 🔍 **Analysis** → "Analyzing frame 10/20..."
5. ✅ **Complete** → "Analysis complete!"

## 🎯 Example Results

### Non-Violent Video
```
✅ NON-VIOLENT
Confidence: 87.3%

Frames Analyzed: 20
Processing Time: 3.2s
Confidence Score: 87.3%
```

### Violent Video
```
⚠️ VIOLENCE DETECTED
Confidence: 92.1%

Frames Analyzed: 20
Processing Time: 3.5s
Confidence Score: 92.1%
```

## 🌐 Deploy to Production

```bash
# Upload to server
scp -r web_prototype admin@31.57.166.18:/home/admin/

# SSH and deploy
ssh admin@31.57.166.18
cd /home/admin/web_prototype
./deploy.sh production

# Configure domain DNS
# Point vision.nexaratech.io → 31.57.166.18

# Setup SSL
sudo certbot --nginx -d vision.nexaratech.io

# Access
# → https://vision.nexaratech.io
```

## 🔧 Troubleshooting

### WebSocket Connection Failed
- Check firewall allows WebSocket connections
- Verify server is running: `docker ps | grep nexara`
- Check logs: `docker logs nexara-vision-app`

### Chart Not Updating
- Ensure JavaScript is enabled
- Check browser console for errors (F12)
- Refresh page and try again

### Slow Processing
- First run downloads VGG19 weights (~550MB)
- Subsequent runs are faster
- GPU acceleration: Install CUDA for 3-5x speedup

## 📞 Support

**Live URL**: https://vision.nexaratech.io
**Docs**: README.md, FEATURES.md
**Server**: 31.57.166.18

---

**Nexara Vision Prototype v1.0**
Real-time AI-Powered Violence Detection
