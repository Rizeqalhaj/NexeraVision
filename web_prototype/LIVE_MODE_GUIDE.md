# 🔴 Nexara Vision - Live Detection Mode Guide

## 🎯 What's New

Your Nexara Vision Prototype now has **TWO MODES**:

### 📁 Upload Mode (Original)
- Upload pre-recorded video files
- Analyze 20 evenly-spaced frames
- Get comprehensive results

### 🔴 Live Mode (NEW!)
- **Real-time webcam detection**
- **Continuous monitoring**
- **Instant alerts** when violence detected
- **Live confidence charts**

---

## 🚀 How to Use Live Mode

### Step 1: Choose Mode
At the top of the page, click **"Live Detection"** button

### Step 2: Grant Camera Access
- Browser will ask for camera permission
- Click "Allow" to enable webcam

### Step 3: Start Monitoring
- Click **"🔴 Start Live Monitoring"**
- Your webcam activates
- AI starts analyzing in real-time

### Step 4: Watch Real-time Analysis
- **Live video feed** shows what camera sees
- **Green "MONITORING"** overlay shows active status
- **Metrics update** every 0.5 seconds:
  - Violence probability
  - Non-violence probability
- **Chart updates** with rolling confidence scores

### Step 5: Violence Alerts
If violence detected (>70% confidence):
- **Red "VIOLENCE DETECTED"** overlay appears
- **Alert banner** shows at top of analysis panel
- **Animated shake** effect for attention
- **Timestamp** of detection

### Step 6: Stop Monitoring
- Click **"⏹️ Stop Monitoring"** button
- Webcam turns off
- Analysis stops

---

## 🎬 What Happens Behind the Scenes

### Frame Capture (Every 0.5 seconds)
1. Webcam captures current frame
2. Frame converted to base64 image
3. Sent to backend via WebSocket

### AI Processing
1. Backend maintains buffer of last 20 frames
2. When buffer full:
   - Extracts VGG19 features from all 20 frames
   - Runs BiLSTM + Attention analysis
   - Generates violence/non-violence probability
3. Sends result back to frontend

### Real-time Updates
1. Frontend receives confidence scores
2. Updates metrics cards
3. Adds point to live chart
4. Checks if violence detected
5. Shows alert if needed

### Buffer System
- **Rolling 20-frame buffer** (like sliding window)
- Analyzes last 4-5 seconds of video
- Continuously updates as new frames arrive
- Detects patterns over time (not just single frame)

---

## 📊 Understanding the Interface

### Mode Selector
```
┌──────────────────────────────┐
│ 📁 Upload Video              │  ← Click to upload files
│ Analyze pre-recorded videos  │
├──────────────────────────────┤
│ 🔴 Live Detection            │  ← Click for webcam
│ Real-time camera monitoring  │
└──────────────────────────────┘
```

### Live Mode Panel
```
┌─────────────────────────────────┐
│ 📹 LIVE VIDEO FEED              │
│ ┌─────────────────────────────┐ │
│ │                             │ │
│ │    [Your webcam video]      │ │
│ │                             │ │
│ │  🔴 MONITORING              │ │ ← Status overlay
│ └─────────────────────────────┘ │
│                                 │
│ [🔴 Start Live Monitoring]      │
│ [⏹️ Stop Monitoring]             │
└─────────────────────────────────┘
```

### Real-time Analysis Panel
```
┌─────────────────────────────────┐
│ 📊 Real-time Analysis           │
├─────────────────────────────────┤
│ ⚠️ VIOLENCE DETECTED!           │ ← Alert banner (if violent)
│ Detected at 3:45:12 PM          │
├─────────────────────────────────┤
│ ● Status: MONITORING            │
│                                 │
│ ┌──────────┐  ┌──────────┐     │
│ │Violence  │  │Non-Viol. │     │
│ │  23.4%   │  │  76.6%   │     │ ← Live metrics
│ └──────────┘  └──────────┘     │
│                                 │
│ 📈 [Live updating chart]        │ ← Rolling chart
└─────────────────────────────────┘
```

---

## 🔥 Use Cases

### 1. Security Monitoring
- Mount laptop/phone on wall
- Point camera at entrance
- Monitors 24/7 for incidents
- Alerts security personnel instantly

### 2. Event Safety
- Monitor crowds at events
- Detect fights/altercations
- Real-time security response
- Document incidents automatically

### 3. School/Campus Safety
- Monitor hallways, cafeteria
- Detect bullying or fights
- Alert staff immediately
- Review footage later

### 4. Retail Loss Prevention
- Monitor store aisles
- Detect aggressive behavior
- Prevent theft/robbery
- Staff safety alerts

### 5. Public Transportation
- Monitor bus/train interiors
- Detect passenger conflicts
- Driver safety alerts
- Evidence collection

---

## ⚙️ Technical Details

### Performance
- **Processing Speed**: ~2 fps (every 0.5 seconds)
- **Latency**: <1 second from capture to result
- **Frame Buffer**: 20 frames (4-5 seconds of video)
- **Alert Threshold**: 70% confidence for violence

### Browser Requirements
- **Chrome/Edge 90+**: ✅ Full support
- **Firefox 88+**: ✅ Full support
- **Safari 14+**: ✅ Full support (may need HTTPS)
- **Mobile browsers**: ✅ Works on iOS/Android

### Camera Requirements
- **Resolution**: Any (auto-scales to 640x480)
- **FPS**: Any (captures 2fps)
- **Type**: Webcam, USB camera, phone camera

### Network Requirements
- **Connection**: WebSocket (ws:// or wss://)
- **Bandwidth**: ~50KB/s upload (per camera)
- **Latency**: <500ms recommended

---

## 🛠️ Configuration

### Adjust Detection Sensitivity
Edit `backend/app.py`:

```python
CONFIG = {
    'live_buffer_size': 20,        # Frames to analyze (20 = 4-5 sec)
    'live_update_interval': 0.5,   # Process every X seconds
}

# Alert threshold (in process_live_detection function)
if result['is_violent'] and result['confidence'] > 0.7:  # Change 0.7 to adjust
```

### Adjust Frame Capture Rate
Edit `frontend/index.html`:

```javascript
liveInterval = setInterval(() => {
    captureAndSendFrame();
}, 500);  // Change 500 (ms) to adjust capture rate
```

**Lower = more frames = higher accuracy + more CPU usage**
**Higher = fewer frames = lower accuracy + less CPU usage**

---

## 🚨 Alerts & Notifications

### Current Alerts
- ⚠️ **Visual alert banner** (red, animated shake)
- 🔴 **Video overlay changes** to "VIOLENCE DETECTED"
- 📊 **Metrics cards** show high violence probability
- 🕒 **Timestamp** of detection

### Future Enhancements (Roadmap)
- 🔔 **Browser notifications** (desktop/mobile)
- 📧 **Email alerts** to security team
- 📱 **SMS alerts** for critical incidents
- 🔊 **Audio alarm** (configurable)
- 📹 **Auto-record** incident clips
- 🌐 **Webhook** to external systems
- 📊 **Incident log** dashboard

---

## 🔐 Privacy & Security

### Data Handling
- ✅ **No storage**: Frames processed and discarded immediately
- ✅ **No recording**: Video not saved unless explicitly configured
- ✅ **Local processing**: All AI runs on your server
- ✅ **Encrypted**: Use HTTPS (wss://) for secure transmission

### Camera Access
- 🔒 **User permission required**: Browser prompts for camera access
- 🔒 **Green indicator**: Browser shows camera is active
- 🔒 **Easy stop**: One-click to stop monitoring and turn off camera
- 🔒 **Mode switching**: Switching modes stops camera automatically

### Recommendations
- Use HTTPS/WSS in production
- Implement user authentication
- Add audit logging for detections
- Comply with local surveillance laws
- Post visible camera notices

---

## 🧪 Testing Live Mode

### Test Scenarios

**1. Normal Activity**
- Walk around normally
- Talk to camera
- **Expected**: Low violence probability (< 30%)

**2. Simulated Conflict** (safe testing)
- Fast arm movements
- Aggressive gestures
- **Expected**: Medium probability (30-60%)

**3. Action Videos** (use Upload Mode)
- Play fight scene on screen
- Point camera at screen
- **Expected**: High probability (> 70%), alert triggered

### Troubleshooting

**Camera Not Working**
- Check browser permissions (camera icon in URL bar)
- Try different browser
- Ensure camera not used by another app
- Restart browser

**No Metrics Updating**
- Check WebSocket connection status (top-right)
- Check browser console for errors (F12)
- Verify backend is running
- Check network connectivity

**False Positives**
- Adjust alert threshold (increase from 0.7 to 0.8+)
- Increase buffer size for more context
- Improve lighting conditions
- Position camera for clearer view

**Slow Performance**
- Reduce frame capture rate (increase interval)
- Close other browser tabs
- Use GPU acceleration (if available)
- Check server CPU usage

---

## 📈 Performance Metrics

### Expected Performance

| Metric | Value |
|--------|-------|
| Frames/Second | 2 fps |
| Latency | 0.5-1.0s |
| CPU Usage | 15-30% (per camera) |
| RAM Usage | ~2GB (model loaded) |
| Network Upload | ~50KB/s |
| Alert Response | <1 second |

### Scaling

**Single Camera**: 1 user, 1 camera
- ✅ CPU: Intel i5 or better
- ✅ RAM: 4GB minimum
- ✅ Network: Any broadband

**Multi-Camera** (5-10 cameras):
- ✅ CPU: Intel i7/Xeon
- ✅ RAM: 8-16GB
- ✅ Network: 100Mbps+
- ✅ GPU: Optional (10x faster)

**Enterprise** (50+ cameras):
- ✅ CPU: Multi-core Xeon / AMD EPYC
- ✅ RAM: 32GB+
- ✅ GPU: NVIDIA Tesla/Quadro
- ✅ Load Balancer: Distribute across servers

---

## 🎯 Comparison: Upload vs Live

| Feature | Upload Mode | Live Mode |
|---------|-------------|-----------|
| **Input** | Video files | Webcam/camera |
| **Processing** | Batch (all frames at once) | Streaming (continuous) |
| **Speed** | 3-5 seconds total | 2 fps ongoing |
| **Use Case** | Analyze recordings | Monitor in real-time |
| **Alerts** | Results at end | Immediate alerts |
| **Chart** | All 20 frames shown | Rolling last 20 points |
| **Best For** | Post-incident review | Live security monitoring |

---

## 🚀 Getting Started Checklist

- [ ] Deploy backend with live detection support
- [ ] Access web interface (vision.nexaratech.io)
- [ ] Click "Live Detection" mode
- [ ] Grant camera permission
- [ ] Click "Start Live Monitoring"
- [ ] Verify webcam activates
- [ ] Check metrics update every 0.5s
- [ ] Test with movement to see chart update
- [ ] Click "Stop Monitoring" to end

---

## 📞 Support

**Issues?**
- Check FEATURES.md for full feature list
- Review README.md for deployment guide
- Check browser console (F12) for errors

**Production Deployment**:
```bash
cd /home/admin/Desktop/NexaraVision/web_prototype
./deploy.sh production
# Access at https://vision.nexaratech.io
```

---

**Nexara Vision Prototype v1.0**
**Upload & Live Detection | Powered by NexaraTech**
