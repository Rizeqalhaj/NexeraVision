# Testing /live Features - Step by Step Guide

## Current Setup
- **Frontend**: http://localhost:8001/live
- **Expected API**: Check your backend port (3001 or 8001?)
- **ML Service**: Should be on port 8003

---

## 🧪 Test 1: File Upload Feature

### Steps:
1. **Open**: http://localhost:8001/live
2. **Click**: "File Upload" tab
3. **Select**: A test video file (MP4, AVI, or similar)
4. **Upload**: Click "Upload Video" button

### Expected Behavior:
```
✅ Upload starts with progress bar (0% → 100%)
✅ Processing message appears
✅ Results display:
   - Violence Probability: XX%
   - Confidence Level: Low/Medium/High
   - Prediction: Violence or Non-Violence
✅ Alert appears if violence detected (>85%)
```

### What to Check:
- **Browser Console** (F12): Look for network requests
- **Network Tab**: Check `/api/upload` request
  - Status: Should be 200 OK
  - Response: Should contain detection results

### Common Issues:
❌ **CORS Error**: API not allowing frontend requests
❌ **404 Error**: API endpoint not found (check port)
❌ **500 Error**: ML model loading failed
❌ **Network Error**: Backend not running

---

## 🎥 Test 2: Live Camera Detection

### Steps:
1. **Open**: http://localhost:8001/live
2. **Click**: "Live Camera" tab
3. **Click**: "Start Live Detection" button
4. **Allow**: Browser camera permission

### Expected Behavior:
```
✅ Browser asks for camera permission
✅ Camera feed appears in preview
✅ "LIVE" badge shows (red, pulsing)
✅ Violence Probability updates in real-time
✅ Progress bar changes color based on probability
✅ Alerts appear in "Recent Alerts" section when violence detected
```

### What to Check:
- **Browser Console** (F12): Look for WebSocket messages
- **Network Tab → WS**: Check WebSocket connection
  - Connection: `ws://localhost:XXXX/ws/live`
  - Status: 101 Switching Protocols (success)
  - Messages: Should see frames being sent, results coming back

### WebSocket Message Format:
```javascript
// Sent to server every ~0.66s:
{
  "type": "analyze_frames",
  "frames": ["base64_image1", "base64_image2", ...] // 20 frames
}

// Received from server:
{
  "type": "detection_result",
  "violence_probability": 0.42,
  "confidence": "Medium",
  "prediction": "non_violence",
  "per_class_scores": {
    "non_violence": 0.58,
    "violence": 0.42
  },
  "processing_time_ms": 87.34
}
```

### Common Issues:
❌ **Camera Permission Denied**: Allow camera in browser settings
❌ **WebSocket Connection Failed**: Check ML service is running
❌ **No Camera Found**: Connect a webcam or use different device
❌ **Frames Not Sending**: Check browser console for errors

---

## 🔍 Debugging Commands

### Check if ML Service is Running:
```bash
# Check Docker container
docker ps | grep nexara-ml-service

# Check logs
docker logs nexara-ml-service

# Test endpoint directly
curl http://localhost:8003/
```

### Check Backend API:
```bash
# Test health endpoint
curl http://localhost:3001/api/health

# Or port 8001 if that's where it's running
curl http://localhost:8001/api/health
```

### Browser Console Tests:
```javascript
// Test WebSocket connection
const ws = new WebSocket('ws://localhost:3001/ws/live');
ws.onopen = () => console.log('✅ WebSocket connected');
ws.onerror = (e) => console.error('❌ WebSocket error:', e);
ws.onmessage = (e) => console.log('📨 Message:', e.data);

// Test file upload API
fetch('http://localhost:3001/api/upload')
  .then(r => console.log('✅ API reachable:', r.status))
  .catch(e => console.error('❌ API error:', e));
```

---

## ✅ Success Criteria

### File Upload:
- ✅ Video uploads without errors
- ✅ Detection results appear within 5-10 seconds
- ✅ Probability percentage is shown (0-100%)
- ✅ Violence/Non-Violence label is displayed

### Live Camera:
- ✅ Camera feed shows in preview
- ✅ WebSocket connection stays active
- ✅ Probability updates every ~1 second
- ✅ Alert sound plays when violence detected (>85%)
- ✅ Recent alerts list populates

---

## 📊 Expected Model Performance (CPU)

- **Processing Time**: 60-100ms per frame batch
- **Update Frequency**: ~1 prediction per second
- **Accuracy**: Should be consistent with training metrics
- **Memory Usage**: ~2GB (model + inference)

---

## 🚨 If Tests Fail

### 1. Check Ports:
```bash
# Find what's running on each port
lsof -i :3001
lsof -i :8001
lsof -i :8003
```

### 2. Check ML Service Logs:
```bash
docker logs nexara-ml-service --tail 100
```

### 3. Check Frontend Environment:
```bash
# In web_app_nextjs directory
cat .env.local
# Should have:
# NEXT_PUBLIC_API_URL=http://localhost:XXXX/api
# NEXT_PUBLIC_WS_URL=ws://localhost:XXXX/ws/live
```

### 4. Restart Services:
```bash
# Restart ML service
docker restart nexara-ml-service

# Restart frontend
cd web_app_nextjs
npm run dev
```

---

**Ready to Test!** 🚀

Open http://localhost:8001/live and try both features!
