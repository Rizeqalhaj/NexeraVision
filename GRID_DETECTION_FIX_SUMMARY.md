# Multi-Camera Grid Detection - Fix Summary

## Issues Fixed

### 1. Backend Not Running
**Problem**: GridDetector backend service wasn't started
**Fix**: Created startup script `grid_detection/start_backend.sh`

### 2. Port Configuration
**Problem**: Code used port 5000, project uses 8000-8003 range
**Fix**: Changed GridDetector to use port **8004**
- `api_integration.py` line 145
- `MultiCameraGrid.tsx` line 79
- `start_backend.sh` line 24

### 3. CORS Configuration
**Problem**: Backend only allowed localhost:3000 and localhost:3001
**Fix**: Added localhost:8001 to CORS allowed origins
- `api_integration.py` line 21

### 4. API Response Format
**Problem**: Backend returned `gridLayout: {rows, cols}`, frontend expected `grid_layout: [rows, cols]`
**Fix**: Changed response format to return array
- `api_integration.py` line 59

### 5. Error Visibility
**Problem**: Silent failures made debugging difficult
**Fix**: Added comprehensive console logging to `MultiCameraGrid.tsx`
- Lines 65-66, 72, 78, 84, 87-88, 93, 96-97, 106, 111-113

## How to Start Backend (Manual Method - FASTER)

The automatic installation is slow because NumPy compiles from source. Use system packages instead:

```bash
cd /home/admin/Desktop/NexaraVision/grid_detection

# Activate venv
source venv/bin/activate

# Install using system NumPy and OpenCV (faster)
pip install fastapi uvicorn
pip install --no-build-isolation numpy opencv-python

# OR use system packages
# sudo apt install python3-opencv python3-numpy
# pip install fastapi uvicorn

# Start backend on port 8004
python3 -m uvicorn api_integration:app --host 0.0.0.0 --port 8004 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8004 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Test Backend

```bash
# Health check
curl http://localhost:8004/health

# Expected response:
# {"status":"healthy","service":"grid-detection"}
```

## Test Detection on Frontend

1. **Start backend** (see above)

2. **Open frontend**: http://localhost:8001/live

3. **Navigate to**: Multi-Camera Grid tab

4. **Enable auto-detection**: ✅ Check "Auto-detect camera grid"

5. **Play CCTV video**: https://motionarray.com/stock-video/cctv-split-screen-surveillance-monitors-1528116/

6. **Start recording**: Click "Start Screen Recording" → Select video tab → Share

7. **Watch browser console** for detection logs:
   ```
   [GridDetection] Starting detection...
   [GridDetection] Canvas size: 1920 x 1080
   [GridDetection] Blob created, size: 245678 bytes
   [GridDetection] Sending request to http://localhost:8004/api/detect-grid
   [GridDetection] Response status: 200 OK
   [GridDetection] Result: {success: true, regions: [...], grid_layout: [3,4], ...}
   [GridDetection] ✅ Detected 12 cameras (3×4 grid, confidence: 87%)
   ```

8. **Expected UI**: Status message should show:
   > ✅ Detected 12 cameras (3×4 grid, confidence: 87%)

## Port Allocation Reference

- **Frontend**: 8001
- **Backend (NexaraVision)**: 8002
- **ML Services**: 8003
- **GridDetector Backend**: 8004 ⭐ (NEW)

## Files Changed

1. `/home/admin/Desktop/NexaraVision/grid_detection/api_integration.py`
   - Line 21: Added localhost:8001 to CORS
   - Line 59: Fixed response format (grid_layout array)
   - Line 145: Changed port to 8004

2. `/home/admin/Desktop/NexaraVision/web_app_nextjs/src/app/live/components/MultiCameraGrid.tsx`
   - Line 79: Changed API endpoint to port 8004
   - Lines 62-116: Added comprehensive logging and error handling

3. `/home/admin/Desktop/NexaraVision/grid_detection/start_backend.sh`
   - Lines 4, 24, 26-27: Changed port to 8004

## Troubleshooting

### Backend won't start
```bash
# Check if port 8004 is in use
lsof -i :8004

# Kill existing process if needed
kill -9 <PID>
```

### CORS errors in browser
Check backend console output - should show:
```
INFO:     127.0.0.1:xxxxx - "POST /api/detect-grid HTTP/1.1" 200 OK
```

Not:
```
WARNING:  CORS: Request from origin http://localhost:8001 not allowed
```

### Detection always fails
1. Check backend is running: `curl http://localhost:8004/health`
2. Check browser console for fetch errors
3. Try with clearer CCTV grid (less overlays)
4. Use manual grid mode as fallback

### "Could not connect to server"
Backend is not running. Start it manually (see above).

## Quick Test Commands

```bash
# Terminal 1: Start backend
cd /home/admin/Desktop/NexaraVision/grid_detection
source venv/bin/activate
python3 -m uvicorn api_integration:app --host 0.0.0.0 --port 8004

# Terminal 2: Test backend
curl http://localhost:8004/health

# Terminal 3: Test frontend
# Open http://localhost:8001/live in browser
# Open DevTools Console (F12)
# Test detection with CCTV video
```

## Success Criteria

✅ Backend responds to health check
✅ Frontend shows detection status message
✅ Browser console shows `[GridDetection]` logs
✅ Camera grid updates from manual (3×3) to detected (e.g., 3×4)
✅ Individual camera regions are cropped correctly

---

**Last Updated**: 2025-11-15
**Status**: Backend installation in progress (NumPy compiling from source - slow)
**Recommendation**: Use manual installation method above for faster startup
