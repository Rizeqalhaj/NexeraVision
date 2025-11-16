"""
FastAPI Integration for Grid Detection

Provides REST API endpoints for the web application to use grid detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import List, Dict
import io
from detector import GridDetector

app = FastAPI(title="Grid Detection API", version="1.0.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = GridDetector(min_confidence=0.7)


@app.post("/api/detect-grid")
async def detect_grid(screenshot: UploadFile = File(...)) -> JSONResponse:
    """
    Detect camera grid from CCTV screenshot

    Research-validated: 80-91% automatic success rate

    Args:
        screenshot: Screenshot image file (JPEG, PNG)

    Returns:
        Detection result with camera regions
    """
    try:
        # Read image
        contents = await screenshot.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run detection
        result = detector.detect(image)

        # Calculate actual camera grid dimensions from number of detected regions
        num_cameras = len(result['regions'])
        # Try to infer grid dimensions from number of cameras
        # Common layouts: 1, 2x2=4, 3x3=9, 4x4=16, 2x3=6, 3x4=12
        if num_cameras == 1:
            grid_rows, grid_cols = 1, 1
        elif num_cameras == 4:
            grid_rows, grid_cols = 2, 2
        elif num_cameras == 6:
            grid_rows, grid_cols = 2, 3
        elif num_cameras == 9:
            grid_rows, grid_cols = 3, 3
        elif num_cameras == 12:
            grid_rows, grid_cols = 3, 4
        elif num_cameras == 16:
            grid_rows, grid_cols = 4, 4
        else:
            # Use the detected grid layout (number of lines - 1 = number of cells)
            grid_rows = result['grid_layout'][0]
            grid_cols = result['grid_layout'][1]

        return JSONResponse(content={
            "success": bool(result['success']),
            "regions": result['regions'],
            "grid_layout": [grid_rows, grid_cols],  # Return actual camera grid dimensions
            "confidence": float(result['confidence']),
            "requires_manual": bool(result['requires_manual']),
            "message": get_message(result)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-cameras")
async def extract_cameras(
    screenshot: UploadFile = File(...),
    regions: List[Dict] = None
) -> JSONResponse:
    """
    Extract individual camera frames from screenshot using defined regions

    Args:
        screenshot: Screenshot image file
        regions: List of camera regions (from detection or manual calibration)

    Returns:
        List of base64-encoded camera images
    """
    try:
        # Read image
        contents = await screenshot.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Extract cameras
        camera_frames = []

        for i, region in enumerate(regions):
            x, y, w, h = region['x'], region['y'], region['width'], region['height']

            # Crop camera region
            camera = image[y:y+h, x:x+w]

            # Resize to standard size (640x360 for model input)
            camera_resized = cv2.resize(camera, (640, 360))

            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', camera_resized)
            camera_base64 = buffer.tobytes().hex()

            camera_frames.append({
                'cameraId': i + 1,
                'image': camera_base64,
                'region': region
            })

        return JSONResponse(content={
            "success": True,
            "cameras": camera_frames,
            "count": len(camera_frames)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "grid-detection"}


def get_message(result: Dict) -> str:
    """Generate user-friendly message based on detection result"""
    if result['success'] and result['confidence'] >= 0.9:
        return f"✅ Grid detected successfully! {len(result['regions'])} cameras found with high confidence."
    elif result['success']:
        return f"✅ Grid detected! {len(result['regions'])} cameras found. You may adjust boundaries if needed."
    elif result['confidence'] >= 0.5:
        return f"⚠️ Partial detection. {len(result['regions'])} cameras found but manual calibration recommended."
    else:
        return "❌ Auto-detection failed. Please use manual calibration to define camera boundaries."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
