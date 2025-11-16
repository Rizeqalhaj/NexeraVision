"""
Grid segmentation API endpoints.

Provides REST API for multi-camera grid detection and segmentation.
"""
import base64
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.segmentation import GridDetector, VideoSegmenter, QualityEnhancer, AutoCalibrator

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances
video_segmenter = VideoSegmenter()
quality_enhancer = QualityEnhancer()
auto_calibrator = AutoCalibrator()


class GridDetectionRequest(BaseModel):
    """Request model for grid detection."""
    frame_base64: str = Field(..., description="Base64 encoded video frame")
    visualize: bool = Field(default=False, description="Return visualization image")


class GridDetectionResponse(BaseModel):
    """Response model for grid detection."""
    success: bool
    grid_layout: dict
    total_cameras: int
    visualization: Optional[str] = None


class CameraRegion(BaseModel):
    """Model for individual camera region."""
    id: int
    position: tuple
    bbox: tuple
    resolution: tuple
    is_active: bool
    frame_base64: Optional[str] = None
    enhanced: bool = False


class SegmentationResponse(BaseModel):
    """Response model for frame segmentation."""
    success: bool
    cameras: List[dict]
    grid_layout: dict
    processing_time_ms: float


class BatchSegmentationRequest(BaseModel):
    """Request model for batch video segmentation."""
    video_base64: str = Field(..., description="Base64 encoded video file")
    auto_calibrate: bool = Field(default=True, description="Auto-calibrate grid layout")
    enhance_quality: bool = Field(default=True, description="Apply quality enhancement")
    return_frames: bool = Field(default=False, description="Return segmented frames in response")


@router.post("/segment/detect-grid", response_model=GridDetectionResponse)
async def detect_grid(request: GridDetectionRequest):
    """
    Detect grid layout from a single video frame.

    Analyzes frame using Hough transform and DBSCAN to identify grid structure.
    Returns grid dimensions, line positions, and optional visualization.
    """
    try:
        # Decode frame
        frame_bytes = base64.b64decode(request.frame_base64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Detect grid
        detector = GridDetector(frame)
        grid_layout = detector.detect_grid_layout()

        response = {
            "success": True,
            "grid_layout": grid_layout,
            "total_cameras": grid_layout['total_cameras']
        }

        # Add visualization if requested
        if request.visualize:
            vis_frame = detector.visualize_grid()
            _, buffer = cv2.imencode('.jpg', vis_frame)
            vis_base64 = base64.b64encode(buffer).decode('utf-8')
            response["visualization"] = vis_base64

        return response

    except Exception as e:
        logger.error(f"Grid detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment/extract-cameras")
async def extract_cameras(
    frame: UploadFile = File(...),
    grid_rows: Optional[int] = Form(None),
    grid_cols: Optional[int] = Form(None),
    enhance: bool = Form(True),
    return_frames: bool = Form(False)
):
    """
    Extract individual camera feeds from grid frame.

    Can either auto-detect grid or use manually specified dimensions.
    Optionally applies quality enhancement to low-resolution feeds.
    """
    import time
    start_time = time.time()

    try:
        # Read frame
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Create grid layout
        if grid_rows and grid_cols:
            # Manual grid specification
            height, width = frame_img.shape[:2]
            h_step = height / grid_rows
            v_step = width / grid_cols

            grid_layout = {
                'rows': grid_rows,
                'cols': grid_cols,
                'h_lines': [i * h_step for i in range(grid_rows + 1)],
                'v_lines': [i * v_step for i in range(grid_cols + 1)],
                'total_cameras': grid_rows * grid_cols
            }
            logger.info(f"Using manual grid: {grid_rows}x{grid_cols}")
        else:
            # Auto-detect grid
            detector = GridDetector(frame_img)
            grid_layout = detector.detect_grid_layout()
            logger.info(f"Auto-detected grid: {grid_layout['rows']}x{grid_layout['cols']}")

        # Extract camera regions
        detector = GridDetector(frame_img)
        cameras = detector.extract_camera_regions(grid_layout=grid_layout)

        # Enhance quality if requested
        if enhance:
            cameras = quality_enhancer.batch_enhance(cameras)

        # Prepare response
        camera_data = []
        for camera in cameras:
            cam_dict = {
                'id': camera['id'],
                'position': camera['position'],
                'bbox': camera['bbox'],
                'resolution': camera['resolution'],
                'is_active': camera['is_active'],
                'enhanced': camera.get('enhanced', False)
            }

            # Include frame data if requested
            if return_frames:
                _, buffer = cv2.imencode('.jpg', camera['frame'])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                cam_dict['frame_base64'] = frame_base64

            camera_data.append(cam_dict)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "success": True,
            "cameras": camera_data,
            "grid_layout": grid_layout,
            "processing_time_ms": processing_time
        }

    except Exception as e:
        logger.error(f"Camera extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment/process-video")
async def process_video(
    video: UploadFile = File(...),
    auto_calibrate: bool = Form(True),
    enhance_quality: bool = Form(True),
    save_individual_feeds: bool = Form(False)
):
    """
    Process entire video and segment into individual camera feeds.

    Handles multi-camera grid video files and returns metadata or saves
    individual camera video files.
    """
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            contents = await video.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        logger.info(f"Processing video: {tmp_path}")

        # Create segmenter
        segmenter = VideoSegmenter()

        # Calibrate if requested
        if auto_calibrate:
            grid_layout = segmenter.calibrate(tmp_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Manual grid specification not yet implemented for video processing"
            )

        # Setup output directory if saving feeds
        output_dir = None
        if save_individual_feeds:
            output_dir = Path(tempfile.mkdtemp())
            logger.info(f"Saving individual feeds to: {output_dir}")

        # Process video
        frame_count = 0
        camera_stats = {}

        for cameras in segmenter.segment_video(tmp_path, output_dir=output_dir):
            frame_count += 1

            # Collect statistics
            for camera in cameras:
                cam_id = camera['id']
                if cam_id not in camera_stats:
                    camera_stats[cam_id] = {
                        'id': cam_id,
                        'position': camera['position'],
                        'resolution': camera['resolution'],
                        'active_frames': 0,
                        'total_frames': 0
                    }

                camera_stats[cam_id]['total_frames'] += 1
                if camera['is_active']:
                    camera_stats[cam_id]['active_frames'] += 1

        # Cleanup
        Path(tmp_path).unlink()

        response = {
            "success": True,
            "grid_layout": grid_layout,
            "total_frames": frame_count,
            "camera_statistics": list(camera_stats.values()),
            "output_directory": str(output_dir) if output_dir else None
        }

        logger.info(f"Video processing complete: {frame_count} frames, {len(camera_stats)} cameras")
        return response

    except Exception as e:
        logger.error(f"Video processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment/calibrate")
async def calibrate_grid(
    video: UploadFile = File(...),
    calibration_frames: int = Form(5)
):
    """
    Calibrate grid layout from video file.

    Analyzes multiple frames to determine consistent grid structure.
    Returns grid layout that can be used for subsequent segmentation.
    """
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            contents = await video.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        # Calibrate
        segmenter = VideoSegmenter(calibrate_frames=calibration_frames)
        grid_layout = segmenter.calibrate(tmp_path)

        # Cleanup
        Path(tmp_path).unlink()

        return {
            "success": True,
            "grid_layout": grid_layout,
            "calibration_frames_used": calibration_frames
        }

    except Exception as e:
        logger.error(f"Grid calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/segment/supported-layouts")
async def get_supported_layouts():
    """
    Get list of common supported grid layouts.

    Returns standard surveillance grid configurations.
    """
    layouts = [
        {"rows": 2, "cols": 2, "total": 4, "name": "2x2 Grid"},
        {"rows": 3, "cols": 3, "total": 9, "name": "3x3 Grid"},
        {"rows": 4, "cols": 4, "total": 16, "name": "4x4 Grid"},
        {"rows": 5, "cols": 5, "total": 25, "name": "5x5 Grid"},
        {"rows": 6, "cols": 6, "total": 36, "name": "6x6 Grid"},
        {"rows": 8, "cols": 8, "total": 64, "name": "8x8 Grid"},
        {"rows": 10, "cols": 10, "total": 100, "name": "10x10 Grid"},
        {"rows": 2, "cols": 3, "total": 6, "name": "2x3 Grid"},
        {"rows": 3, "cols": 4, "total": 12, "name": "3x4 Grid"},
        {"rows": 4, "cols": 5, "total": 20, "name": "4x5 Grid"},
    ]

    return {
        "success": True,
        "supported_layouts": layouts,
        "max_cameras": 144  # 12x12 theoretical maximum
    }


@router.get("/segment/health")
async def health_check():
    """Health check endpoint for segmentation service."""
    return {
        "status": "operational",
        "service": "grid_segmentation",
        "features": {
            "auto_detection": True,
            "manual_specification": True,
            "quality_enhancement": True,
            "batch_processing": True
        }
    }
