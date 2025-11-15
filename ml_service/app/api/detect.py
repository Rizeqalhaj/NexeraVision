"""
Video file upload detection endpoint.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.models.violence_detector import ViolenceDetector
from app.utils.frame_extraction import extract_frames, validate_video
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model instance (loaded on startup)
detector: Optional[ViolenceDetector] = None


def initialize_detector():
    """Initialize the violence detector model."""
    global detector
    if detector is None:
        detector = ViolenceDetector(settings.MODEL_PATH)


class DetectionResult(BaseModel):
    """Detection result schema."""
    violence_probability: float
    confidence: str
    prediction: str
    per_class_scores: dict
    video_metadata: dict


@router.post("/detect", response_model=DetectionResult)
async def detect_violence(video: UploadFile = File(...)):
    """
    Detect violence in uploaded video file.

    Args:
        video: Uploaded video file (MP4, AVI, MOV, MKV)

    Returns:
        Detection results with violence probability and confidence
    """
    # Validate content type
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="File must be a video (MP4, AVI, MOV, MKV)"
        )

    # Validate file size
    video_bytes = await video.read()
    file_size_mb = len(video_bytes) / (1024 * 1024)

    if file_size_mb > 500:
        raise HTTPException(
            status_code=413,
            detail=f"Video file too large: {file_size_mb:.1f}MB (max 500MB)"
        )

    logger.info(f"Processing video: {video.filename} ({file_size_mb:.1f}MB)")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        # Validate video
        video_info = validate_video(tmp_path)
        if not video_info.get("valid"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video file: {video_info.get('error')}"
            )

        logger.info(f"Video metadata: {video_info}")

        # Extract frames
        frames = extract_frames(
            tmp_path,
            num_frames=settings.NUM_FRAMES,
            target_size=settings.FRAME_SIZE
        )

        if frames is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract frames from video"
            )

        # Predict
        result = detector.predict(frames)

        # Add video metadata
        result["video_metadata"] = {
            "filename": video.filename,
            "duration_seconds": video_info.get("duration_seconds"),
            "fps": video_info.get("fps"),
            "resolution": f"{video_info.get('width')}x{video_info.get('height')}",
            "total_frames": video_info.get("total_frames"),
        }

        logger.info(
            f"Detection complete: {result['prediction']} "
            f"({result['violence_probability']:.2%} confidence)"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    if detector is None:
        return {
            "status": "unhealthy",
            "error": "Model not loaded"
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": detector.get_model_info(),
    }
