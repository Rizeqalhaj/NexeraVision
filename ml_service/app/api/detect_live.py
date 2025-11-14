"""
Live video stream detection endpoint.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import numpy as np

from app.models.violence_detector import ViolenceDetector
from app.utils.frame_extraction import decode_base64_frame
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model instance (shared with detect.py)
detector: Optional[ViolenceDetector] = None


def initialize_detector():
    """Initialize the violence detector model."""
    global detector
    if detector is None:
        detector = ViolenceDetector(settings.MODEL_PATH)


class LiveDetectionRequest(BaseModel):
    """Request schema for live detection."""
    frames: List[str] = Field(
        ...,
        min_items=20,
        max_items=20,
        description="Exactly 20 base64 encoded image frames"
    )


class LiveDetectionResult(BaseModel):
    """Live detection result schema."""
    violence_probability: float
    confidence: str
    prediction: str
    per_class_scores: dict


@router.post("/detect_live", response_model=LiveDetectionResult)
async def detect_live(request: LiveDetectionRequest):
    """
    Detect violence in live video stream frames.

    Accepts 20 base64 encoded frames from webcam/live stream and returns
    violence probability in real-time.

    Args:
        request: LiveDetectionRequest with 20 base64 encoded frames

    Returns:
        Detection results with violence probability
    """
    if len(request.frames) != 20:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly 20 frames, got {len(request.frames)}"
        )

    logger.info("Processing live detection request")

    try:
        # Decode all frames
        frames = []
        for i, frame_b64 in enumerate(request.frames):
            frame = decode_base64_frame(frame_b64, target_size=settings.FRAME_SIZE)

            if frame is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode frame {i}"
                )

            frames.append(frame)

        # Convert to numpy array
        frames_array = np.array(frames)

        # Predict
        result = detector.predict(frames_array)

        logger.info(
            f"Live detection: {result['prediction']} "
            f"({result['violence_probability']:.2%})"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect_live_batch")
async def detect_live_batch(requests: List[LiveDetectionRequest]):
    """
    Batch process multiple live detection requests.

    Optimized for processing multiple camera streams simultaneously.

    Args:
        requests: List of LiveDetectionRequest objects

    Returns:
        List of detection results
    """
    if len(requests) > 32:
        raise HTTPException(
            status_code=400,
            detail=f"Max batch size is 32, got {len(requests)}"
        )

    logger.info(f"Processing batch of {len(requests)} live detections")

    try:
        # Decode all frames for all requests
        all_frames = []
        for req_idx, request in enumerate(requests):
            if len(request.frames) != 20:
                raise HTTPException(
                    status_code=400,
                    detail=f"Request {req_idx}: Expected 20 frames, got {len(request.frames)}"
                )

            frames = []
            for i, frame_b64 in enumerate(request.frames):
                frame = decode_base64_frame(frame_b64, target_size=settings.FRAME_SIZE)

                if frame is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Request {req_idx}, frame {i}: Failed to decode"
                    )

                frames.append(frame)

            all_frames.append(np.array(frames))

        # Convert to batch array
        frames_batch = np.array(all_frames)

        # Batch prediction
        results = detector.predict_batch(frames_batch)

        logger.info(f"Batch detection complete: {len(results)} results")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
