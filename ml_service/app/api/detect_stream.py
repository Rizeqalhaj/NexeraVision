"""
Streaming video detection endpoint with real-time progress updates.
Uses Server-Sent Events (SSE) for instant feedback during processing.
"""
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np

from app.models.optimized_detector import OptimizedViolenceDetector
from app.utils.frame_extraction import validate_video
from app.core.config import settings
import cv2

logger = logging.getLogger(__name__)

router = APIRouter()

# Global optimized model instance
detector: Optional[OptimizedViolenceDetector] = None


def initialize_optimized_detector():
    """Initialize the optimized violence detector."""
    global detector
    if detector is None:
        logger.info("Initializing optimized TFLite detector...")
        detector = OptimizedViolenceDetector(
            settings.MODEL_PATH,
            optimize_for_speed=True
        )
        logger.info("Optimized detector ready")


async def extract_frames_with_progress(
    video_path: str,
    num_frames: int = 20,
    target_size: tuple = (224, 224)
) -> AsyncGenerator[dict, None]:
    """
    Extract frames with real-time progress updates.

    Yields:
        Progress updates and final frames
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        yield {
            "type": "error",
            "message": "Failed to open video file"
        }
        return

    try:
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_video_frames - 1, num_frames, dtype=int)

        frames = []
        extraction_start = time.time()

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                frames.append(np.zeros((*target_size[::-1], 3), dtype=np.float32))

            # Yield progress update
            progress = ((i + 1) / num_frames) * 100
            yield {
                "type": "progress",
                "stage": "extraction",
                "frame": i + 1,
                "total": num_frames,
                "progress": round(progress, 1),
                "message": f"Extracting frame {i + 1}/{num_frames}"
            }

            # Small delay to prevent overwhelming the event stream
            await asyncio.sleep(0.01)

        extraction_time = (time.time() - extraction_start) * 1000
        yield {
            "type": "progress",
            "stage": "extraction_complete",
            "progress": 100,
            "message": f"Extracted {num_frames} frames in {extraction_time:.0f}ms",
            "extraction_time_ms": round(extraction_time, 2)
        }

        # Return frames array
        yield {
            "type": "frames",
            "data": np.array(frames[:num_frames])
        }

    finally:
        cap.release()


@router.post("/detect_stream")
async def detect_violence_stream(video: UploadFile = File(...)):
    """
    Stream violence detection with real-time progress updates.

    Uses Server-Sent Events to provide instant feedback:
    - Frame extraction progress
    - Inference progress
    - Final results

    Response format (SSE):
        data: {"type": "progress", "stage": "extraction", "progress": 50}
        data: {"type": "progress", "stage": "inference", "progress": 100}
        data: {"type": "result", "violence_probability": 0.85, ...}
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="File must be a video"
        )

    video_bytes = await video.read()
    file_size_mb = len(video_bytes) / (1024 * 1024)

    if file_size_mb > 500:
        raise HTTPException(
            status_code=413,
            detail=f"Video too large: {file_size_mb:.1f}MB (max 500MB)"
        )

    async def generate_events():
        """Generate SSE events for processing progress."""
        total_start = time.time()

        # Initial event
        yield f"data: {json.dumps({'type': 'start', 'filename': video.filename, 'size_mb': round(file_size_mb, 2)})}\n\n"

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            # Validate video
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'validation', 'message': 'Validating video file...'})}\n\n"
            video_info = validate_video(tmp_path)

            if not video_info.get("valid"):
                error_msg = video_info.get("error", "Unknown error")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Invalid video: {error_msg}'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'progress', 'stage': 'validation_complete', 'video_info': video_info, 'message': 'Video validated'})}\n\n"

            # Extract frames with progress
            frames_data = None
            async for update in extract_frames_with_progress(tmp_path, settings.NUM_FRAMES, settings.FRAME_SIZE):
                if update["type"] == "frames":
                    frames_data = update["data"]
                else:
                    yield f"data: {json.dumps(update)}\n\n"

            if frames_data is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to extract frames'})}\n\n"
                return

            # Run inference
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'inference', 'progress': 0, 'message': 'Running AI inference...'})}\n\n"

            inference_start = time.time()
            result = detector.predict(frames_data)
            inference_time = (time.time() - inference_start) * 1000

            yield f"data: {json.dumps({'type': 'progress', 'stage': 'inference_complete', 'progress': 100, 'message': f'Inference completed in {inference_time:.0f}ms', 'inference_time_ms': round(inference_time, 2)})}\n\n"

            # Add metadata
            result["video_metadata"] = {
                "filename": video.filename,
                "duration_seconds": video_info.get("duration_seconds"),
                "fps": video_info.get("fps"),
                "resolution": f"{video_info.get('width')}x{video_info.get('height')}",
                "total_frames": video_info.get("total_frames"),
            }

            total_time = (time.time() - total_start) * 1000
            result["total_processing_time_ms"] = round(total_time, 2)

            # Final result
            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"

            logger.info(f"Streaming detection complete: {result['prediction']} in {total_time:.0f}ms")

        except Exception as e:
            logger.error(f"Streaming detection error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

            # End stream
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/detect_fast")
async def detect_fast(video: UploadFile = File(...)):
    """
    Fast detection endpoint using optimized TFLite model.

    Optimized for speed with minimal overhead.
    Target: <500ms total response time.
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    video_bytes = await video.read()
    file_size_mb = len(video_bytes) / (1024 * 1024)

    if file_size_mb > 500:
        raise HTTPException(status_code=413, detail=f"Video too large: {file_size_mb:.1f}MB")

    logger.info(f"Fast detection: {video.filename} ({file_size_mb:.1f}MB)")
    total_start = time.time()

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        # Validate
        video_info = validate_video(tmp_path)
        if not video_info.get("valid"):
            raise HTTPException(status_code=400, detail=f"Invalid video: {video_info.get('error')}")

        # Extract frames (optimized)
        extraction_start = time.time()
        cap = cv2.VideoCapture(tmp_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_video_frames - 1, settings.NUM_FRAMES, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, settings.FRAME_SIZE, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                frames.append(np.zeros((*settings.FRAME_SIZE[::-1], 3), dtype=np.float32))
        cap.release()

        extraction_time = (time.time() - extraction_start) * 1000

        # Inference
        result = detector.predict(np.array(frames[:settings.NUM_FRAMES]))

        # Add metadata
        result["video_metadata"] = {
            "filename": video.filename,
            "duration_seconds": video_info.get("duration_seconds"),
            "fps": video_info.get("fps"),
            "resolution": f"{video_info.get('width')}x{video_info.get('height')}",
            "total_frames": video_info.get("total_frames"),
        }

        total_time = (time.time() - total_start) * 1000
        result["timing"] = {
            "extraction_ms": round(extraction_time, 2),
            "inference_ms": result.get("inference_time_ms", 0),
            "total_ms": round(total_time, 2),
        }

        logger.info(f"Fast detection: {result['prediction']} in {total_time:.0f}ms (inference: {result.get('inference_time_ms', 0)}ms)")
        return result

    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass
