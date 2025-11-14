"""
Video frame extraction utilities with robust error handling.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    num_frames: int = 20,
    target_size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    Extract frames uniformly from video with preprocessing.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: Target size for resizing (width, height)

    Returns:
        Numpy array of shape (num_frames, height, width, 3) or None on failure
    """
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            logger.warning(
                f"Video has only {total_frames} frames, "
                f"requested {num_frames}. Will duplicate frames."
            )
            frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Resize to target size
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                # Add black frame if read fails
                logger.warning(f"Failed to read frame {idx}, adding black frame")
                black_frame = np.zeros((*target_size[::-1], 3), dtype=np.float32)
                frames.append(black_frame)

        # Handle case where video has fewer frames than requested
        while len(frames) < num_frames:
            # Duplicate last frame
            frames.append(frames[-1].copy())

        return np.array(frames[:num_frames])

    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        return None

    finally:
        cap.release()


def validate_video(video_path: str) -> dict:
    """
    Validate video file and return metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata and validation status
    """
    if not Path(video_path).exists():
        return {
            "valid": False,
            "error": "File not found"
        }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "valid": False,
            "error": "Could not open video file"
        }

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        return {
            "valid": total_frames > 0,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

    finally:
        cap.release()


def decode_base64_frame(frame_b64: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """
    Decode base64 encoded frame to numpy array.

    Args:
        frame_b64: Base64 encoded image string
        target_size: Target size for resizing (width, height)

    Returns:
        Numpy array of shape (height, width, 3) or None on failure
    """
    import base64

    try:
        # Remove data URI prefix if present
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]

        # Decode base64
        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Failed to decode image from base64")
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        # Normalize
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        logger.error(f"Error decoding base64 frame: {e}")
        return None
