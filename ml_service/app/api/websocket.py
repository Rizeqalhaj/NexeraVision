"""
WebSocket endpoint for real-time live camera violence detection.

Handles streaming connections from frontend, processes frame batches,
and returns violence detection results in real-time.
"""
import logging
import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np

from app.models.violence_detector import ViolenceDetector
from app.utils.frame_extraction import decode_base64_frame
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model instance (shared with detect.py)
detector: Optional[ViolenceDetector] = None


class ConnectionManager:
    """
    Manages active WebSocket connections for live detection.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        """Send JSON data to WebSocket client."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")


manager = ConnectionManager()


@router.websocket("/ws/live")
async def websocket_live_detection(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live camera violence detection.

    Protocol:
        Client sends: {
            "type": "frames",
            "frames": ["base64_frame1", "base64_frame2", ...],  # Exactly 20 frames
            "timestamp": 1234567890.123
        }

        Server responds: {
            "type": "detection_result",
            "violence_probability": 0.85,
            "confidence": "High",
            "prediction": "violence",
            "per_class_scores": {
                "non_violence": 0.15,
                "violence": 0.85
            },
            "timestamp": 1234567890.123,
            "processing_time_ms": 123.45
        }

        On error: {
            "type": "error",
            "error": "Error message",
            "detail": "Detailed error information"
        }
    """
    await manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError as e:
                await manager.send_json(websocket, {
                    "type": "error",
                    "error": "Invalid JSON",
                    "detail": str(e)
                })
                continue

            # Validate message type
            if message.get("type") != "frames":
                await manager.send_json(websocket, {
                    "type": "error",
                    "error": "Invalid message type",
                    "detail": f"Expected 'frames', got '{message.get('type')}'"
                })
                continue

            # Extract frames
            frames_b64 = message.get("frames", [])
            timestamp = message.get("timestamp")

            # Validate frame count
            if len(frames_b64) != settings.NUM_FRAMES:
                await manager.send_json(websocket, {
                    "type": "error",
                    "error": "Invalid frame count",
                    "detail": f"Expected {settings.NUM_FRAMES} frames, got {len(frames_b64)}"
                })
                continue

            # Process frames
            import time
            start_time = time.time()

            try:
                # Decode all frames
                frames = []
                for i, frame_b64 in enumerate(frames_b64):
                    frame = decode_base64_frame(frame_b64, target_size=settings.FRAME_SIZE)

                    if frame is None:
                        await manager.send_json(websocket, {
                            "type": "error",
                            "error": "Frame decode failed",
                            "detail": f"Failed to decode frame {i}"
                        })
                        break

                    frames.append(frame)

                # Skip prediction if frame decoding failed
                if len(frames) != settings.NUM_FRAMES:
                    continue

                # Convert to numpy array
                frames_array = np.array(frames, dtype=np.float32)

                # Predict violence
                if detector is None:
                    await manager.send_json(websocket, {
                        "type": "error",
                        "error": "Model not initialized",
                        "detail": "Violence detector model is not loaded"
                    })
                    continue

                result = detector.predict(frames_array)

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # milliseconds

                # Send result back to client
                response = {
                    "type": "detection_result",
                    "violence_probability": result["violence_probability"],
                    "confidence": result["confidence"],
                    "prediction": result["prediction"],
                    "per_class_scores": result["per_class_scores"],
                    "timestamp": timestamp,
                    "processing_time_ms": round(processing_time, 2)
                }

                await manager.send_json(websocket, response)

                logger.info(
                    f"Live detection: {result['prediction']} "
                    f"({result['violence_probability']:.2%}) "
                    f"in {processing_time:.0f}ms"
                )

            except Exception as e:
                logger.error(f"Detection error: {e}", exc_info=True)
                await manager.send_json(websocket, {
                    "type": "error",
                    "error": "Detection failed",
                    "detail": str(e)
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected normally")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)


@router.get("/ws/status")
async def websocket_status():
    """
    Get WebSocket connection status and statistics.

    Returns:
        Dictionary with connection stats
    """
    return {
        "active_connections": len(manager.active_connections),
        "model_loaded": detector is not None,
        "endpoint": "/ws/live",
        "protocol": {
            "input_format": "JSON with 'type': 'frames' and 'frames': [base64_strings]",
            "frame_count": settings.NUM_FRAMES,
            "frame_size": settings.FRAME_SIZE,
        }
    }
