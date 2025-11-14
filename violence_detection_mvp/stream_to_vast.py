#!/usr/bin/env python3
"""
Stream local webcam to Vast.ai for violence detection
This runs on your LOCAL machine and sends frames to Vast.ai
"""

import cv2
import requests
import base64
import json
import time

def stream_webcam_to_vast(vast_url, camera_id=0):
    """
    Stream webcam frames to Vast.ai instance

    Args:
        vast_url: URL of your Vast.ai detection endpoint (e.g., http://vast-ip:8000/detect)
        camera_id: Local webcam ID (default: 0)
    """
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        return

    print(f"✅ Camera opened, streaming to {vast_url}")
    print("Press 'q' to quit")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Send to Vast.ai
            try:
                response = requests.post(
                    vast_url,
                    json={'frame': frame_b64},
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    is_violence = result.get('violence', False)
                    confidence = result.get('confidence', 0.0)

                    # Draw overlay
                    color = (0, 0, 255) if is_violence else (0, 255, 0)
                    status = "⚠️ VIOLENCE" if is_violence else "✅ Normal"

                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), color, -1)
                    cv2.putText(frame, status, (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(frame, f"Conf: {confidence*100:.1f}%",
                               (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            except requests.RequestException as e:
                print(f"⚠️ Connection error: {e}")

            # Display
            cv2.imshow('Webcam Stream - Press Q to quit', frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frames sent: {frame_count}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Streamed {frame_count} frames")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 stream_to_vast.py <vast_url> [camera_id]")
        print("Example: python3 stream_to_vast.py http://12.34.56.78:8000/detect 0")
        sys.exit(1)

    vast_url = sys.argv[1]
    camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    stream_webcam_to_vast(vast_url, camera_id)
