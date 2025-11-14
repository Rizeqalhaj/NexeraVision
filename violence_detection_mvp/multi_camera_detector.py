#!/usr/bin/env python3
"""
Multi-Camera Violence Detection System
Handles 18+ cameras efficiently with motion detection pre-filtering

Architecture:
- Lightweight motion detection on all cameras (2% CPU each)
- Full VGG19+BiLSTM only on cameras with movement
- Priority queue for processing active cameras
- Central dashboard for monitoring
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
from datetime import datetime
import argparse
import logging
import threading
import queue
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MotionDetector:
    """Lightweight motion detection for pre-filtering"""

    def __init__(self, threshold: float = 25.0, min_area: int = 500):
        """
        Initialize motion detector

        Args:
            threshold: Motion sensitivity (lower = more sensitive)
            min_area: Minimum motion area in pixels
        """
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
        self.motion_score = 0.0

    def detect_motion(self, frame: np.ndarray) -> tuple:
        """
        Fast motion detection using frame differencing

        Args:
            frame: Current frame (BGR)

        Returns:
            (has_motion, motion_score) tuple
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0

        # Compute difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate motion score
        motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > self.min_area)
        frame_area = gray.shape[0] * gray.shape[1]
        motion_score = (motion_area / frame_area) * 100  # Percentage

        self.prev_frame = gray
        self.motion_score = motion_score

        has_motion = motion_score > 0.5  # 0.5% of frame has motion

        return has_motion, motion_score


class CameraStream:
    """Individual camera stream handler"""

    def __init__(self, camera_id: int, source: str, motion_detector: MotionDetector):
        """
        Initialize camera stream

        Args:
            camera_id: Unique camera identifier
            source: Video source (camera index, IP camera URL, or file path)
            motion_detector: Motion detection instance
        """
        self.camera_id = camera_id
        self.source = source
        self.motion_detector = motion_detector

        self.cap = None
        self.is_active = False
        self.has_motion = False
        self.motion_score = 0.0
        self.last_frame = None
        self.frame_count = 0

        # Statistics
        self.total_frames = 0
        self.motion_frames = 0
        self.violence_detections = 0

    def start(self) -> bool:
        """Open camera stream"""
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            self.is_active = True
            logger.info(f"Camera {self.camera_id} started: {self.source}")
            return True
        else:
            logger.error(f"Failed to open camera {self.camera_id}: {self.source}")
            return False

    def stop(self):
        """Close camera stream"""
        if self.cap:
            self.cap.release()
            self.is_active = False
            logger.info(f"Camera {self.camera_id} stopped")

    def read_frame(self) -> tuple:
        """
        Read and analyze frame

        Returns:
            (success, frame, has_motion, motion_score)
        """
        if not self.is_active:
            return False, None, False, 0.0

        ret, frame = self.cap.read()
        if not ret:
            return False, None, False, 0.0

        self.total_frames += 1
        self.last_frame = frame

        # Detect motion
        has_motion, motion_score = self.motion_detector.detect_motion(frame)

        if has_motion:
            self.motion_frames += 1

        self.has_motion = has_motion
        self.motion_score = motion_score

        return True, frame, has_motion, motion_score


class MultiCameraViolenceDetector:
    """Multi-camera violence detection system with motion pre-filtering"""

    def __init__(
        self,
        model_path: str,
        num_cameras: int = 18,
        confidence_threshold: float = 0.7,
        max_concurrent_analysis: int = 5,
        use_gpu: bool = False
    ):
        """
        Initialize multi-camera detector

        Args:
            model_path: Path to trained BiLSTM model
            num_cameras: Number of cameras to monitor
            confidence_threshold: Violence detection threshold
            max_concurrent_analysis: Max cameras analyzed simultaneously
            use_gpu: Use GPU for inference
        """
        self.model_path = model_path
        self.num_cameras = num_cameras
        self.confidence_threshold = confidence_threshold
        self.max_concurrent_analysis = max_concurrent_analysis

        # Camera streams
        self.cameras = {}
        self.motion_detectors = {}

        # Processing queue (priority: cameras with motion)
        self.processing_queue = queue.PriorityQueue()

        # Load models
        self._load_models(use_gpu)

        # Frame buffers for each camera (20 frames)
        self.frame_buffers = {i: deque(maxlen=20) for i in range(num_cameras)}

        # Statistics
        self.stats = {
            'total_cameras': num_cameras,
            'active_cameras': 0,
            'cameras_with_motion': 0,
            'total_detections': 0,
            'fps_per_camera': {}
        }

    def _load_models(self, use_gpu: bool):
        """Load VGG19 and BiLSTM models"""
        import os

        # Configure device
        if use_gpu:
            logger.info("Using GPU for inference")
        else:
            # CPU optimizations
            tf.config.threading.set_intra_op_parallelism_threads(0)
            tf.config.threading.set_inter_op_parallelism_threads(0)
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
            logger.info("Using CPU with Intel optimizations")

        # Load VGG19
        logger.info("Loading VGG19 feature extractor...")
        base_model = tf.keras.applications.VGG19(
            include_top=True,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False

        self.feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('fc2').output
        )
        logger.info("✅ VGG19 loaded")

        # Load BiLSTM
        logger.info(f"Loading BiLSTM model: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        logger.info("✅ BiLSTM loaded")

    def add_camera(self, camera_id: int, source: str):
        """
        Add camera to monitoring system

        Args:
            camera_id: Unique camera ID (0-17 for 18 cameras)
            source: Video source (0, 1, 'rtsp://...', '/path/to/video.mp4')
        """
        motion_detector = MotionDetector(threshold=25.0, min_area=500)
        camera = CameraStream(camera_id, source, motion_detector)

        if camera.start():
            self.cameras[camera_id] = camera
            self.motion_detectors[camera_id] = motion_detector
            self.stats['active_cameras'] += 1
            logger.info(f"✅ Camera {camera_id} added and started")
        else:
            logger.error(f"❌ Failed to add camera {camera_id}")

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract VGG19 features from frame"""
        # Resize and preprocess
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        batch = np.expand_dims(normalized, axis=0)

        # Extract features
        features = self.feature_extractor.predict(batch, verbose=0)
        return features[0]

    def predict_violence(self, camera_id: int) -> tuple:
        """
        Predict violence from camera's frame buffer

        Args:
            camera_id: Camera to analyze

        Returns:
            (is_violence, confidence)
        """
        buffer = self.frame_buffers[camera_id]

        if len(buffer) < 20:
            return False, 0.0

        # Stack features
        feature_sequence = np.array(list(buffer))
        batch = np.expand_dims(feature_sequence, axis=0)

        # Predict
        prediction = self.model.predict(batch, verbose=0)[0]
        violence_confidence = prediction[1]
        is_violence = violence_confidence >= self.confidence_threshold

        return is_violence, violence_confidence

    def process_camera(self, camera_id: int):
        """Process single camera (extract features and predict)"""
        camera = self.cameras.get(camera_id)
        if not camera or not camera.last_frame is not None:
            return

        # Extract features
        features = self.extract_features(camera.last_frame)
        self.frame_buffers[camera_id].append(features)

        # Predict if buffer full
        if len(self.frame_buffers[camera_id]) >= 20:
            is_violence, confidence = self.predict_violence(camera_id)

            if is_violence:
                camera.violence_detections += 1
                self.stats['total_detections'] += 1
                logger.warning(
                    f"⚠️ VIOLENCE DETECTED on Camera {camera_id}! "
                    f"Confidence: {confidence*100:.1f}%"
                )

    def monitoring_loop(self):
        """Main monitoring loop for all cameras"""
        logger.info("=" * 80)
        logger.info(f"🚀 MONITORING {self.num_cameras} CAMERAS")
        logger.info(f"Max concurrent analysis: {self.max_concurrent_analysis}")
        logger.info("=" * 80 + "\n")

        frame_times = {i: [] for i in range(self.num_cameras)}

        try:
            while True:
                start_time = time.time()

                # Phase 1: Read all cameras and detect motion (lightweight)
                cameras_with_motion = []

                for camera_id, camera in self.cameras.items():
                    ret, frame, has_motion, motion_score = camera.read_frame()

                    if ret and has_motion:
                        cameras_with_motion.append((motion_score, camera_id))

                self.stats['cameras_with_motion'] = len(cameras_with_motion)

                # Phase 2: Process cameras with motion (priority queue)
                # Sort by motion score (highest first)
                cameras_with_motion.sort(reverse=True)

                # Process top N cameras with most motion
                for motion_score, camera_id in cameras_with_motion[:self.max_concurrent_analysis]:
                    self.process_camera(camera_id)

                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0

                # Log statistics every 30 frames
                if sum(cam.total_frames for cam in self.cameras.values()) % 30 == 0:
                    logger.info(
                        f"📊 Stats: "
                        f"Active: {self.stats['active_cameras']}/{self.num_cameras} | "
                        f"Motion: {self.stats['cameras_with_motion']} | "
                        f"Detections: {self.stats['total_detections']} | "
                        f"FPS: {fps:.1f}"
                    )

                # Small delay to prevent CPU overload
                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down monitoring system...")
            self.stop_all()

    def stop_all(self):
        """Stop all camera streams"""
        for camera in self.cameras.values():
            camera.stop()

        logger.info("✅ All cameras stopped")
        logger.info(f"📊 Final statistics:")
        logger.info(f"   Total detections: {self.stats['total_detections']}")
        for camera_id, camera in self.cameras.items():
            logger.info(
                f"   Camera {camera_id}: "
                f"{camera.violence_detections} detections, "
                f"{camera.motion_frames}/{camera.total_frames} motion frames "
                f"({camera.motion_frames/max(camera.total_frames, 1)*100:.1f}%)"
            )


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Camera Violence Detection System'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.h5 or .keras)'
    )

    parser.add_argument(
        '--cameras',
        type=int,
        default=18,
        help='Number of cameras to monitor (default: 18)'
    )

    parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        help='Camera sources (indices, URLs, or file paths)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Violence detection threshold (default: 0.7)'
    )

    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Max cameras analyzed simultaneously (default: 5)'
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for inference'
    )

    args = parser.parse_args()

    # Validate model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return

    # Initialize detector
    detector = MultiCameraViolenceDetector(
        model_path=args.model,
        num_cameras=args.cameras,
        confidence_threshold=args.threshold,
        max_concurrent_analysis=args.max_concurrent,
        use_gpu=args.gpu
    )

    # Add cameras
    if args.sources:
        # Use provided sources
        for i, source in enumerate(args.sources[:args.cameras]):
            detector.add_camera(i, source)
    else:
        # Use camera indices 0, 1, 2, ...
        logger.warning("No sources provided, using camera indices 0-17")
        for i in range(args.cameras):
            detector.add_camera(i, i)

    # Start monitoring
    detector.monitoring_loop()


if __name__ == "__main__":
    main()
