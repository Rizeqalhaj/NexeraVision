#!/usr/bin/env python3
"""
Real-Time Violence Detection MVP
Detects fighting/violence in live video streams using trained BiLSTM model

Features:
- Live webcam or video file input
- Rolling buffer of 20 frames
- VGG19 + BiLSTM inference
- Visual alerts and logging
- Configurable detection threshold
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
from datetime import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeViolenceDetector:
    """Real-time violence detection system"""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.7,
        frame_size: tuple = (224, 224),
        buffer_size: int = 20,
        use_gpu: bool = True
    ):
        """
        Initialize detector

        Args:
            model_path: Path to trained BiLSTM model (.h5 or .keras)
            confidence_threshold: Minimum confidence for violence alert (0-1)
            frame_size: Input frame size for VGG19 (height, width)
            buffer_size: Number of frames to analyze (should match training)
            use_gpu: Whether to use GPU acceleration
        """
        self.confidence_threshold = confidence_threshold
        self.frame_size = frame_size
        self.buffer_size = buffer_size

        # Configure GPU or CPU
        if use_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Use GPU:1 if available (GPU:0 for desktop)
                    if len(gpus) > 1:
                        tf.config.set_visible_devices(gpus[1], 'GPU')
                        logger.info(f"Using GPU:1 for inference")
                    else:
                        tf.config.set_visible_devices(gpus[0], 'GPU')
                        logger.info(f"Using GPU:0 for inference")

                    # Enable memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"GPU configuration error: {e}")
        else:
            # CPU-only mode with optimizations
            import os

            # Use all available CPU threads (for i9-14900K: 10+ threads)
            tf.config.threading.set_intra_op_parallelism_threads(0)  # Auto-detect
            tf.config.threading.set_inter_op_parallelism_threads(0)  # Auto-detect

            # Enable oneDNN optimizations for Intel CPUs
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

            logger.info("Using CPU with Intel optimizations (oneDNN)")

        # Load VGG19 feature extractor
        logger.info("Loading VGG19 feature extractor...")
        base_model = tf.keras.applications.VGG19(
            include_top=True,
            weights='imagenet',
            input_shape=(*frame_size, 3)
        )
        base_model.trainable = False

        # Extract fc2 layer (4096 features)
        self.feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('fc2').output
        )
        logger.info("✅ VGG19 loaded")

        # Load trained BiLSTM model
        logger.info(f"Loading trained model: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("✅ BiLSTM model loaded")

        # Frame buffer (rolling window)
        self.frame_buffer = deque(maxlen=buffer_size)

        # Detection statistics
        self.frame_count = 0
        self.violence_detected_count = 0
        self.last_detection_time = None

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for VGG19

        Args:
            frame: Raw BGR frame from OpenCV

        Returns:
            Preprocessed frame ready for VGG19
        """
        # Resize to target size
        resized = cv2.resize(frame, self.frame_size)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        return normalized

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract VGG19 features from single frame

        Args:
            frame: Preprocessed frame

        Returns:
            Feature vector (4096,)
        """
        # Add batch dimension
        batch = np.expand_dims(frame, axis=0)

        # Extract features
        features = self.feature_extractor.predict(batch, verbose=0)

        return features[0]  # Remove batch dimension

    def predict_violence(self) -> tuple:
        """
        Predict violence from current frame buffer

        Returns:
            (is_violence, confidence) tuple
        """
        if len(self.frame_buffer) < self.buffer_size:
            return False, 0.0

        # Stack features into sequence
        feature_sequence = np.array(list(self.frame_buffer))

        # Add batch dimension: (20, 4096) -> (1, 20, 4096)
        batch = np.expand_dims(feature_sequence, axis=0)

        # Predict
        prediction = self.model.predict(batch, verbose=0)[0]

        # prediction = [non_violence_prob, violence_prob]
        violence_confidence = prediction[1]
        is_violence = violence_confidence >= self.confidence_threshold

        return is_violence, violence_confidence

    def draw_overlay(
        self,
        frame: np.ndarray,
        is_violence: bool,
        confidence: float
    ) -> np.ndarray:
        """
        Draw detection overlay on frame

        Args:
            frame: Original frame
            is_violence: Whether violence detected
            confidence: Detection confidence

        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Status bar at top
        status_color = (0, 0, 255) if is_violence else (0, 255, 0)
        status_text = "⚠️ VIOLENCE DETECTED" if is_violence else "✅ Normal"

        cv2.rectangle(overlay, (0, 0), (width, 60), status_color, -1)
        cv2.putText(
            overlay,
            status_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        # Confidence meter
        conf_text = f"Confidence: {confidence*100:.1f}%"
        cv2.putText(
            overlay,
            conf_text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Buffer status
        buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}"
        cv2.putText(
            overlay,
            buffer_text,
            (width - 250, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Statistics
        stats_text = f"Detections: {self.violence_detected_count}"
        cv2.putText(
            overlay,
            stats_text,
            (10, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return overlay

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process single frame

        Args:
            frame: Raw frame from video source

        Returns:
            (annotated_frame, is_violence, confidence)
        """
        self.frame_count += 1

        # Preprocess and extract features
        preprocessed = self.preprocess_frame(frame)
        features = self.extract_features(preprocessed)

        # Add to buffer
        self.frame_buffer.append(features)

        # Predict if buffer is full
        is_violence, confidence = self.predict_violence()

        # Update statistics
        if is_violence:
            self.violence_detected_count += 1
            self.last_detection_time = datetime.now()
            logger.warning(
                f"⚠️ VIOLENCE DETECTED! Confidence: {confidence*100:.1f}%"
            )

        # Draw overlay
        annotated = self.draw_overlay(frame, is_violence, confidence)

        return annotated, is_violence, confidence

    def run_webcam(self, camera_id: int = 0):
        """
        Run detection on webcam feed

        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        logger.info(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return

        logger.info("✅ Camera opened successfully")
        logger.info("Press 'q' to quit")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Process frame
                annotated, is_violence, confidence = self.process_frame(frame)

                # Display
                cv2.imshow('Violence Detection - Press Q to quit', annotated)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released")

    def run_video_file(self, video_path: str, output_path: str = None):
        """
        Run detection on video file

        Args:
            video_path: Input video file path
            output_path: Optional output video path (annotated)
        """
        logger.info(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Output will be saved to: {output_path}")

        try:
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                # Process frame
                annotated, is_violence, confidence = self.process_frame(frame)

                # Write to output
                if writer:
                    writer.write(annotated)

                # Display progress
                if frame_num % 30 == 0:
                    logger.info(
                        f"Progress: {frame_num}/{total_frames} "
                        f"({frame_num/total_frames*100:.1f}%)"
                    )

                # Display (optional)
                cv2.imshow('Processing - Press Q to quit', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            logger.info(f"✅ Processing complete")
            logger.info(f"Total frames: {frame_num}")
            logger.info(f"Violence detections: {self.violence_detected_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Violence Detection System'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.h5 or .keras)'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='webcam',
        help='Video source: "webcam", "0", "1", or video file path'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path (for video file processing)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Violence detection threshold (0-1), default: 0.7'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )

    args = parser.parse_args()

    # Validate model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.error("Train a model first using train_rtx5000_dual_IMPROVED.py")
        return

    # Initialize detector
    logger.info("=" * 80)
    logger.info("🚀 REAL-TIME VIOLENCE DETECTION SYSTEM")
    logger.info("=" * 80)

    detector = RealtimeViolenceDetector(
        model_path=args.model,
        confidence_threshold=args.threshold,
        use_gpu=not args.no_gpu
    )

    logger.info(f"Detection threshold: {args.threshold*100:.1f}%")
    logger.info("=" * 80 + "\n")

    # Run detection based on source
    if args.source in ['webcam', '0', '1', '2']:
        camera_id = 0 if args.source == 'webcam' else int(args.source)
        detector.run_webcam(camera_id=camera_id)
    else:
        # Video file
        detector.run_video_file(args.source, args.output)


if __name__ == "__main__":
    main()
