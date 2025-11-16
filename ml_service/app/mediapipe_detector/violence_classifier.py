"""
Skeleton-based violence classification.

Uses pose features to classify violent actions.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

from .skeleton_detector import SkeletonDetector
from .pose_features import PoseFeatureExtractor

logger = logging.getLogger(__name__)


class SkeletonViolenceClassifier:
    """
    Violence classifier using skeleton-based features.

    Complements CNN-based detection with pose-based analysis.
    """

    def __init__(self):
        """Initialize skeleton-based violence classifier."""
        self.skeleton_detector = SkeletonDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.feature_extractor = PoseFeatureExtractor()

        # Rule-based thresholds (can be replaced with trained model)
        self.violence_threshold = 0.6
        self.punching_threshold = 0.7
        self.kicking_threshold = 0.6
        self.aggressive_threshold = 0.5

        logger.info("Skeleton-based violence classifier initialized")

    def predict(self, frame: np.ndarray) -> Dict:
        """
        Predict violence from single frame.

        Args:
            frame: Video frame (BGR)

        Returns:
            Dictionary with violence prediction and confidence
        """
        # Detect pose
        pose_data = self.skeleton_detector.detect_pose(frame)

        if pose_data is None:
            return {
                'violence_probability': 0.0,
                'confidence': 'Low',
                'prediction': 'non_violence',
                'pose_detected': False,
                'patterns': {}
            }

        # Extract features
        features = self.feature_extractor.extract_features(pose_data)

        # Detect violent patterns
        patterns = self.feature_extractor.detect_violent_patterns(features)

        # Classification based on patterns
        violence_prob = patterns['violence_score']

        return {
            'violence_probability': violence_prob,
            'confidence': self._get_confidence(violence_prob),
            'prediction': 'violence' if violence_prob > self.violence_threshold else 'non_violence',
            'pose_detected': True,
            'patterns': patterns,
            'features': features
        }

    def predict_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Predict violence from frame sequence (temporal analysis).

        Args:
            frames: List of video frames

        Returns:
            Dictionary with violence prediction
        """
        # Detect poses in all frames
        pose_sequence = []

        for frame in frames:
            pose_data = self.skeleton_detector.detect_pose(frame)
            if pose_data is not None:
                pose_sequence.append(pose_data)

        if len(pose_sequence) == 0:
            return {
                'violence_probability': 0.0,
                'confidence': 'Low',
                'prediction': 'non_violence',
                'pose_detected': False,
                'temporal_features': {}
            }

        # Extract temporal features
        temporal_features = self.feature_extractor.extract_temporal_features(pose_sequence)

        # Calculate violence score from temporal patterns
        violence_indicators = [
            temporal_features.get('wrist_velocity_max', 0.0) / 0.5,
            temporal_features.get('movement_intensity_max', 0.0) / 0.3,
            temporal_features.get('forward_lean_max', 0.0) / 0.2,
        ]

        violence_prob = np.mean([min(v, 1.0) for v in violence_indicators])

        return {
            'violence_probability': violence_prob,
            'confidence': self._get_confidence(violence_prob),
            'prediction': 'violence' if violence_prob > self.violence_threshold else 'non_violence',
            'pose_detected': True,
            'frames_with_pose': len(pose_sequence),
            'total_frames': len(frames),
            'temporal_features': temporal_features
        }

    def _get_confidence(self, prob: float) -> str:
        """
        Get confidence level from probability.

        Args:
            prob: Violence probability

        Returns:
            Confidence level string
        """
        if prob > 0.9 or prob < 0.1:
            return "High"
        elif prob > 0.7 or prob < 0.3:
            return "Medium"
        else:
            return "Low"

    def visualize_prediction(
        self,
        frame: np.ndarray,
        prediction: Dict
    ) -> np.ndarray:
        """
        Visualize skeleton and violence prediction on frame.

        Args:
            frame: Original frame
            prediction: Prediction dictionary from predict()

        Returns:
            Annotated frame
        """
        import cv2

        annotated = frame.copy()

        # Draw skeleton if detected
        if prediction['pose_detected']:
            pose_data = self.skeleton_detector.detect_pose(frame)
            if pose_data:
                annotated = self.skeleton_detector.visualize_pose(annotated, pose_data)

        # Add violence prediction text
        violence_text = f"Violence: {prediction['violence_probability']:.2f}"
        prediction_text = f"Prediction: {prediction['prediction']}"
        confidence_text = f"Confidence: {prediction['confidence']}"

        # Color based on prediction
        color = (0, 0, 255) if prediction['prediction'] == 'violence' else (0, 255, 0)

        cv2.putText(annotated, violence_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, prediction_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, confidence_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Add pattern scores if available
        if 'patterns' in prediction and prediction['patterns']:
            patterns = prediction['patterns']
            y_offset = 120

            for pattern_name, score in patterns.items():
                if score > 0.3:  # Only show significant patterns
                    pattern_text = f"{pattern_name}: {score:.2f}"
                    cv2.putText(annotated, pattern_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 25

        return annotated

    def close(self):
        """Release resources."""
        self.skeleton_detector.close()
        logger.info("Skeleton violence classifier closed")


class EnsembleViolenceDetector:
    """
    Ensemble detector combining CNN and skeleton-based predictions.

    Weights: 70% CNN (VGG19) + 30% Skeleton for optimal accuracy.
    """

    def __init__(self, cnn_detector, skeleton_weight: float = 0.3):
        """
        Initialize ensemble detector.

        Args:
            cnn_detector: ViolenceDetector instance (VGG19 model)
            skeleton_weight: Weight for skeleton predictions (0-1)
        """
        self.cnn_detector = cnn_detector
        self.skeleton_classifier = SkeletonViolenceClassifier()
        self.skeleton_weight = skeleton_weight
        self.cnn_weight = 1.0 - skeleton_weight

        logger.info(f"Ensemble detector initialized (CNN:{self.cnn_weight:.0%}, Skeleton:{self.skeleton_weight:.0%})")

    def predict(self, frames: np.ndarray) -> Dict:
        """
        Ensemble prediction from frame sequence.

        Args:
            frames: Numpy array of shape (20, 224, 224, 3) for CNN

        Returns:
            Combined prediction dictionary
        """
        # CNN prediction
        cnn_result = self.cnn_detector.predict(frames)
        cnn_prob = cnn_result['violence_probability']

        # Skeleton prediction (use middle frame for single-frame analysis)
        middle_frame_idx = len(frames) // 2
        middle_frame = frames[middle_frame_idx]

        # Convert normalized frame back to 0-255 BGR for MediaPipe
        skeleton_frame = (middle_frame * 255).astype(np.uint8)

        skeleton_result = self.skeleton_classifier.predict(skeleton_frame)
        skeleton_prob = skeleton_result['violence_probability']

        # Weighted ensemble
        ensemble_prob = (
            cnn_prob * self.cnn_weight +
            skeleton_prob * self.skeleton_weight
        )

        return {
            'violence_probability': ensemble_prob,
            'confidence': self._get_confidence(ensemble_prob),
            'prediction': 'violence' if ensemble_prob > 0.5 else 'non_violence',
            'cnn_probability': cnn_prob,
            'skeleton_probability': skeleton_prob,
            'pose_detected': skeleton_result['pose_detected'],
            'per_class_scores': {
                'non_violence': 1.0 - ensemble_prob,
                'violence': ensemble_prob,
            },
            'details': {
                'cnn': cnn_result,
                'skeleton': skeleton_result
            }
        }

    def predict_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Ensemble prediction from frame sequence with temporal analysis.

        Args:
            frames: List of BGR frames

        Returns:
            Combined prediction dictionary
        """
        # CNN prediction (needs 20 normalized frames)
        if len(frames) >= 20:
            # Sample 20 frames evenly
            indices = np.linspace(0, len(frames) - 1, 20, dtype=int)
            sampled_frames = [frames[i] for i in indices]

            # Prepare for CNN (resize and normalize)
            import cv2
            cnn_frames = []
            for frame in sampled_frames:
                resized = cv2.resize(frame, (224, 224))
                normalized = resized.astype(np.float32) / 255.0
                cnn_frames.append(normalized)

            cnn_frames = np.array(cnn_frames)
            cnn_result = self.cnn_detector.predict(cnn_frames)
            cnn_prob = cnn_result['violence_probability']
        else:
            # Not enough frames for CNN
            cnn_prob = 0.5
            cnn_result = {'violence_probability': 0.5}

        # Skeleton temporal prediction
        skeleton_result = self.skeleton_classifier.predict_sequence(frames)
        skeleton_prob = skeleton_result['violence_probability']

        # Weighted ensemble
        ensemble_prob = (
            cnn_prob * self.cnn_weight +
            skeleton_prob * self.skeleton_weight
        )

        return {
            'violence_probability': ensemble_prob,
            'confidence': self._get_confidence(ensemble_prob),
            'prediction': 'violence' if ensemble_prob > 0.5 else 'non_violence',
            'cnn_probability': cnn_prob,
            'skeleton_probability': skeleton_prob,
            'pose_detected': skeleton_result['pose_detected'],
            'per_class_scores': {
                'non_violence': 1.0 - ensemble_prob,
                'violence': ensemble_prob,
            },
            'details': {
                'cnn': cnn_result,
                'skeleton': skeleton_result
            }
        }

    def _get_confidence(self, prob: float) -> str:
        """Get confidence level from probability."""
        if prob > 0.9 or prob < 0.1:
            return "High"
        elif prob > 0.7 or prob < 0.3:
            return "Medium"
        else:
            return "Low"

    def close(self):
        """Release resources."""
        self.skeleton_classifier.close()
