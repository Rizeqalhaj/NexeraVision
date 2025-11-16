"""
Skeleton detection using MediaPipe Pose estimation.

Extracts 33-point body landmarks for violence detection.
"""
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class SkeletonDetector:
    """
    Real-time pose estimation using MediaPipe.

    Detects 33 body landmarks including face, torso, arms, and legs
    for skeleton-based violence detection.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Initialize MediaPipe Pose detector.

        Args:
            min_detection_confidence: Minimum confidence for person detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info(f"MediaPipe Pose initialized (complexity={model_complexity})")

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect pose landmarks in a single frame.

        Args:
            frame: BGR image frame

        Returns:
            Dictionary with pose landmarks and metadata, or None if no person detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })

        # Get frame dimensions for absolute coordinates
        height, width = frame.shape[:2]

        return {
            'landmarks': landmarks,
            'landmarks_absolute': self._to_absolute_coords(landmarks, width, height),
            'visibility_scores': [lm['visibility'] for lm in landmarks],
            'mean_visibility': np.mean([lm['visibility'] for lm in landmarks]),
            'frame_shape': (height, width)
        }

    def detect_poses_batch(self, frames: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        Detect poses in multiple frames (batch processing).

        Args:
            frames: List of BGR image frames

        Returns:
            List of pose dictionaries (None for frames with no detection)
        """
        results = []

        for frame in frames:
            pose_data = self.detect_pose(frame)
            results.append(pose_data)

        return results

    def _to_absolute_coords(
        self,
        landmarks: List[Dict],
        width: int,
        height: int
    ) -> List[Tuple[int, int]]:
        """
        Convert normalized landmarks to absolute pixel coordinates.

        Args:
            landmarks: List of normalized landmark dictionaries
            width: Frame width
            height: Frame height

        Returns:
            List of (x, y) absolute coordinates
        """
        absolute = []
        for lm in landmarks:
            x = int(lm['x'] * width)
            y = int(lm['y'] * height)
            absolute.append((x, y))

        return absolute

    def visualize_pose(
        self,
        frame: np.ndarray,
        pose_data: Optional[Dict] = None,
        draw_landmarks: bool = True,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Visualize detected pose on frame.

        Args:
            frame: Original frame
            pose_data: Pre-computed pose data (optional)
            draw_landmarks: Draw landmark points
            draw_connections: Draw skeleton connections

        Returns:
            Frame with pose overlay
        """
        if pose_data is None:
            pose_data = self.detect_pose(frame)

        if pose_data is None:
            return frame

        annotated_frame = frame.copy()

        # Convert landmarks back to MediaPipe format for drawing
        landmark_list = self.mp_pose.PoseLandmark
        pose_landmarks = self.mp_pose.PoseLandmark

        # Draw landmarks
        if draw_landmarks or draw_connections:
            # Reconstruct MediaPipe landmarks object
            from mediapipe.framework.formats import landmark_pb2

            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

            for lm in pose_data['landmarks']:
                landmark = pose_landmarks_proto.landmark.add()
                landmark.x = lm['x']
                landmark.y = lm['y']
                landmark.z = lm['z']
                landmark.visibility = lm['visibility']

            # Draw using MediaPipe utilities
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_landmarks_proto,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return annotated_frame

    def get_landmark_by_name(self, pose_data: Dict, landmark_name: str) -> Optional[Dict]:
        """
        Get specific landmark by body part name.

        Args:
            pose_data: Pose detection result
            landmark_name: Name from MediaPipe PoseLandmark enum
                          (e.g., 'NOSE', 'LEFT_SHOULDER', 'RIGHT_WRIST')

        Returns:
            Landmark dictionary or None
        """
        try:
            landmark_idx = self.mp_pose.PoseLandmark[landmark_name].value
            return pose_data['landmarks'][landmark_idx]
        except (KeyError, IndexError):
            return None

    def is_pose_visible(self, pose_data: Dict, visibility_threshold: float = 0.5) -> bool:
        """
        Check if pose is sufficiently visible for analysis.

        Args:
            pose_data: Pose detection result
            visibility_threshold: Minimum mean visibility score

        Returns:
            True if pose is visible enough for analysis
        """
        if pose_data is None:
            return False

        return pose_data['mean_visibility'] >= visibility_threshold

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
            logger.info("MediaPipe Pose detector closed")


class MultiPersonSkeletonDetector:
    """
    Detect multiple people in frame using YOLOv8 + MediaPipe.

    Note: This is a simplified version. For production, integrate YOLOv8
    for person detection, then run MediaPipe on each detected person.
    """

    def __init__(self):
        """Initialize multi-person detector."""
        self.single_detector = SkeletonDetector()
        logger.info("Multi-person skeleton detector initialized (simplified version)")

    def detect_multiple_poses(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect multiple people in frame.

        Note: Current implementation detects single person only.
        TODO: Integrate YOLOv8 for multi-person detection.

        Args:
            frame: Input frame

        Returns:
            List of pose dictionaries (one per person)
        """
        # Simplified: detect single person
        pose_data = self.single_detector.detect_pose(frame)

        if pose_data is None:
            return []

        return [pose_data]

    def close(self):
        """Release resources."""
        self.single_detector.close()
