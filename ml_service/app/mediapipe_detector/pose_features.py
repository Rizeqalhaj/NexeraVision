"""
Pose feature extraction for violence detection.

Extracts meaningful features from skeleton keypoints for violence classification.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PoseFeatureExtractor:
    """
    Extract violence-relevant features from pose landmarks.

    Features include joint angles, movement velocity, body pose patterns,
    and inter-person distances.
    """

    # MediaPipe landmark indices
    LANDMARKS = {
        'nose': 0,
        'left_eye': 2,
        'right_eye': 5,
        'left_ear': 7,
        'right_ear': 8,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }

    def __init__(self):
        """Initialize feature extractor."""
        self.prev_poses: List[Dict] = []
        self.max_history = 10  # Keep last 10 poses for velocity calculation

    def extract_features(self, pose_data: Dict) -> Dict[str, float]:
        """
        Extract all violence-relevant features from pose.

        Args:
            pose_data: Pose detection result from SkeletonDetector

        Returns:
            Dictionary of extracted features
        """
        if pose_data is None or len(pose_data['landmarks']) == 0:
            return self._get_empty_features()

        features = {}

        # 1. Joint angles
        features.update(self._extract_joint_angles(pose_data))

        # 2. Body pose patterns
        features.update(self._extract_pose_patterns(pose_data))

        # 3. Movement velocity (if history available)
        features.update(self._extract_movement_features(pose_data))

        # 4. Spatial features
        features.update(self._extract_spatial_features(pose_data))

        # 5. Visibility scores
        features['mean_visibility'] = pose_data['mean_visibility']

        # Update history
        self.prev_poses.append(pose_data)
        if len(self.prev_poses) > self.max_history:
            self.prev_poses.pop(0)

        return features

    def _extract_joint_angles(self, pose_data: Dict) -> Dict[str, float]:
        """
        Calculate joint angles (elbows, knees, etc.).

        Args:
            pose_data: Pose data with landmarks

        Returns:
            Dictionary of joint angles in degrees
        """
        landmarks = pose_data['landmarks']

        angles = {}

        # Left elbow angle
        angles['left_elbow_angle'] = self._calculate_angle(
            landmarks[self.LANDMARKS['left_shoulder']],
            landmarks[self.LANDMARKS['left_elbow']],
            landmarks[self.LANDMARKS['left_wrist']]
        )

        # Right elbow angle
        angles['right_elbow_angle'] = self._calculate_angle(
            landmarks[self.LANDMARKS['right_shoulder']],
            landmarks[self.LANDMARKS['right_elbow']],
            landmarks[self.LANDMARKS['right_wrist']]
        )

        # Left knee angle
        angles['left_knee_angle'] = self._calculate_angle(
            landmarks[self.LANDMARKS['left_hip']],
            landmarks[self.LANDMARKS['left_knee']],
            landmarks[self.LANDMARKS['left_ankle']]
        )

        # Right knee angle
        angles['right_knee_angle'] = self._calculate_angle(
            landmarks[self.LANDMARKS['right_hip']],
            landmarks[self.LANDMARKS['right_knee']],
            landmarks[self.LANDMARKS['right_ankle']]
        )

        # Shoulder spread (indicates arms raised)
        angles['shoulder_spread'] = self._calculate_distance(
            landmarks[self.LANDMARKS['left_shoulder']],
            landmarks[self.LANDMARKS['right_shoulder']]
        )

        return angles

    def _calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """
        Calculate angle between three points.

        Args:
            point1: First landmark (x, y, z)
            point2: Middle landmark (vertex)
            point3: Third landmark

        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        a = np.array([point1['x'], point1['y']])
        b = np.array([point2['x'], point2['y']])
        c = np.array([point3['x'], point3['y']])

        # Calculate vectors
        ba = a - b
        bc = c - b

        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: First landmark
            point2: Second landmark

        Returns:
            Normalized distance
        """
        dx = point1['x'] - point2['x']
        dy = point1['y'] - point2['y']

        return np.sqrt(dx**2 + dy**2)

    def _extract_pose_patterns(self, pose_data: Dict) -> Dict[str, float]:
        """
        Extract pose pattern features (stance, posture).

        Args:
            pose_data: Pose data

        Returns:
            Dictionary of pose pattern features
        """
        landmarks = pose_data['landmarks']
        patterns = {}

        # Body vertical alignment (standing vs crouching)
        nose_y = landmarks[self.LANDMARKS['nose']]['y']
        hip_y = (landmarks[self.LANDMARKS['left_hip']]['y'] +
                 landmarks[self.LANDMARKS['right_hip']]['y']) / 2

        patterns['body_vertical_ratio'] = abs(nose_y - hip_y)

        # Arms raised indicator
        left_wrist_y = landmarks[self.LANDMARKS['left_wrist']]['y']
        right_wrist_y = landmarks[self.LANDMARKS['right_wrist']]['y']
        shoulder_y = (landmarks[self.LANDMARKS['left_shoulder']]['y'] +
                      landmarks[self.LANDMARKS['right_shoulder']]['y']) / 2

        patterns['left_arm_raised'] = 1.0 if left_wrist_y < shoulder_y else 0.0
        patterns['right_arm_raised'] = 1.0 if right_wrist_y < shoulder_y else 0.0

        # Stance width (wider = more aggressive)
        patterns['stance_width'] = self._calculate_distance(
            landmarks[self.LANDMARKS['left_ankle']],
            landmarks[self.LANDMARKS['right_ankle']]
        )

        # Forward lean (aggressive posture)
        nose_x = landmarks[self.LANDMARKS['nose']]['x']
        hip_x = (landmarks[self.LANDMARKS['left_hip']]['x'] +
                 landmarks[self.LANDMARKS['right_hip']]['x']) / 2

        patterns['forward_lean'] = abs(nose_x - hip_x)

        return patterns

    def _extract_movement_features(self, pose_data: Dict) -> Dict[str, float]:
        """
        Extract movement velocity and acceleration features.

        Args:
            pose_data: Current pose data

        Returns:
            Dictionary of movement features
        """
        features = {}

        if len(self.prev_poses) < 2:
            # Not enough history
            features['wrist_velocity'] = 0.0
            features['body_velocity'] = 0.0
            features['movement_intensity'] = 0.0
            return features

        prev_pose = self.prev_poses[-1]
        landmarks = pose_data['landmarks']
        prev_landmarks = prev_pose['landmarks']

        # Wrist velocity (punching/striking indicator)
        left_wrist_vel = self._calculate_distance(
            landmarks[self.LANDMARKS['left_wrist']],
            prev_landmarks[self.LANDMARKS['left_wrist']]
        )

        right_wrist_vel = self._calculate_distance(
            landmarks[self.LANDMARKS['right_wrist']],
            prev_landmarks[self.LANDMARKS['right_wrist']]
        )

        features['wrist_velocity'] = max(left_wrist_vel, right_wrist_vel)

        # Body center velocity
        curr_center = np.array([
            (landmarks[self.LANDMARKS['left_hip']]['x'] +
             landmarks[self.LANDMARKS['right_hip']]['x']) / 2,
            (landmarks[self.LANDMARKS['left_hip']]['y'] +
             landmarks[self.LANDMARKS['right_hip']]['y']) / 2
        ])

        prev_center = np.array([
            (prev_landmarks[self.LANDMARKS['left_hip']]['x'] +
             prev_landmarks[self.LANDMARKS['right_hip']]['x']) / 2,
            (prev_landmarks[self.LANDMARKS['left_hip']]['y'] +
             prev_landmarks[self.LANDMARKS['right_hip']]['y']) / 2
        ])

        features['body_velocity'] = np.linalg.norm(curr_center - prev_center)

        # Overall movement intensity
        features['movement_intensity'] = features['wrist_velocity'] + features['body_velocity']

        return features

    def _extract_spatial_features(self, pose_data: Dict) -> Dict[str, float]:
        """
        Extract spatial arrangement features.

        Args:
            pose_data: Pose data

        Returns:
            Dictionary of spatial features
        """
        landmarks = pose_data['landmarks']
        spatial = {}

        # Bounding box size (person scale)
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]

        spatial['bbox_width'] = max(x_coords) - min(x_coords)
        spatial['bbox_height'] = max(y_coords) - min(y_coords)
        spatial['bbox_area'] = spatial['bbox_width'] * spatial['bbox_height']

        # Body center position
        spatial['center_x'] = np.mean(x_coords)
        spatial['center_y'] = np.mean(y_coords)

        return spatial

    def _get_empty_features(self) -> Dict[str, float]:
        """
        Get empty feature dictionary when no pose detected.

        Returns:
            Dictionary with zero values for all features
        """
        return {
            # Joint angles
            'left_elbow_angle': 0.0,
            'right_elbow_angle': 0.0,
            'left_knee_angle': 0.0,
            'right_knee_angle': 0.0,
            'shoulder_spread': 0.0,

            # Pose patterns
            'body_vertical_ratio': 0.0,
            'left_arm_raised': 0.0,
            'right_arm_raised': 0.0,
            'stance_width': 0.0,
            'forward_lean': 0.0,

            # Movement
            'wrist_velocity': 0.0,
            'body_velocity': 0.0,
            'movement_intensity': 0.0,

            # Spatial
            'bbox_width': 0.0,
            'bbox_height': 0.0,
            'bbox_area': 0.0,
            'center_x': 0.0,
            'center_y': 0.0,

            # Visibility
            'mean_visibility': 0.0,
        }

    def extract_temporal_features(self, pose_sequence: List[Dict]) -> Dict[str, float]:
        """
        Extract temporal features from sequence of poses.

        Args:
            pose_sequence: List of pose data dictionaries

        Returns:
            Dictionary of temporal features
        """
        if len(pose_sequence) == 0:
            return {}

        temporal = {}

        # Extract features for each pose
        all_features = [self.extract_features(pose) for pose in pose_sequence]

        # Calculate statistics over time
        feature_keys = all_features[0].keys()

        for key in feature_keys:
            values = [f[key] for f in all_features]

            temporal[f'{key}_mean'] = np.mean(values)
            temporal[f'{key}_std'] = np.std(values)
            temporal[f'{key}_max'] = np.max(values)
            temporal[f'{key}_min'] = np.min(values)

        return temporal

    def detect_violent_patterns(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Detect violence-indicative patterns from features.

        Args:
            features: Extracted pose features

        Returns:
            Dictionary of violence indicators (0-1 scores)
        """
        patterns = {}

        # Punching pattern: High wrist velocity + raised arm
        punching_score = (
            min(features['wrist_velocity'] / 0.5, 1.0) * 0.6 +
            features['left_arm_raised'] * 0.2 +
            features['right_arm_raised'] * 0.2
        )
        patterns['punching_likelihood'] = punching_score

        # Kicking pattern: Raised leg + body lean
        kicking_score = (
            min(features['movement_intensity'] / 0.3, 1.0) * 0.5 +
            min(features['forward_lean'] / 0.2, 1.0) * 0.5
        )
        patterns['kicking_likelihood'] = kicking_score

        # Aggressive stance: Wide stance + forward lean
        aggressive_stance = (
            min(features['stance_width'] / 0.5, 1.0) * 0.5 +
            min(features['forward_lean'] / 0.2, 1.0) * 0.5
        )
        patterns['aggressive_stance_likelihood'] = aggressive_stance

        # Falling/knocked down: Low vertical ratio
        falling_score = 1.0 - min(features['body_vertical_ratio'] / 0.5, 1.0)
        patterns['falling_likelihood'] = falling_score

        # Overall violence indicator
        patterns['violence_score'] = np.mean([
            punching_score,
            kicking_score,
            aggressive_stance,
            falling_score
        ])

        return patterns
