"""
Grid detection using Hough transform and DBSCAN clustering.

Automatically detects grid layouts from 2x2 to 10x10 for multi-camera
surveillance screen recordings.
"""
import logging
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class GridDetector:
    """
    Detects and analyzes grid structures in multi-camera screen recordings.

    Uses enhanced Hough transform for line detection and DBSCAN for clustering
    to identify grid layout automatically.
    """

    def __init__(self, frame: np.ndarray):
        """
        Initialize grid detector with video frame.

        Args:
            frame: Video frame as numpy array (H, W, 3)
        """
        self.frame = frame
        self.height, self.width = frame.shape[:2]
        self.grid_layout: Optional[Dict] = None

    def detect_grid_lines(self) -> Tuple[List, List]:
        """
        Detect horizontal and vertical lines using Hough transform.

        Returns:
            Tuple of (horizontal_lines, vertical_lines)
        """
        # 1. Preprocessing
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive edge detection using Canny
        edges = cv2.Canny(filtered, 50, 150, apertureSize=3)

        # 2. Line detection with Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=int(min(self.width, self.height) * 0.1),
            maxLineGap=10
        )

        if lines is None:
            logger.warning("No lines detected in frame")
            return [], []

        # 3. Filter for horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            if x2 - x1 == 0:
                angle = 90.0
            else:
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Classify as horizontal or vertical
            if angle < 5 or angle > 175:  # Horizontal (±5 degrees tolerance)
                horizontal_lines.append(line[0])
            elif 85 < angle < 95:  # Vertical (±5 degrees tolerance)
                vertical_lines.append(line[0])

        logger.info(f"Detected {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")
        return horizontal_lines, vertical_lines

    def cluster_lines(self, lines: List, is_horizontal: bool = True) -> List[float]:
        """
        Cluster lines to find grid structure using DBSCAN.

        Args:
            lines: List of lines [x1, y1, x2, y2]
            is_horizontal: True for horizontal lines, False for vertical

        Returns:
            Sorted list of cluster center positions
        """
        if not lines:
            return []

        # Extract position (y for horizontal, x for vertical)
        positions = []
        for line in lines:
            x1, y1, x2, y2 = line
            pos = (y1 + y2) / 2 if is_horizontal else (x1 + x2) / 2
            positions.append([pos])

        # Cluster using DBSCAN
        positions = np.array(positions)
        clustering = DBSCAN(eps=10, min_samples=2).fit(positions)

        # Get cluster centers
        centers = []
        for label in set(clustering.labels_):
            if label == -1:  # Skip noise
                continue
            mask = clustering.labels_ == label
            center = positions[mask].mean()
            centers.append(float(center))

        return sorted(centers)

    def detect_grid_layout(self) -> Dict:
        """
        Detect complete grid layout with rows, columns, and line positions.

        Returns:
            Dictionary with grid metadata:
                - rows: Number of grid rows
                - cols: Number of grid columns
                - h_lines: Horizontal line positions
                - v_lines: Vertical line positions
        """
        h_lines, v_lines = self.detect_grid_lines()

        # Cluster lines to find grid structure
        h_positions = self.cluster_lines(h_lines, is_horizontal=True)
        v_positions = self.cluster_lines(v_lines, is_horizontal=False)

        # Add frame boundaries
        h_positions = [0.0] + h_positions + [float(self.height)]
        v_positions = [0.0] + v_positions + [float(self.width)]

        # Calculate grid dimensions
        rows = len(h_positions) - 1
        cols = len(v_positions) - 1

        self.grid_layout = {
            'rows': rows,
            'cols': cols,
            'h_lines': h_positions,
            'v_lines': v_positions,
            'total_cameras': rows * cols,
        }

        logger.info(f"Detected grid layout: {rows}x{cols} = {rows * cols} cameras")
        return self.grid_layout

    def extract_camera_regions(self, grid_layout: Optional[Dict] = None) -> List[Dict]:
        """
        Extract individual camera regions from grid.

        Args:
            grid_layout: Optional pre-computed grid layout. If None, auto-detects.

        Returns:
            List of camera info dictionaries containing:
                - id: Camera index
                - position: (row, col) tuple
                - bbox: (x1, y1, x2, y2) bounding box
                - frame: Extracted camera frame
                - resolution: (width, height) of camera region
        """
        if grid_layout is None:
            grid_layout = self.detect_grid_layout()

        cameras = []

        for row in range(grid_layout['rows']):
            for col in range(grid_layout['cols']):
                # Get boundaries
                y1 = int(grid_layout['h_lines'][row])
                y2 = int(grid_layout['h_lines'][row + 1])
                x1 = int(grid_layout['v_lines'][col])
                x2 = int(grid_layout['v_lines'][col + 1])

                # Extract region (add small padding to avoid grid lines)
                padding = 2
                y1_padded = min(y1 + padding, self.height)
                y2_padded = max(y2 - padding, 0)
                x1_padded = min(x1 + padding, self.width)
                x2_padded = max(x2 - padding, 0)

                # Validate bounds
                if y2_padded <= y1_padded or x2_padded <= x1_padded:
                    logger.warning(f"Invalid camera bounds at ({row}, {col}), skipping")
                    continue

                region = self.frame[y1_padded:y2_padded, x1_padded:x2_padded]

                # Store camera info
                camera_info = {
                    'id': row * grid_layout['cols'] + col,
                    'position': (row, col),
                    'bbox': (x1, y1, x2, y2),
                    'frame': region,
                    'resolution': (x2 - x1, y2 - y1),
                    'is_active': self._is_camera_active(region),
                }
                cameras.append(camera_info)

        logger.info(f"Extracted {len(cameras)} camera regions")
        return cameras

    def _is_camera_active(self, camera_frame: np.ndarray, threshold: float = 10.0) -> bool:
        """
        Check if camera feed is active (not black screen).

        Args:
            camera_frame: Camera region frame
            threshold: Mean pixel value threshold for black detection

        Returns:
            True if camera is active, False if black screen
        """
        if camera_frame.size == 0:
            return False

        mean_val = np.mean(camera_frame)
        return mean_val > threshold

    def visualize_grid(self, output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected grid with overlay.

        Args:
            output_path: Optional path to save visualization

        Returns:
            Frame with grid overlay
        """
        if self.grid_layout is None:
            self.detect_grid_layout()

        # Create copy for visualization
        vis_frame = self.frame.copy()

        # Draw horizontal lines
        for y in self.grid_layout['h_lines']:
            cv2.line(vis_frame, (0, int(y)), (self.width, int(y)), (0, 255, 0), 2)

        # Draw vertical lines
        for x in self.grid_layout['v_lines']:
            cv2.line(vis_frame, (int(x), 0), (int(x), self.height), (0, 255, 0), 2)

        # Add grid info text
        info_text = f"Grid: {self.grid_layout['rows']}x{self.grid_layout['cols']} ({self.grid_layout['total_cameras']} cameras)"
        cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, vis_frame)
            logger.info(f"Saved grid visualization to {output_path}")

        return vis_frame


class AutoCalibrator:
    """
    Automatic grid layout calibration for known surveillance layouts.
    """

    def __init__(self):
        """Initialize with common grid layouts."""
        self.known_layouts = [
            (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
            (2, 3), (3, 4), (4, 5), (5, 6),
            (8, 8), (10, 10), (12, 12)  # Common surveillance layouts
        ]

    def detect_layout(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Detect grid layout with validation against known layouts.

        Args:
            frame: Video frame

        Returns:
            Tuple of (rows, cols)
        """
        detector = GridDetector(frame)
        detected = detector.detect_grid_layout()

        # Validate against known layouts
        detected_layout = (detected['rows'], detected['cols'])

        # If detected layout matches a known layout, use it
        if detected_layout in self.known_layouts:
            logger.info(f"Matched known layout: {detected_layout[0]}x{detected_layout[1]}")
            return detected_layout

        # Otherwise, return closest match or detected layout
        logger.info(f"Using detected layout: {detected_layout[0]}x{detected_layout[1]}")
        return detected_layout

    def handle_dynamic_layout(self, frame_sequence: List[np.ndarray]) -> Tuple[int, int]:
        """
        Handle changing grid layouts by analyzing multiple frames.

        Args:
            frame_sequence: List of video frames

        Returns:
            Most common grid layout tuple (rows, cols)
        """
        from collections import Counter

        layouts = []
        sample_size = min(30, len(frame_sequence))  # Sample first second (at 30 FPS)

        for frame in frame_sequence[:sample_size]:
            layout = self.detect_layout(frame)
            layouts.append(layout)

        # Most common layout
        most_common = Counter(layouts).most_common(1)[0][0]
        logger.info(f"Most common layout across {sample_size} frames: {most_common[0]}x{most_common[1]}")

        return most_common
