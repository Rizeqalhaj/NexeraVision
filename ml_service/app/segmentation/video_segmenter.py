"""
Video segmentation for splitting multi-camera grid recordings into individual feeds.

Handles real-time video stream segmentation with temporal consistency.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Generator

import cv2
import numpy as np

from .grid_detector import GridDetector, AutoCalibrator

logger = logging.getLogger(__name__)


class VideoSegmenter:
    """
    Segments multi-camera grid video streams into individual camera feeds.

    Maintains temporal consistency and handles dynamic grid layouts.
    """

    def __init__(self, calibrate_frames: int = 5):
        """
        Initialize video segmenter.

        Args:
            calibrate_frames: Number of frames to use for grid calibration
        """
        self.calibrate_frames = calibrate_frames
        self.grid_layout: Optional[Dict] = None
        self.calibrator = AutoCalibrator()
        self.is_calibrated = False

    def calibrate(self, video_source) -> Dict:
        """
        Calibrate grid layout from video source.

        Args:
            video_source: Path to video file or cv2.VideoCapture object

        Returns:
            Grid layout dictionary
        """
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = video_source

        calibration_frames = []

        logger.info(f"Collecting {self.calibrate_frames} frames for calibration...")

        # Collect frames for calibration
        for i in range(self.calibrate_frames):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {i} for calibration")
                break
            calibration_frames.append(frame)

        if not calibration_frames:
            raise ValueError("No frames available for calibration")

        # Detect layout from calibration frames
        if len(calibration_frames) > 1:
            rows, cols = self.calibrator.handle_dynamic_layout(calibration_frames)
        else:
            rows, cols = self.calibrator.detect_layout(calibration_frames[0])

        # Use first frame to get exact grid positions
        detector = GridDetector(calibration_frames[0])
        self.grid_layout = detector.detect_grid_layout()

        # Validate detected layout matches calibrated layout
        if self.grid_layout['rows'] != rows or self.grid_layout['cols'] != cols:
            logger.warning(
                f"Layout mismatch: detected {self.grid_layout['rows']}x{self.grid_layout['cols']}, "
                f"calibrated {rows}x{cols}. Using calibrated layout."
            )
            # Force the calibrated layout
            self.grid_layout['rows'] = rows
            self.grid_layout['cols'] = cols
            self.grid_layout['total_cameras'] = rows * cols

        self.is_calibrated = True
        logger.info(
            f"Calibration complete: {self.grid_layout['rows']}x{self.grid_layout['cols']} "
            f"= {self.grid_layout['total_cameras']} cameras"
        )

        # Release if we opened it
        if isinstance(video_source, str):
            cap.release()

        return self.grid_layout

    def segment_frame(self, frame: np.ndarray, use_cached_layout: bool = True) -> List[Dict]:
        """
        Segment a single frame into individual camera feeds.

        Args:
            frame: Video frame to segment
            use_cached_layout: Use cached grid layout if available

        Returns:
            List of camera dictionaries with extracted regions
        """
        if use_cached_layout and self.grid_layout is not None:
            # Use cached layout for faster processing
            detector = GridDetector(frame)
            cameras = detector.extract_camera_regions(grid_layout=self.grid_layout)
        else:
            # Detect layout for this frame
            detector = GridDetector(frame)
            cameras = detector.extract_camera_regions()
            if not self.is_calibrated:
                self.grid_layout = detector.grid_layout
                self.is_calibrated = True

        return cameras

    def segment_video(
        self,
        video_source,
        output_dir: Optional[Path] = None,
        save_format: str = 'mp4',
        fps: int = 30
    ) -> Generator[List[Dict], None, None]:
        """
        Segment entire video into individual camera feeds.

        Args:
            video_source: Path to video file or cv2.VideoCapture object
            output_dir: Directory to save individual camera videos (optional)
            save_format: Video format for saved files (mp4, avi)
            fps: Frames per second for output videos

        Yields:
            List of camera dictionaries for each frame
        """
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = video_source

        if not cap.isOpened():
            raise ValueError("Could not open video source")

        # Calibrate if not already done
        if not self.is_calibrated:
            logger.info("Auto-calibrating grid layout...")
            self.calibrate(cap)
            # Reset to beginning
            if isinstance(video_source, str):
                cap.release()
                cap = cv2.VideoCapture(video_source)

        # Setup video writers if saving
        writers = {}
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v') if save_format == 'mp4' else cv2.VideoWriter_fourcc(*'XVID')

            for camera_id in range(self.grid_layout['total_cameras']):
                output_path = output_dir / f"camera_{camera_id:03d}.{save_format}"
                writers[camera_id] = {
                    'path': output_path,
                    'writer': None,  # Will be initialized with first frame
                }

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Segment frame
                cameras = self.segment_frame(frame, use_cached_layout=True)

                # Save individual camera feeds if output_dir specified
                if output_dir:
                    for camera in cameras:
                        camera_id = camera['id']

                        # Initialize writer on first frame
                        if writers[camera_id]['writer'] is None:
                            h, w = camera['frame'].shape[:2]
                            writers[camera_id]['writer'] = cv2.VideoWriter(
                                str(writers[camera_id]['path']),
                                fourcc,
                                fps,
                                (w, h)
                            )

                        # Write frame
                        writers[camera_id]['writer'].write(camera['frame'])

                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames...")

                yield cameras

        finally:
            # Cleanup
            cap.release()

            if output_dir:
                for camera_id, writer_info in writers.items():
                    if writer_info['writer'] is not None:
                        writer_info['writer'].release()

                logger.info(f"Saved {len(writers)} camera videos to {output_dir}")

        logger.info(f"Segmentation complete: processed {frame_count} frames")

    def segment_video_batch(
        self,
        video_source,
        batch_size: int = 30
    ) -> Generator[List[List[Dict]], None, None]:
        """
        Segment video in batches for efficient batch processing.

        Args:
            video_source: Path to video file or cv2.VideoCapture object
            batch_size: Number of frames per batch

        Yields:
            Batch of camera lists (batch_size x num_cameras)
        """
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = video_source

        if not cap.isOpened():
            raise ValueError("Could not open video source")

        # Calibrate if not already done
        if not self.is_calibrated:
            self.calibrate(cap)
            if isinstance(video_source, str):
                cap.release()
                cap = cv2.VideoCapture(video_source)

        batch = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Yield remaining batch if any
                    if batch:
                        yield batch
                    break

                cameras = self.segment_frame(frame, use_cached_layout=True)
                batch.append(cameras)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        finally:
            cap.release()

    def get_camera_stream(
        self,
        video_source,
        camera_id: int
    ) -> Generator[np.ndarray, None, None]:
        """
        Extract single camera stream from grid video.

        Args:
            video_source: Path to video file or cv2.VideoCapture object
            camera_id: Camera ID to extract (0 to total_cameras-1)

        Yields:
            Individual camera frames
        """
        if not self.is_calibrated:
            raise ValueError("Must calibrate before extracting camera stream")

        if camera_id >= self.grid_layout['total_cameras']:
            raise ValueError(
                f"Camera ID {camera_id} out of range (0-{self.grid_layout['total_cameras'] - 1})"
            )

        for cameras in self.segment_video(video_source):
            camera_frame = cameras[camera_id]['frame']
            yield camera_frame


class StreamSegmenter:
    """
    Real-time stream segmentation for live video feeds.

    Optimized for low-latency processing of live camera grids.
    """

    def __init__(self, grid_layout: Optional[Dict] = None):
        """
        Initialize stream segmenter.

        Args:
            grid_layout: Pre-defined grid layout (optional)
        """
        self.grid_layout = grid_layout
        self.is_calibrated = grid_layout is not None

    def process_frame(self, frame: np.ndarray, auto_calibrate: bool = True) -> List[Dict]:
        """
        Process single frame from live stream.

        Args:
            frame: Live video frame
            auto_calibrate: Auto-calibrate grid on first frame

        Returns:
            List of segmented camera feeds
        """
        if not self.is_calibrated and auto_calibrate:
            detector = GridDetector(frame)
            self.grid_layout = detector.detect_grid_layout()
            self.is_calibrated = True
            logger.info("Stream auto-calibrated")

        if not self.is_calibrated:
            raise ValueError("Stream not calibrated. Set grid_layout or enable auto_calibrate.")

        detector = GridDetector(frame)
        return detector.extract_camera_regions(grid_layout=self.grid_layout)

    def set_grid_layout(self, rows: int, cols: int, frame_shape: Tuple[int, int]):
        """
        Manually set grid layout.

        Args:
            rows: Number of grid rows
            cols: Number of grid columns
            frame_shape: (height, width) of video frame
        """
        height, width = frame_shape

        # Calculate evenly spaced grid
        h_step = height / rows
        v_step = width / cols

        h_lines = [i * h_step for i in range(rows + 1)]
        v_lines = [i * v_step for i in range(cols + 1)]

        self.grid_layout = {
            'rows': rows,
            'cols': cols,
            'h_lines': h_lines,
            'v_lines': v_lines,
            'total_cameras': rows * cols,
        }
        self.is_calibrated = True

        logger.info(f"Manual grid layout set: {rows}x{cols}")
