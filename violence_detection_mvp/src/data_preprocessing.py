"""
Data preprocessing module for Violence Detection MVP.
Handles video frame extraction, labeling, and data preparation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator, Optional
from random import shuffle
import logging
from tqdm import tqdm

from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract frames from video files for violence detection."""

    def __init__(self, config: Config = Config):
        """Initialize the frame extractor with configuration."""
        self.config = config
        self.img_size = config.IMG_SIZE
        self.frames_per_video = config.FRAMES_PER_VIDEO
        self.img_size_tuple = config.IMG_SIZE_TUPLE

    def _calculate_evenly_spaced_indices(self, total_frames: int) -> List[int]:
        """
        Calculate evenly-spaced frame indices for extraction.

        Args:
            total_frames: Total number of frames in the video

        Returns:
            List of frame indices to extract
        """
        if total_frames <= self.frames_per_video:
            # If video has fewer frames than needed, take all and repeat last
            indices = list(range(total_frames))
            # Pad with last frame index
            while len(indices) < self.frames_per_video:
                indices.append(total_frames - 1 if total_frames > 0 else 0)
            return indices

        # Calculate evenly spaced indices
        step = (total_frames - 1) / (self.frames_per_video - 1)
        indices = [int(round(i * step)) for i in range(self.frames_per_video)]

        # Ensure indices are within bounds
        indices = [min(idx, total_frames - 1) for idx in indices]

        return indices

    def extract_frames(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Extract fixed number of evenly-spaced frames from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Array of extracted frames normalized to [0,1] range, or None if extraction fails
        """
        try:
            vidcap = cv2.VideoCapture(str(video_path))

            # CRITICAL FIX: Set timeout for corrupted videos
            vidcap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
            vidcap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)  # 3 second read timeout

            if not vidcap.isOpened():
                logger.warning(f"Could not open video (skipping): {video_path}")
                return None

            # Get video properties
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            logger.debug(f"Video info - Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")

            if total_frames < self.frames_per_video:
                logger.warning(f"Video has only {total_frames} frames, need {self.frames_per_video}")

            # Calculate frame indices for evenly-spaced extraction
            if self.config.FRAME_EXTRACTION_METHOD == "evenly_spaced":
                frame_indices = self._calculate_evenly_spaced_indices(total_frames)
            else:
                frame_indices = list(range(min(self.frames_per_video, total_frames)))

            frames = []
            current_frame = 0

            # Extract frames at calculated indices
            for target_index in frame_indices:
                # Seek to target frame
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
                success, image = vidcap.read()

                if success:
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Resize to target size
                    interpolation = getattr(cv2, f"INTER_{self.config.FRAME_RESIZE_METHOD.upper()}")
                    resized_image = cv2.resize(
                        rgb_image,
                        dsize=self.img_size_tuple,
                        interpolation=interpolation
                    )

                    frames.append(resized_image)
                else:
                    logger.warning(f"Failed to read frame at index {target_index}")
                    # Use last valid frame if available
                    if frames:
                        frames.append(frames[-1])
                    else:
                        # Create a black frame as fallback
                        black_frame = np.zeros(self.img_size_tuple + (3,), dtype=np.uint8)
                        frames.append(black_frame)

            vidcap.release()

            # Ensure we have exactly the required number of frames
            while len(frames) < self.frames_per_video:
                # Pad with last frame if video is too short
                last_frame = frames[-1] if frames else np.zeros(
                    self.img_size_tuple + (3,), dtype=np.uint8
                )
                frames.append(last_frame)

            # Trim if we have too many frames
            frames = frames[:self.frames_per_video]

            # Convert to numpy array and normalize
            result = np.array(frames, dtype=np.float32)

            if self.config.NORMALIZE_FRAMES:
                min_val, max_val = self.config.FRAMES_NORMALIZATION_RANGE
                result = result / 255.0 * (max_val - min_val) + min_val
                result = result.astype(getattr(np, self.config.FEATURE_DTYPE))

            return result

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return None

    def extract_frames_batch(self, video_paths: List[Path]) -> List[np.ndarray]:
        """
        Extract frames from multiple videos.

        Args:
            video_paths: List of video file paths

        Returns:
            List of frame arrays
        """
        extracted_frames = []

        for i, video_path in enumerate(video_paths):
            if i % 10 == 0:
                logger.info(f"Processing video {i+1}/{len(video_paths)}")

            frames = self.extract_frames(video_path)
            if frames is not None:
                extracted_frames.append(frames)

        return extracted_frames


class VideoLabeler:
    """Handle video labeling based on filename patterns."""

    def __init__(self, config: Config = Config):
        """Initialize the labeler with configuration."""
        self.config = config
        self.violence_prefixes = config.VIOLENCE_PREFIXES
        self.no_violence_prefixes = config.NO_VIOLENCE_PREFIXES

    def label_from_filename(self, filename: str) -> List[int]:
        """
        Generate label from filename based on prefixes.

        Args:
            filename: Video filename

        Returns:
            One-hot encoded label [violence, no_violence]
        """
        filename_lower = filename.lower()

        # Check for violence indicators
        for prefix in self.violence_prefixes:
            if filename_lower.startswith(prefix.lower()):
                return [1, 0]  # Violence

        # Check for no-violence indicators
        for prefix in self.no_violence_prefixes:
            if filename_lower.startswith(prefix.lower()):
                return [0, 1]  # No violence

        # Default to no violence if pattern not found
        logger.warning(f"Unknown filename pattern: {filename}, defaulting to no violence")
        return [0, 1]

    def create_labels_from_directory(self, data_dir: Path) -> Tuple[List[str], List[List[int]]]:
        """
        Create labels for all videos in a directory.

        Args:
            data_dir: Directory containing video files

        Returns:
            Tuple of (video_names, labels)
        """
        names = []
        labels = []

        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return names, labels

        # Walk through directory and collect video files
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                # Check if file has valid video extension
                if any(filename.lower().endswith(ext) for ext in self.config.VIDEO_EXTENSIONS):
                    names.append(filename)
                    label = self.label_from_filename(filename)
                    labels.append(label)

        # Shuffle the data
        combined = list(zip(names, labels))
        shuffle(combined)
        names, labels = zip(*combined) if combined else ([], [])

        logger.info(f"Found {len(names)} video files")
        return list(names), list(labels)


class DatasetSplitter:
    """Handle train/test dataset splitting."""

    def __init__(self, config: Config = Config):
        """Initialize the splitter with configuration."""
        self.config = config
        self.train_ratio = config.TRAIN_TEST_SPLIT

    def split_dataset(
        self,
        names: List[str],
        labels: List[List[int]]
    ) -> Tuple[List[str], List[str], List[List[int]], List[List[int]]]:
        """
        Split dataset into training and testing sets.

        Args:
            names: List of video filenames
            labels: List of corresponding labels

        Returns:
            Tuple of (train_names, test_names, train_labels, test_labels)
        """
        if len(names) != len(labels):
            raise ValueError("Names and labels must have the same length")

        split_index = int(len(names) * self.train_ratio)

        train_names = names[:split_index]
        test_names = names[split_index:]
        train_labels = labels[:split_index]
        test_labels = labels[split_index:]

        logger.info(f"Split dataset: {len(train_names)} training, {len(test_names)} testing")

        return train_names, test_names, train_labels, test_labels


class DataPreprocessor:
    """Main data preprocessing pipeline."""

    def __init__(self, config: Config = Config):
        """Initialize the preprocessor with all components."""
        self.config = config
        self.frame_extractor = VideoFrameExtractor(config)
        self.labeler = VideoLabeler(config)
        self.splitter = DatasetSplitter(config)

    def preprocess_dataset(self, data_dir: Path) -> Tuple[
        List[str], List[str], List[List[int]], List[List[int]]
    ]:
        """
        Complete preprocessing pipeline.

        Args:
            data_dir: Directory containing video files

        Returns:
            Tuple of (train_names, test_names, train_labels, test_labels)
        """
        logger.info("Starting dataset preprocessing...")

        # Create labels from directory
        names, labels = self.labeler.create_labels_from_directory(data_dir)

        if not names:
            raise ValueError("No video files found in the specified directory")

        # Split into train/test
        train_names, test_names, train_labels, test_labels = self.splitter.split_dataset(
            names, labels
        )

        logger.info("Dataset preprocessing completed")
        return train_names, test_names, train_labels, test_labels

    def extract_and_save_frames(
        self,
        video_names: List[str],
        data_dir: Path,
        output_dir: Path,
        prefix: str = "frames"
    ) -> None:
        """
        Extract frames from videos and save to disk.

        Args:
            video_names: List of video filenames
            data_dir: Directory containing video files
            output_dir: Directory to save extracted frames
            prefix: Prefix for output files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, video_name in enumerate(video_names):
            video_path = data_dir / video_name
            frames = self.frame_extractor.extract_frames(video_path)

            if frames is not None:
                output_path = output_dir / f"{prefix}_{i:04d}.npy"
                np.save(output_path, frames)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(video_names)} videos")

    def print_progress(self, count: int, max_count: int) -> None:
        """Print progress percentage."""
        pct_complete = count / max_count
        msg = f"\r- Progress: {pct_complete:.1%}"
        print(msg, end='', flush=True)

    def validate_dataset_integrity(self, data_dir: Path) -> dict:
        """
        Validate the integrity of all videos in the dataset.

        Args:
            data_dir: Directory containing video files

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating dataset integrity in {data_dir}")
        return analyze_dataset_videos(data_dir, self.config)

    def preprocess_single_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None
    ) -> Optional[np.ndarray]:
        """
        Preprocess a single video file.

        Args:
            video_path: Path to the video file
            output_path: Optional path to save preprocessed frames

        Returns:
            Preprocessed frames array or None if processing fails
        """
        # Validate video first
        validation = validate_video_file(video_path)
        if not validation["valid"]:
            logger.error(f"Invalid video {video_path}: {validation['error']}")
            return None

        # Extract frames
        frames = self.frame_extractor.extract_frames(video_path)

        if frames is not None and output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, frames)
            logger.info(f"Saved preprocessed frames to {output_path}")

        return frames


def validate_video_file(video_path: Path) -> dict:
    """
    Validate if a video file can be opened and read.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with validation results and metadata
    """
    try:
        vidcap = cv2.VideoCapture(str(video_path))
        is_valid = vidcap.isOpened()

        if not is_valid:
            vidcap.release()
            return {"valid": False, "error": "Cannot open video file"}

        # Get video properties
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Test reading first frame
        success, frame = vidcap.read()
        vidcap.release()

        if not success:
            return {"valid": False, "error": "Cannot read video frames"}

        result = {
            "valid": True,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "file_size_mb": video_path.stat().st_size / (1024 * 1024)
        }

        # Add warnings for potential issues
        warnings = []
        if total_frames < 20:
            warnings.append(f"Video has only {total_frames} frames (need 20)")
        if fps < 10:
            warnings.append(f"Low FPS: {fps}")
        if duration < 1.0:
            warnings.append(f"Very short video: {duration:.2f}s")

        result["warnings"] = warnings
        return result

    except Exception as e:
        return {"valid": False, "error": str(e)}


def get_video_info(video_path: Path) -> dict:
    """
    Get detailed information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing comprehensive video information
    """
    return validate_video_file(video_path)


def analyze_dataset_videos(data_dir: Path, config: Config = Config) -> dict:
    """
    Analyze all videos in a dataset directory.

    Args:
        data_dir: Directory containing video files
        config: Configuration object

    Returns:
        Dictionary with dataset analysis results
    """
    if not data_dir.exists():
        return {"error": f"Directory not found: {data_dir}"}

    video_files = []
    for ext in config.VIDEO_EXTENSIONS:
        video_files.extend(data_dir.glob(f"**/*{ext}"))

    if not video_files:
        return {"error": "No video files found"}

    logger.info(f"Analyzing {len(video_files)} video files...")

    valid_videos = 0
    invalid_videos = 0
    total_duration = 0
    total_frames = 0
    total_size_mb = 0
    fps_values = []
    resolution_stats = {}
    warnings = []

    for video_path in video_files:
        info = validate_video_file(video_path)

        if info["valid"]:
            valid_videos += 1
            total_duration += info["duration_seconds"]
            total_frames += info["total_frames"]
            total_size_mb += info["file_size_mb"]
            fps_values.append(info["fps"])

            # Track resolutions
            resolution = f"{info['width']}x{info['height']}"
            resolution_stats[resolution] = resolution_stats.get(resolution, 0) + 1

            if info["warnings"]:
                warnings.extend([f"{video_path.name}: {w}" for w in info["warnings"]])
        else:
            invalid_videos += 1
            warnings.append(f"{video_path.name}: {info['error']}")

    analysis = {
        "total_videos": len(video_files),
        "valid_videos": valid_videos,
        "invalid_videos": invalid_videos,
        "total_duration_hours": total_duration / 3600,
        "total_frames": total_frames,
        "total_size_gb": total_size_mb / 1024,
        "average_fps": np.mean(fps_values) if fps_values else 0,
        "resolution_distribution": resolution_stats,
        "warnings": warnings[:10],  # Limit warnings shown
        "total_warnings": len(warnings)
    }

    logger.info(f"Dataset analysis completed: {analysis}")
    return analysis