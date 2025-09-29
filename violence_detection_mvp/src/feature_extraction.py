"""
Feature extraction module using VGG19 transfer learning.
Handles extraction and caching of transfer values from video frames.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator, Optional
import logging
import time

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Progbar

from .config import Config
from .data_preprocessing import VideoFrameExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VGG19FeatureExtractor:
    """Extract features using pre-trained VGG19 model."""

    def __init__(self, config: Config = Config):
        """Initialize the VGG19 feature extractor."""
        self.config = config
        self.transfer_layer_name = config.VGG19_TRANSFER_LAYER
        self.transfer_values_size = config.TRANSFER_VALUES_SIZE

        # Load VGG19 model
        self._load_vgg19_model()
        self._create_transfer_model()

    def _load_vgg19_model(self) -> None:
        """Load the pre-trained VGG19 model."""
        logger.info("Loading VGG19 model...")
        try:
            self.vgg19_model = VGG19(
                include_top=self.config.VGG19_INCLUDE_TOP,
                weights=self.config.VGG19_WEIGHTS,
                input_shape=self.config.VGG19_INPUT_SHAPE
            )
            logger.info(f"VGG19 model loaded successfully with input shape: {self.config.VGG19_INPUT_SHAPE}")
        except Exception as e:
            logger.error(f"Failed to load VGG19 model: {str(e)}")
            raise

    def _create_transfer_model(self) -> None:
        """Create the transfer learning model using fc2 layer."""
        transfer_layer = self.vgg19_model.get_layer(self.transfer_layer_name)

        self.transfer_model = Model(
            inputs=self.vgg19_model.input,
            outputs=transfer_layer.output
        )

        # Verify transfer values size
        actual_size = K.int_shape(transfer_layer.output)[1]
        if actual_size != self.transfer_values_size:
            logger.warning(
                f"Transfer values size mismatch: expected {self.transfer_values_size}, "
                f"got {actual_size}"
            )
            self.transfer_values_size = actual_size

        logger.info(f"Transfer model created with output size: {self.transfer_values_size}")

    def _preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess frames for VGG19 model.

        Args:
            frames: Raw frames with values in [0,1] range

        Returns:
            Preprocessed frames ready for VGG19
        """
        # Convert to [0,255] range if normalized
        if frames.max() <= 1.0:
            frames_255 = frames * 255.0
        else:
            frames_255 = frames

        # Apply VGG19 preprocessing
        return preprocess_input(frames_255.astype(np.float32))

    def extract_features_from_frames(self, frames: np.ndarray, batch_size: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Extract VGG19 features from video frames.

        Args:
            frames: Array of video frames with shape (n_frames, height, width, channels)
            batch_size: Batch size for processing (uses config default if None)

        Returns:
            Transfer values array with shape (n_frames, transfer_values_size)
        """
        try:
            if frames.shape[0] != self.config.FRAMES_PER_VIDEO:
                logger.warning(
                    f"Expected {self.config.FRAMES_PER_VIDEO} frames, "
                    f"got {frames.shape[0]}"
                )

            # Validate input shape
            expected_shape = (frames.shape[0],) + self.config.VGG19_INPUT_SHAPE
            if frames.shape != expected_shape:
                logger.error(f"Invalid input shape: {frames.shape}, expected: {expected_shape}")
                return None

            # Preprocess frames for VGG19
            preprocessed_frames = self._preprocess_frames(frames)

            # Use batch processing for efficiency
            if batch_size is None:
                batch_size = self.config.FEATURE_EXTRACTION_BATCH_SIZE

            # Predict transfer values in batches
            transfer_values = self.transfer_model.predict(
                preprocessed_frames,
                batch_size=batch_size,
                verbose=0
            )

            # Convert to specified dtype to save memory
            return transfer_values.astype(getattr(np, self.config.FEATURE_DTYPE))

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None

    def extract_features_from_video(
        self,
        video_path: Path,
        frame_extractor: VideoFrameExtractor
    ) -> Optional[np.ndarray]:
        """
        Extract features directly from a video file.

        Args:
            video_path: Path to the video file
            frame_extractor: VideoFrameExtractor instance

        Returns:
            Transfer values array or None if extraction fails
        """
        frames = frame_extractor.extract_frames(video_path)
        if frames is None:
            return None

        return self.extract_features_from_frames(frames)


class FeatureCache:
    """Handle caching and loading of extracted features."""

    def __init__(self, config: Config = Config):
        """Initialize the feature cache."""
        self.config = config

    def save_features_to_cache(
        self,
        features: List[np.ndarray],
        labels: List[List[int]],
        cache_path: Path,
        compression: Optional[str] = None
    ) -> None:
        """
        Save features and labels to HDF5 cache file.

        Args:
            features: List of feature arrays
            labels: List of corresponding labels
            cache_path: Path to save the cache file
            compression: Compression method (gzip, lzf, szip)
        """
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert lists to numpy arrays
            all_features = np.concatenate(features, axis=0)
            all_labels = np.repeat(labels, self.config.FRAMES_PER_VIDEO, axis=0)

            logger.info(f"Saving {len(all_features)} feature vectors to {cache_path}")

            # Use configuration compression or provided compression
            comp = compression or self.config.FEATURE_CACHE_COMPRESSION

            with h5py.File(cache_path, 'w') as f:
                # Save features with compression
                f.create_dataset(
                    'data',
                    data=all_features,
                    dtype=getattr(np, self.config.FEATURE_DTYPE),
                    compression=comp,
                    chunks=True
                )

                # Save labels
                f.create_dataset(
                    'labels',
                    data=all_labels,
                    dtype=np.int8,  # Efficient for binary labels
                    compression=comp,
                    chunks=True
                )

                # Save metadata
                f.attrs['num_videos'] = len(features)
                f.attrs['frames_per_video'] = self.config.FRAMES_PER_VIDEO
                f.attrs['feature_size'] = all_features.shape[1]
                f.attrs['compression'] = comp

            # Log file size
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"Features saved to cache successfully. File size: {file_size_mb:.2f} MB")

        except Exception as e:
            logger.error(f"Error saving features to cache: {str(e)}")
            # Clean up partial file if it exists
            if cache_path.exists():
                cache_path.unlink()
            raise

    def load_features_from_cache(self, cache_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels from HDF5 cache file.

        Args:
            cache_path: Path to the cache file

        Returns:
            Tuple of (features, labels)
        """
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        try:
            logger.info(f"Loading features from {cache_path}")

            with h5py.File(cache_path, 'r') as f:
                # Load data
                features = f['data'][:]
                labels = f['labels'][:]

                # Log metadata if available
                if 'num_videos' in f.attrs:
                    logger.info(f"Cache metadata - Videos: {f.attrs['num_videos']}, "
                              f"Feature size: {f.attrs['feature_size']}, "
                              f"Compression: {f.attrs.get('compression', 'none')}")

            logger.info(f"Loaded {len(features)} feature vectors from cache")
            return features, labels

        except Exception as e:
            logger.error(f"Error loading features from cache: {str(e)}")
            raise

    def cache_exists(self, cache_path: Path) -> bool:
        """Check if cache file exists."""
        return cache_path.exists()


class FeaturePipeline:
    """Complete feature extraction pipeline."""

    def __init__(self, config: Config = Config):
        """Initialize the feature pipeline."""
        self.config = config
        self.feature_extractor = VGG19FeatureExtractor(config)
        self.frame_extractor = VideoFrameExtractor(config)
        self.cache = FeatureCache(config)

    def process_video_batch(
        self,
        video_names: List[str],
        video_labels: List[List[int]],
        data_dir: Path,
        show_progress: bool = True
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Process a batch of videos and yield features and labels.

        Args:
            video_names: List of video filenames
            video_labels: List of corresponding labels
            data_dir: Directory containing video files
            show_progress: Whether to show progress bar

        Yields:
            Tuple of (transfer_values, labels) for each video
        """
        total_videos = len(video_names)
        processed_count = 0
        error_count = 0

        if show_progress:
            progress_bar = Progbar(total_videos, stateful_metrics=['processed', 'errors'])

        for i, (video_name, label) in enumerate(zip(video_names, video_labels)):
            video_path = data_dir / video_name

            if not video_path.exists():
                logger.warning(f"Video file not found: {video_path}")
                error_count += 1
                continue

            try:
                # Extract features
                features = self.feature_extractor.extract_features_from_video(
                    video_path, self.frame_extractor
                )

                if features is not None:
                    # Create labels for all frames
                    frame_labels = np.tile(label, (self.config.FRAMES_PER_VIDEO, 1))
                    yield features, frame_labels
                    processed_count += 1
                else:
                    error_count += 1

            except Exception as e:
                logger.error(f"Error processing video {video_name}: {str(e)}")
                error_count += 1

            # Update progress
            if show_progress:
                progress_bar.update(i + 1, values=[('processed', processed_count), ('errors', error_count)])
            elif (i + 1) % 10 == 0:
                logger.info(f"Processed {processed_count}/{total_videos} videos successfully, {error_count} errors")

        logger.info(f"Batch processing completed: {processed_count}/{total_videos} videos processed successfully")

    def extract_and_cache_features(
        self,
        video_names: List[str],
        video_labels: List[List[int]],
        data_dir: Path,
        cache_path: Path,
        force_recompute: bool = False,
        show_progress: bool = True
    ) -> dict:
        """
        Extract features from videos and save to cache.

        Args:
            video_names: List of video filenames
            video_labels: List of corresponding labels
            data_dir: Directory containing video files
            cache_path: Path to save the cache file
            force_recompute: Whether to recompute features even if cache exists
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with extraction statistics
        """
        if self.cache.cache_exists(cache_path) and not force_recompute:
            logger.info(f"Cache file already exists: {cache_path}")
            return {"status": "skipped", "reason": "cache_exists"}

        logger.info(f"Extracting features for {len(video_names)} videos...")

        all_features = []
        all_labels = []
        start_time = time.time()

        # Process videos in batches
        for features, labels in self.process_video_batch(
            video_names, video_labels, data_dir, show_progress
        ):
            all_features.append(features)
            all_labels.extend(labels)

        processing_time = time.time() - start_time

        if all_features:
            # Save to cache
            logger.info("Saving features to cache...")
            self.cache.save_features_to_cache(all_features, all_labels, cache_path)

            stats = {
                "status": "success",
                "total_videos": len(video_names),
                "processed_videos": len(all_features),
                "total_frames": len(all_labels),
                "processing_time_seconds": processing_time,
                "cache_path": str(cache_path)
            }
            logger.info(f"Feature extraction completed: {stats}")
            return stats
        else:
            error_msg = "No features extracted from videos"
            logger.error(error_msg)
            return {"status": "error", "reason": error_msg}

    def load_processed_features(self, cache_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load and organize features for training.

        Args:
            cache_path: Path to the cache file

        Returns:
            Tuple of (data, targets) where each element represents one video
        """
        features, labels = self.cache.load_features_from_cache(cache_path)

        # Organize features by video (20 frames per video)
        data = []
        targets = []

        frames_per_video = self.config.FRAMES_PER_VIDEO
        num_videos = len(features) // frames_per_video

        for i in range(num_videos):
            start_idx = i * frames_per_video
            end_idx = start_idx + frames_per_video

            video_features = features[start_idx:end_idx]
            video_label = labels[start_idx]  # All frames have the same label

            data.append(video_features)
            targets.append(video_label)

        logger.info(f"Loaded {len(data)} videos from processed features")
        return data, targets

    def get_feature_statistics(self, cache_path: Path) -> dict:
        """
        Get statistics about cached features.

        Args:
            cache_path: Path to the cache file

        Returns:
            Dictionary containing feature statistics
        """
        if not self.cache.cache_exists(cache_path):
            return {"error": "Cache file does not exist"}

        features, labels = self.cache.load_features_from_cache(cache_path)

        # Count classes
        violence_count = np.sum(labels[:, 0])
        no_violence_count = np.sum(labels[:, 1])

        stats = {
            "total_frames": len(features),
            "total_videos": len(features) // self.config.FRAMES_PER_VIDEO,
            "feature_size": features.shape[1],
            "violence_frames": int(violence_count),
            "no_violence_frames": int(no_violence_count),
            "violence_ratio": float(violence_count / len(features)),
            "feature_mean": float(np.mean(features)),
            "feature_std": float(np.std(features)),
            "feature_min": float(np.min(features)),
            "feature_max": float(np.max(features))
        }

        return stats


def print_vgg19_info() -> None:
    """Print information about the VGG19 model."""
    model = VGG19(include_top=True, weights='imagenet')
    print(f"VGG19 input shape: {model.input_shape}")
    print(f"VGG19 output shape: {model.output_shape}")

    # Print layer names and shapes
    print("\nVGG19 layers:")
    for i, layer in enumerate(model.layers):
        print(f"{i:2d}: {layer.name:20s} {layer.output_shape}")


def validate_feature_extraction(
    video_path: Path,
    config: Config = Config
) -> dict:
    """
    Validate feature extraction on a single video.

    Args:
        video_path: Path to test video
        config: Configuration object

    Returns:
        Dictionary containing validation results
    """
    try:
        # Initialize components
        frame_extractor = VideoFrameExtractor(config)
        feature_extractor = VGG19FeatureExtractor(config)

        # Extract frames
        frames = frame_extractor.extract_frames(video_path)
        if frames is None:
            return {"error": "Failed to extract frames"}

        # Extract features
        features = feature_extractor.extract_features_from_frames(frames)
        if features is None:
            return {"error": "Failed to extract features"}

        return {
            "success": True,
            "frames_shape": frames.shape,
            "features_shape": features.shape,
            "frames_dtype": str(frames.dtype),
            "features_dtype": str(features.dtype),
            "features_mean": float(np.mean(features)),
            "features_std": float(np.std(features))
        }

    except Exception as e:
        return {"error": str(e)}