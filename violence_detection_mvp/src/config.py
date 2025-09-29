"""
Configuration file for Violence Detection MVP project.
Contains hyperparameters, paths, and model settings.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models"

class Config:
    """Configuration class containing all project settings."""

    # Data dimensions
    IMG_SIZE: int = 224
    IMG_SIZE_TUPLE: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)
    NUM_CHANNELS: int = 3
    IMG_SIZE_FLAT: int = IMG_SIZE * IMG_SIZE * NUM_CHANNELS

    # Model parameters
    NUM_CLASSES: int = 2  # Violence, No Violence
    FRAMES_PER_VIDEO: int = 20
    TRANSFER_VALUES_SIZE: int = 4096  # VGG19 fc2 layer output

    # LSTM model parameters
    RNN_SIZE: int = 128
    CHUNK_SIZE: int = 4096
    N_CHUNKS: int = 20

    # Training parameters
    BATCH_SIZE: int = 64
    EPOCHS: int = 250
    LEARNING_RATE: float = 0.0001
    VALIDATION_SPLIT: float = 0.2
    TRAIN_TEST_SPLIT: float = 0.8

    # Dropout and regularization
    DROPOUT_RATE: float = 0.5

    # File paths
    RAW_DATA_DIR: Path = DATA_ROOT / "raw"
    PROCESSED_DATA_DIR: Path = DATA_ROOT / "processed"
    MODELS_DIR: Path = MODEL_ROOT

    # Cache files
    TRAIN_CACHE_FILE: str = "train_features.h5"
    TEST_CACHE_FILE: str = "test_features.h5"

    # Video parameters
    VIDEO_EXTENSIONS: Tuple[str, ...] = (".avi", ".mp4", ".mov", ".mkv")

    # VGG19 configuration
    VGG19_WEIGHTS: str = "imagenet"
    VGG19_INCLUDE_TOP: bool = True
    VGG19_TRANSFER_LAYER: str = "fc2"
    VGG19_INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
    VGG19_PREPROCESSING: str = "tf"  # TensorFlow preprocessing

    # Feature extraction parameters
    FEATURE_EXTRACTION_BATCH_SIZE: int = 32
    FEATURE_CACHE_COMPRESSION: str = "gzip"
    FEATURE_DTYPE: str = "float16"  # Memory optimization

    # Frame extraction parameters
    FRAME_EXTRACTION_METHOD: str = "evenly_spaced"  # or "sequential"
    FRAME_RESIZE_METHOD: str = "cubic"  # cv2.INTER_CUBIC
    NORMALIZE_FRAMES: bool = True
    FRAMES_NORMALIZATION_RANGE: Tuple[float, float] = (0.0, 1.0)

    # Training callbacks
    EARLY_STOPPING_PATIENCE: int = 10
    REDUCE_LR_PATIENCE: int = 7
    REDUCE_LR_FACTOR: float = 0.1

    # Logging and visualization
    PLOT_DPI: int = 1000
    FIGURE_SIZE: Tuple[int, int] = (10, 6)

    # Class mapping
    CLASS_LABELS: Dict[str, int] = {
        "violence": 0,
        "no_violence": 1
    }

    # File naming patterns for labeling
    VIOLENCE_PREFIXES: Tuple[str, ...] = ("fi", "V")
    NO_VIOLENCE_PREFIXES: Tuple[str, ...] = ("no", "NV")

    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_cache_path(cls, cache_type: str) -> Path:
        """Get the full path for cache files."""
        if cache_type == "train":
            return cls.PROCESSED_DATA_DIR / cls.TRAIN_CACHE_FILE
        elif cache_type == "test":
            return cls.PROCESSED_DATA_DIR / cls.TEST_CACHE_FILE
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get the full path for model files."""
        return cls.MODELS_DIR / f"{model_name}.h5"

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration parameters."""
        assert cls.IMG_SIZE > 0, "Image size must be positive"
        assert cls.NUM_CHANNELS in [1, 3], "Number of channels must be 1 or 3"
        assert cls.FRAMES_PER_VIDEO > 0, "Frames per video must be positive"
        assert 0 < cls.VALIDATION_SPLIT < 1, "Validation split must be between 0 and 1"
        assert 0 < cls.TRAIN_TEST_SPLIT < 1, "Train test split must be between 0 and 1"
        assert cls.LEARNING_RATE > 0, "Learning rate must be positive"
        assert cls.BATCH_SIZE > 0, "Batch size must be positive"
        return True

# Create directories on import
Config.create_directories()
Config.validate_config()