"""
Logging configuration for Violence Detection MVP project.
Provides structured logging with different levels and output formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import Config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    config: Config = Config
) -> logging.Logger:
    """
    Set up logging configuration for the project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        config: Configuration object

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("violence_detection")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Reduce TensorFlow logging noise
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(logging.WARNING)

    return logger


def create_log_file_path(config: Config = Config) -> Path:
    """
    Create a timestamped log file path.

    Args:
        config: Configuration object

    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"violence_detection_{timestamp}.log"
    return config.PROCESSED_DATA_DIR / "logs" / log_filename


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(f"violence_detection.{self.__class__.__name__}")

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging.

    Args:
        logger: Logger instance
    """
    import platform
    import psutil
    import tensorflow as tf

    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    logger.info(f"TensorFlow version: {tf.__version__}")

    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Available GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
    else:
        logger.info("No GPUs available")

    logger.info("=== End System Information ===")


def log_config_info(config: Config, logger: logging.Logger) -> None:
    """
    Log configuration information.

    Args:
        config: Configuration object
        logger: Logger instance
    """
    logger.info("=== Configuration ===")
    logger.info(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    logger.info(f"Frames per video: {config.FRAMES_PER_VIDEO}")
    logger.info(f"Transfer values size: {config.TRANSFER_VALUES_SIZE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info(f"VGG19 layer: {config.VGG19_TRANSFER_LAYER}")
    logger.info(f"Frame extraction method: {config.FRAME_EXTRACTION_METHOD}")
    logger.info(f"Feature dtype: {config.FEATURE_DTYPE}")
    logger.info("=== End Configuration ===")


def get_progress_logger(name: str = "progress") -> logging.Logger:
    """
    Get a logger specifically for progress reporting.

    Args:
        name: Logger name

    Returns:
        Progress logger instance
    """
    logger = logging.getLogger(f"violence_detection.{name}")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger