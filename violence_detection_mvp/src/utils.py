"""
Utility functions for Violence Detection MVP.
Contains helper functions, data manipulation, and common operations.
"""

import os
import sys
import json
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time
from datetime import datetime

import numpy as np
import cv2

from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileManager:
    """Utility class for file operations."""

    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """
        Ensure directory exists, create if it doesn't.

        Args:
            directory: Directory path
        """
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy_file(source: Path, destination: Path, overwrite: bool = False) -> bool:
        """
        Copy file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing file

        Returns:
            True if successful, False otherwise
        """
        try:
            if destination.exists() and not overwrite:
                logger.warning(f"Destination file exists: {destination}")
                return False

            FileManager.ensure_directory(destination.parent)
            shutil.copy2(source, destination)
            logger.info(f"Copied {source} to {destination}")
            return True

        except Exception as e:
            logger.error(f"Failed to copy file: {str(e)}")
            return False

    @staticmethod
    def get_file_size(file_path: Path) -> Optional[int]:
        """
        Get file size in bytes.

        Args:
            file_path: Path to file

        Returns:
            File size in bytes, None if file doesn't exist
        """
        try:
            return file_path.stat().st_size
        except Exception:
            return None

    @staticmethod
    def get_file_hash(file_path: Path, algorithm: str = 'md5') -> Optional[str]:
        """
        Calculate file hash.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            File hash as hex string, None if failed
        """
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()

        except Exception as e:
            logger.error(f"Failed to calculate hash: {str(e)}")
            return None

    @staticmethod
    def clean_directory(directory: Path, pattern: str = "*", dry_run: bool = True) -> List[Path]:
        """
        Clean files matching pattern from directory.

        Args:
            directory: Directory to clean
            pattern: File pattern to match
            dry_run: If True, only list files without deleting

        Returns:
            List of files that would be/were deleted
        """
        if not directory.exists():
            return []

        matching_files = list(directory.glob(pattern))

        if dry_run:
            logger.info(f"Would delete {len(matching_files)} files from {directory}")
            return matching_files

        deleted_files = []
        for file_path in matching_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    deleted_files.append(file_path)
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    deleted_files.append(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {str(e)}")

        logger.info(f"Deleted {len(deleted_files)} files from {directory}")
        return deleted_files


class DataSaver:
    """Utility class for saving and loading data."""

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Path) -> bool:
        """
        Save data to JSON file.

        Args:
            data: Data to save
            file_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            FileManager.ensure_directory(file_path.parent)

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=DataSaver._json_serializer)

            logger.info(f"Saved JSON data to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save JSON: {str(e)}")
            return False

    @staticmethod
    def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file.

        Args:
            file_path: Input file path

        Returns:
            Loaded data, None if failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            logger.info(f"Loaded JSON data from: {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load JSON: {str(e)}")
            return None

    @staticmethod
    def save_pickle(data: Any, file_path: Path) -> bool:
        """
        Save data to pickle file.

        Args:
            data: Data to save
            file_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            FileManager.ensure_directory(file_path.parent)

            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved pickle data to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save pickle: {str(e)}")
            return False

    @staticmethod
    def load_pickle(file_path: Path) -> Any:
        """
        Load data from pickle file.

        Args:
            file_path: Input file path

        Returns:
            Loaded data, None if failed
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            logger.info(f"Loaded pickle data from: {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load pickle: {str(e)}")
            return None

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ProgressTracker:
    """Utility class for tracking progress."""

    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            description: Description of the process
        """
        self.total_items = total_items
        self.current_item = 0
        self.description = description
        self.start_time = time.time()

    def update(self, increment: int = 1) -> None:
        """
        Update progress.

        Args:
            increment: Number of items processed
        """
        self.current_item += increment
        self._print_progress()

    def _print_progress(self) -> None:
        """Print progress to console."""
        if self.total_items == 0:
            return

        percentage = (self.current_item / self.total_items) * 100
        elapsed_time = time.time() - self.start_time

        if self.current_item > 0:
            eta = (elapsed_time / self.current_item) * (self.total_items - self.current_item)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: N/A"

        progress_bar = self._create_progress_bar(percentage)

        msg = f"\r{self.description}: {progress_bar} {percentage:.1f}% ({self.current_item}/{self.total_items}) {eta_str}"
        print(msg, end='', flush=True)

        if self.current_item >= self.total_items:
            print()  # New line when complete

    def _create_progress_bar(self, percentage: float, length: int = 30) -> str:
        """Create ASCII progress bar."""
        filled_length = int(length * percentage / 100)
        bar = '█' * filled_length + '-' * (length - filled_length)
        return f"[{bar}]"


class VideoUtils:
    """Utility functions for video processing."""

    @staticmethod
    def get_video_properties(video_path: Path) -> Dict[str, Any]:
        """
        Get video properties.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary containing video properties
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return {"error": "Could not open video"}

            properties = {
                "filename": video_path.name,
                "path": str(video_path),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
                "file_size_bytes": FileManager.get_file_size(video_path)
            }

            # Calculate duration
            if properties["fps"] > 0:
                properties["duration_seconds"] = properties["frame_count"] / properties["fps"]
            else:
                properties["duration_seconds"] = 0

            cap.release()
            return properties

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def validate_video_format(video_path: Path, supported_extensions: List[str]) -> bool:
        """
        Validate video format.

        Args:
            video_path: Path to video file
            supported_extensions: List of supported file extensions

        Returns:
            True if format is supported, False otherwise
        """
        extension = video_path.suffix.lower()
        return extension in [ext.lower() for ext in supported_extensions]

    @staticmethod
    def extract_single_frame(video_path: Path, frame_number: int = 0) -> Optional[np.ndarray]:
        """
        Extract a single frame from video.

        Args:
            video_path: Path to video file
            frame_number: Frame number to extract

        Returns:
            Frame as numpy array, None if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return None

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            cap.release()

            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to extract frame: {str(e)}")
            return None


class ConfigManager:
    """Utility class for configuration management."""

    @staticmethod
    def save_config(config: Config, file_path: Path) -> bool:
        """
        Save configuration to file.

        Args:
            config: Configuration object
            file_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        config_dict = {}

        # Get all public attributes
        for attr_name in dir(config):
            if not attr_name.startswith('_') and not callable(getattr(config, attr_name)):
                value = getattr(config, attr_name)
                # Convert Path objects to strings
                if isinstance(value, Path):
                    config_dict[attr_name] = str(value)
                else:
                    config_dict[attr_name] = value

        return DataSaver.save_json(config_dict, file_path)

    @staticmethod
    def load_config_dict(file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load configuration dictionary from file.

        Args:
            file_path: Input file path

        Returns:
            Configuration dictionary, None if failed
        """
        return DataSaver.load_json(file_path)

    @staticmethod
    def create_config_from_dict(config_dict: Dict[str, Any]) -> Config:
        """
        Create configuration object from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration object
        """
        config = Config()

        for key, value in config_dict.items():
            if hasattr(config, key):
                # Convert string paths back to Path objects
                if key.endswith('_DIR') or key.endswith('_PATH'):
                    value = Path(value)
                setattr(config, key, value)

        return config


class SystemInfo:
    """Utility class for system information."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information.

        Returns:
            Dictionary containing system information
        """
        import platform
        import psutil

        try:
            info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "disk_usage": {}
            }

            # Get disk usage for current directory
            disk_usage = psutil.disk_usage('.')
            info["disk_usage"] = {
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_gb": disk_usage.free / (1024**3)
            }

            return info

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def check_dependencies() -> Dict[str, Any]:
        """
        Check if required dependencies are available.

        Returns:
            Dictionary containing dependency information
        """
        dependencies = {
            "tensorflow": False,
            "opencv": False,
            "sklearn": False,
            "matplotlib": False,
            "seaborn": False,
            "h5py": False,
            "numpy": False
        }

        versions = {}

        # Check each dependency
        for dep_name in dependencies.keys():
            try:
                if dep_name == "opencv":
                    import cv2
                    dependencies[dep_name] = True
                    versions[dep_name] = cv2.__version__
                elif dep_name == "sklearn":
                    import sklearn
                    dependencies[dep_name] = True
                    versions[dep_name] = sklearn.__version__
                else:
                    module = __import__(dep_name)
                    dependencies[dep_name] = True
                    versions[dep_name] = getattr(module, '__version__', 'unknown')

            except ImportError:
                dependencies[dep_name] = False
                versions[dep_name] = "not installed"

        return {
            "dependencies": dependencies,
            "versions": versions,
            "all_available": all(dependencies.values())
        }


class Logger:
    """Enhanced logging utility."""

    @staticmethod
    def setup_logging(
        log_file: Optional[Path] = None,
        level: str = "INFO",
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup enhanced logging.

        Args:
            log_file: Optional log file path
            level: Logging level
            format_string: Custom format string

        Returns:
            Configured logger
        """
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Create logger
        logger = logging.getLogger('violence_detection')
        logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(format_string)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            FileManager.ensure_directory(log_file.parent)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


def print_project_structure(project_root: Path, max_depth: int = 3) -> None:
    """
    Print project directory structure.

    Args:
        project_root: Root directory of the project
        max_depth: Maximum depth to display
    """
    def _print_tree(directory: Path, prefix: str = "", depth: int = 0) -> None:
        if depth >= max_depth:
            return

        items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")

            if item.is_dir() and not item.name.startswith('.'):
                extension = "    " if is_last else "│   "
                _print_tree(item, prefix + extension, depth + 1)

    print(f"Project structure: {project_root.name}/")
    _print_tree(project_root)


def validate_project_setup(project_root: Path) -> Dict[str, Any]:
    """
    Validate project setup and structure.

    Args:
        project_root: Root directory of the project

    Returns:
        Validation results
    """
    required_dirs = [
        "src",
        "data/raw",
        "data/processed",
        "models",
        "notebooks"
    ]

    required_files = [
        "src/config.py",
        "src/data_preprocessing.py",
        "src/feature_extraction.py",
        "src/model_architecture.py",
        "src/training.py",
        "src/evaluation.py",
        "src/inference.py",
        "src/utils.py",
        "src/visualization.py"
    ]

    results = {
        "project_root_exists": project_root.exists(),
        "required_directories": {},
        "required_files": {},
        "system_info": SystemInfo.get_system_info(),
        "dependencies": SystemInfo.check_dependencies()
    }

    # Check directories
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        results["required_directories"][dir_path] = full_path.exists()

    # Check files
    for file_path in required_files:
        full_path = project_root / file_path
        results["required_files"][file_path] = full_path.exists()

    # Summary
    results["all_directories_exist"] = all(results["required_directories"].values())
    results["all_files_exist"] = all(results["required_files"].values())
    results["setup_complete"] = (
        results["all_directories_exist"] and
        results["all_files_exist"] and
        results["dependencies"]["all_available"]
    )

    return results