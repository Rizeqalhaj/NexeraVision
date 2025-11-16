"""
Core configuration for NexaraVision ML Service.
"""
import os
import logging
from pydantic_settings import BaseSettings
from pathlib import Path

logger = logging.getLogger(__name__)


def find_model_path() -> str:
    """
    Find the best available model file with fallback paths.

    Returns:
        Path to model file
    """
    # Possible model paths in priority order (.keras format preferred over .h5)
    search_paths = [
        # Docker paths - Keras 3 format first
        "/app/models/initial_best_model.keras",
        "/app/models/ultimate_best_model.h5",
        "/app/models/best_model.h5",
        # Local development paths - Keras 3 format first
        "ml_service/models/initial_best_model.keras",
        "ml_service/models/best_model.h5",
        "models/initial_best_model.keras",
        "models/best_model.h5",
        "downloaded_models/ultimate_best_model.h5",
        "downloaded_models/best_model.h5",
        # Absolute local paths
        "/home/admin/Desktop/NexaraVision/ml_service/models/initial_best_model.keras",
        "/home/admin/Desktop/NexaraVision/ml_service/models/best_model.h5",
        "/home/admin/Desktop/NexaraVision/downloaded_models/ultimate_best_model.h5",
    ]

    # Check environment variable first
    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        model_path = Path(env_model_path)
        if model_path.exists():
            logger.info(f"Using model from environment variable: {model_path}")
            return str(model_path)

    # Search for model in priority order
    for path_str in search_paths:
        model_path = Path(path_str)
        if model_path.exists():
            logger.info(f"Found model at: {model_path}")
            return str(model_path)

    # Default fallback
    default_path = "models/best_model.h5"
    logger.warning(f"No model found in search paths. Using default: {default_path}")
    return default_path


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "NexaraVision ML Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model configuration
    MODEL_PATH: str = find_model_path()
    NUM_FRAMES: int = 20
    FRAME_SIZE: tuple = (224, 224)

    # Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 32
    GPU_MEMORY_FRACTION: float = 0.8

    # API
    HOST: str = "0.0.0.0"
    PORT: int = 8003  # NexaraVision ML service port
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB

    # Redis (optional)
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_ENABLED: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
