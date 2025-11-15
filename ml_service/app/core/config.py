"""
Core configuration for NexaraVision ML Service.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "NexaraVision ML Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model configuration
    MODEL_PATH: str = "/app/models/ultimate_best_model.h5"
    NUM_FRAMES: int = 20
    FRAME_SIZE: tuple = (224, 224)

    # Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 32
    GPU_MEMORY_FRACTION: float = 0.8

    # API
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB

    # Redis (optional)
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_ENABLED: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
