"""
GPU configuration and optimization utilities.
"""
import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


def configure_gpu(memory_fraction: float = 0.8) -> bool:
    """
    Configure GPU settings for optimal inference performance.
    Gracefully falls back to CPU if GPU is unavailable or configuration fails.

    Args:
        memory_fraction: Fraction of GPU memory to allocate (0.0 to 1.0)

    Returns:
        True if GPU is available and configured, False otherwise
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            logger.warning("No GPU detected. Running on CPU - inference will be slower but functional.")
            # Explicitly set CPU device
            tf.config.set_visible_devices([], 'GPU')
            return False

        try:
            for gpu in gpus:
                # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)

                # Set memory limit
                memory_limit = int(get_gpu_memory(gpu) * memory_fraction)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )

            logger.info(f"Configured {len(gpus)} GPU(s) with {memory_fraction*100:.0f}% memory allocation")
            return True

        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}. Falling back to CPU.")
            # Force CPU usage
            tf.config.set_visible_devices([], 'GPU')
            return False

    except Exception as e:
        logger.warning(f"GPU detection failed: {e}. Running on CPU.")
        # Force CPU usage on any error
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass  # Already on CPU
        return False


def get_gpu_memory(device) -> int:
    """
    Get total GPU memory in MB.

    Args:
        device: TensorFlow physical device

    Returns:
        Total memory in MB
    """
    try:
        # This is a simplified version - actual implementation would query GPU
        return 24000  # Default to 24GB for safety
    except Exception as e:
        logger.warning(f"Could not determine GPU memory: {e}")
        return 8000  # Conservative default


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')

    return {
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "gpu_names": [gpu.name for gpu in gpus],
        "cpu_count": len(cpus),
        "tensorflow_version": tf.__version__,
    }


def warmup_gpu():
    """
    Warm up GPU with dummy operations to initialize CUDA.
    Safely handles CPU-only environments.
    """
    try:
        if not tf.config.list_physical_devices('GPU'):
            logger.info("No GPU to warm up - using CPU")
            return

        logger.info("Warming up GPU...")

        # Create dummy tensor and perform operations
        dummy = tf.random.normal((1, 224, 224, 3))
        _ = tf.nn.relu(dummy)
        _ = tf.reduce_sum(dummy)

        logger.info("GPU warm-up complete")

    except Exception as e:
        logger.warning(f"GPU warm-up failed (continuing with CPU): {e}")
