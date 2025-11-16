"""
Violence detection model loader and inference engine.
Supports both Keras 3 (.keras) and legacy Keras 2 (.h5) formats.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer for focusing on important frames."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(1, use_bias=False, dtype=self.dtype_policy)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Cast to input dtype to handle mixed precision
        attention_scores = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(tf.cast(attention_scores, inputs.dtype), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_spec(self, input_spec):
        """Keras 3.x output spec computation"""
        output_shape = (input_spec.shape[0], input_spec.shape[2])
        return tf.keras.KerasTensor(output_shape, dtype=input_spec.dtype)


class ViolenceDetector:
    """
    Violence detection model with optimized inference.

    Handles model loading, GPU warming, and prediction with confidence scoring.
    """

    def __init__(self, model_path: str):
        """
        Initialize violence detector with trained model.

        Args:
            model_path: Path to trained Keras model file (.h5 or .keras)
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model()

        logger.info("Model loaded successfully. Warming up GPU...")
        self._warmup()
        logger.info("Model ready for inference")

    def _load_model(self) -> tf.keras.Model:
        """
        Load Keras model with device-agnostic error handling.
        Supports both .h5 and .keras formats, handles GPU/CPU gracefully.

        Returns:
            Loaded Keras model
        """
        try:
            # Force CPU if GPU is not available
            # Load model with custom layers support
            with tf.device('/CPU:0'):
                # Determine if we're loading .keras (Keras 3) or .h5 (Keras 2)
                is_keras3_format = str(self.model_path).endswith('.keras')

                if is_keras3_format:
                    # Keras 3 native format - no safe_mode needed
                    model = tf.keras.models.load_model(
                        str(self.model_path),
                        custom_objects={'AttentionLayer': AttentionLayer},
                        compile=False
                    )
                else:
                    # Legacy Keras 2 .h5 format - needs safe_mode=False
                    model = tf.keras.models.load_model(
                        str(self.model_path),
                        custom_objects={'AttentionLayer': AttentionLayer},
                        compile=False,
                        safe_mode=False
                    )

            # Log model architecture summary
            try:
                total_params = sum([np.prod(layer.get_weights()[0].shape)
                                  for layer in model.layers
                                  if len(layer.get_weights()) > 0])
                logger.info(f"Model loaded: {len(model.layers)} layers, {total_params:,} parameters")
            except Exception as e:
                logger.warning(f"Could not compute parameters: {e}")
                logger.info(f"Model loaded: {len(model.layers)} layers")

            # Detect device being used
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            device = "GPU" if gpu_available else "CPU"
            logger.info(f"Model will run on: {device}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _warmup(self):
        """
        Warm up model with dummy inference to initialize kernels.
        Works for both GPU and CPU.
        """
        dummy_input = np.zeros((1, 20, 224, 224, 3), dtype=np.float32)

        try:
            _ = self.model.predict(dummy_input, verbose=0)
            device = "GPU" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU"
            logger.info(f"Model warm-up successful on {device}")
        except Exception as e:
            logger.warning(f"Model warm-up failed (model may still work): {e}")

    def predict(self, frames: np.ndarray) -> Dict[str, any]:
        """
        Predict violence probability from video frames.

        Args:
            frames: Numpy array of shape (20, 224, 224, 3) with normalized values

        Returns:
            Dictionary with:
                - violence_probability: float [0.0, 1.0]
                - confidence: str ("Low", "Medium", "High")
                - per_class_scores: dict with class probabilities
        """
        if frames.shape != (20, 224, 224, 3):
            raise ValueError(
                f"Invalid frame shape: expected (20, 224, 224, 3), got {frames.shape}"
            )

        # Expand dims for batch
        batch = np.expand_dims(frames, axis=0)

        try:
            # Inference
            prediction = self.model.predict(batch, verbose=0)

            # Extract probabilities (assuming binary classification: [non-violence, violence])
            violence_prob = float(prediction[0][1])
            non_violence_prob = float(prediction[0][0])

            return {
                "violence_probability": violence_prob,
                "confidence": self._get_confidence(violence_prob),
                "per_class_scores": {
                    "non_violence": non_violence_prob,
                    "violence": violence_prob,
                },
                "prediction": "violence" if violence_prob > 0.5 else "non_violence",
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(self, frames_batch: np.ndarray) -> list:
        """
        Predict violence probability for multiple videos in batch.

        Args:
            frames_batch: Numpy array of shape (batch_size, 20, 224, 224, 3)

        Returns:
            List of prediction dictionaries
        """
        if len(frames_batch.shape) != 5:
            raise ValueError(
                f"Invalid batch shape: expected (N, 20, 224, 224, 3), got {frames_batch.shape}"
            )

        try:
            predictions = self.model.predict(frames_batch, verbose=0)

            results = []
            for pred in predictions:
                violence_prob = float(pred[1])
                results.append({
                    "violence_probability": violence_prob,
                    "confidence": self._get_confidence(violence_prob),
                    "per_class_scores": {
                        "non_violence": float(pred[0]),
                        "violence": float(pred[1]),
                    },
                    "prediction": "violence" if violence_prob > 0.5 else "non_violence",
                })

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

    def _get_confidence(self, prob: float) -> str:
        """
        Get confidence level based on probability.

        Args:
            prob: Violence probability [0.0, 1.0]

        Returns:
            Confidence level string
        """
        # High confidence if probability is very certain (near 0 or 1)
        if prob > 0.9 or prob < 0.1:
            return "High"
        elif prob > 0.7 or prob < 0.3:
            return "Medium"
        else:
            return "Low"

    def get_model_info(self) -> Dict[str, any]:
        """
        Get model metadata and information.

        Returns:
            Dictionary with model information
        """
        # Convert TensorShape to list for JSON serialization
        input_shape = [int(x) if x is not None else None for x in self.model.input_shape]
        output_shape = [int(x) if x is not None else None for x in self.model.output_shape]

        return {
            "model_path": str(self.model_path),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "total_layers": len(self.model.layers),
            "trainable_params": int(sum([np.prod(v.shape) for v in self.model.trainable_weights])),
        }
