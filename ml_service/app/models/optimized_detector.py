"""
Optimized violence detection with TensorFlow Lite for fast CPU inference.
Achieves 50-100ms inference time vs 90+ seconds with standard TF.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional
import time

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
        attention_scores = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(tf.cast(attention_scores, inputs.dtype), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_spec(self, input_spec):
        output_shape = (input_spec.shape[0], input_spec.shape[2])
        return tf.keras.KerasTensor(output_shape, dtype=input_spec.dtype)


class OptimizedViolenceDetector:
    """
    Fast violence detection with TensorFlow Lite optimization.

    Performance targets:
    - Model loading: ~30-60 seconds (one-time)
    - Inference: 50-100ms per batch of 20 frames
    - Total response: <500ms including frame extraction
    """

    def __init__(self, model_path: str, optimize_for_speed: bool = True):
        """
        Initialize optimized violence detector.

        Args:
            model_path: Path to trained Keras model file
            optimize_for_speed: Enable TFLite conversion for faster inference
        """
        self.model_path = Path(model_path)
        self.tflite_path = self.model_path.with_suffix('.tflite')
        self.interpreter = None
        self.keras_model = None
        self.use_tflite = optimize_for_speed
        self.input_details = None
        self.output_details = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Configure TensorFlow for optimal CPU performance
        self._configure_tensorflow()

        logger.info(f"Loading model from {model_path}")

        if self.use_tflite:
            self._load_or_convert_tflite()
        else:
            self.keras_model = self._load_keras_model()
            self._warmup_keras()

        logger.info("Model ready for fast inference")

    def _configure_tensorflow(self):
        """Configure TensorFlow for optimal CPU performance."""
        # Set number of threads based on CPU cores
        num_cores = os.cpu_count() or 4

        # Only configure threading if TensorFlow hasn't been initialized yet
        try:
            # Check if threading can still be configured
            current_inter = tf.config.threading.get_inter_op_parallelism_threads()
            current_intra = tf.config.threading.get_intra_op_parallelism_threads()
            if current_inter == 0 and current_intra == 0:
                tf.config.threading.set_inter_op_parallelism_threads(num_cores)
                tf.config.threading.set_intra_op_parallelism_threads(num_cores)
                logger.info(f"TensorFlow threading configured with {num_cores} threads")
            else:
                logger.info(f"TensorFlow already initialized with threading config")
        except RuntimeError:
            # TensorFlow already initialized, skip threading config
            logger.info("TensorFlow already initialized, using existing config")

        logger.info(f"TensorFlow configured for CPU with {num_cores} available cores")

    def _load_keras_model(self) -> tf.keras.Model:
        """Load Keras model with optimizations."""
        try:
            with tf.device('/CPU:0'):
                is_keras3_format = str(self.model_path).endswith('.keras')

                if is_keras3_format:
                    model = tf.keras.models.load_model(
                        str(self.model_path),
                        custom_objects={'AttentionLayer': AttentionLayer},
                        compile=False
                    )
                else:
                    model = tf.keras.models.load_model(
                        str(self.model_path),
                        custom_objects={'AttentionLayer': AttentionLayer},
                        compile=False,
                        safe_mode=False
                    )

            logger.info(f"Keras model loaded: {len(model.layers)} layers")
            return model

        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _load_or_convert_tflite(self):
        """Load existing TFLite model or convert from Keras."""
        # Check if TFLite model already exists
        if self.tflite_path.exists():
            logger.info(f"Loading existing TFLite model from {self.tflite_path}")
            self._load_tflite_interpreter()
            return

        # Convert Keras to TFLite
        logger.info("Converting Keras model to TensorFlow Lite for fast inference...")
        self.keras_model = self._load_keras_model()

        try:
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)

            # Optimization settings for speed
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float32]

            # Convert model
            start_time = time.time()
            tflite_model = converter.convert()
            conversion_time = time.time() - start_time

            # Save TFLite model
            with open(self.tflite_path, 'wb') as f:
                f.write(tflite_model)

            tflite_size = len(tflite_model) / (1024 * 1024)
            logger.info(f"TFLite conversion completed in {conversion_time:.1f}s")
            logger.info(f"TFLite model saved: {self.tflite_path} ({tflite_size:.1f}MB)")

            # Load the converted model
            self._load_tflite_interpreter()

        except Exception as e:
            logger.warning(f"TFLite conversion failed: {e}. Falling back to Keras model.")
            self.use_tflite = False
            self._warmup_keras()

    def _load_tflite_interpreter(self):
        """Load TFLite interpreter for fast inference."""
        try:
            # Create interpreter with optimized settings
            self.interpreter = tf.lite.Interpreter(
                model_path=str(self.tflite_path),
                num_threads=os.cpu_count() or 4
            )

            # Allocate tensors
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            logger.info(f"TFLite interpreter loaded with {os.cpu_count()} threads")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")

            # Warmup
            self._warmup_tflite()

        except Exception as e:
            logger.error(f"Failed to load TFLite interpreter: {e}")
            raise

    def _warmup_tflite(self):
        """Warm up TFLite interpreter with dummy inference."""
        logger.info("Warming up TFLite interpreter...")
        dummy_input = np.zeros((1, 20, 224, 224, 3), dtype=np.float32)

        for i in range(3):  # Multiple warmup runs
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()

        logger.info("TFLite warmup complete")

    def _warmup_keras(self):
        """Warm up Keras model."""
        logger.info("Warming up Keras model...")
        dummy_input = np.zeros((1, 20, 224, 224, 3), dtype=np.float32)

        try:
            _ = self.keras_model.predict(dummy_input, verbose=0)
            logger.info("Keras warmup complete")
        except Exception as e:
            logger.warning(f"Keras warmup failed: {e}")

    def predict(self, frames: np.ndarray) -> Dict[str, any]:
        """
        Fast prediction using TFLite or optimized Keras.

        Args:
            frames: Numpy array of shape (20, 224, 224, 3) with normalized values

        Returns:
            Dictionary with prediction results and timing info
        """
        if frames.shape != (20, 224, 224, 3):
            raise ValueError(
                f"Invalid frame shape: expected (20, 224, 224, 3), got {frames.shape}"
            )

        start_time = time.time()

        # Expand dims for batch
        batch = np.expand_dims(frames, axis=0).astype(np.float32)

        try:
            if self.use_tflite and self.interpreter is not None:
                # TFLite inference (FAST)
                self.interpreter.set_tensor(self.input_details[0]['index'], batch)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            else:
                # Keras inference (slower fallback)
                prediction = self.keras_model.predict(batch, verbose=0)

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Extract probabilities
            violence_prob = float(prediction[0][1])
            non_violence_prob = float(prediction[0][0])

            result = {
                "violence_probability": violence_prob,
                "confidence": self._get_confidence(violence_prob),
                "per_class_scores": {
                    "non_violence": non_violence_prob,
                    "violence": violence_prob,
                },
                "prediction": "violence" if violence_prob > 0.5 else "non_violence",
                "inference_time_ms": round(inference_time, 2),
                "backend": "tflite" if self.use_tflite else "keras",
            }

            logger.info(f"Inference completed in {inference_time:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(self, frames_batch: np.ndarray) -> list:
        """Predict violence for multiple video samples."""
        if len(frames_batch.shape) != 5:
            raise ValueError(
                f"Invalid batch shape: expected (N, 20, 224, 224, 3), got {frames_batch.shape}"
            )

        start_time = time.time()

        try:
            if self.use_tflite and self.interpreter is not None:
                # Process one by one for TFLite
                results = []
                for i in range(frames_batch.shape[0]):
                    single_batch = frames_batch[i:i+1].astype(np.float32)
                    self.interpreter.set_tensor(self.input_details[0]['index'], single_batch)
                    self.interpreter.invoke()
                    pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                    results.append(pred)
                predictions = np.array(results)
            else:
                predictions = self.keras_model.predict(frames_batch, verbose=0)

            inference_time = (time.time() - start_time) * 1000

            batch_results = []
            for pred in predictions:
                violence_prob = float(pred[1])
                batch_results.append({
                    "violence_probability": violence_prob,
                    "confidence": self._get_confidence(violence_prob),
                    "per_class_scores": {
                        "non_violence": float(pred[0]),
                        "violence": float(pred[1]),
                    },
                    "prediction": "violence" if violence_prob > 0.5 else "non_violence",
                })

            logger.info(f"Batch inference ({len(predictions)} samples) in {inference_time:.2f}ms")
            return batch_results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

    def _get_confidence(self, prob: float) -> str:
        """Get confidence level based on probability."""
        if prob > 0.9 or prob < 0.1:
            return "High"
        elif prob > 0.7 or prob < 0.3:
            return "Medium"
        else:
            return "Low"

    def get_model_info(self) -> Dict[str, any]:
        """Get model metadata and performance info."""
        info = {
            "model_path": str(self.model_path),
            "backend": "tflite" if self.use_tflite else "keras",
            "optimized": self.use_tflite,
            "num_threads": os.cpu_count() or 4,
        }

        if self.use_tflite and self.interpreter is not None:
            info["tflite_path"] = str(self.tflite_path)
            info["input_shape"] = self.input_details[0]['shape'].tolist()
            info["output_shape"] = self.output_details[0]['shape'].tolist()
        elif self.keras_model is not None:
            input_shape = [int(x) if x is not None else None for x in self.keras_model.input_shape]
            output_shape = [int(x) if x is not None else None for x in self.keras_model.output_shape]
            info["input_shape"] = input_shape
            info["output_shape"] = output_shape
            info["total_layers"] = len(self.keras_model.layers)
            info["trainable_params"] = int(sum([np.prod(v.shape) for v in self.keras_model.trainable_weights]))

        return info
