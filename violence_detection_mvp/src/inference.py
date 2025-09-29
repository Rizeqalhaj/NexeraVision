"""
Inference module for Violence Detection MVP.
Handles real-time prediction and video classification.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import time

import tensorflow as tf
from tensorflow.keras.models import Model

from .config import Config
from .model_architecture import ViolenceDetectionModel
from .feature_extraction import VGG19FeatureExtractor
from .data_preprocessing import VideoFrameExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViolencePredictor:
    """Real-time violence detection predictor."""

    def __init__(self, model_path: Path, config: Config = Config):
        """
        Initialize the violence predictor.

        Args:
            model_path: Path to the trained model
            config: Configuration object
        """
        self.config = config
        self.model_path = model_path
        self.model: Optional[Model] = None

        # Initialize components
        self.frame_extractor = VideoFrameExtractor(config)
        self.feature_extractor = VGG19FeatureExtractor(config)

        # Load model
        self._load_model()

        # Prediction settings
        self.confidence_threshold = 0.5
        self.class_names = ['Violence', 'No Violence']

        logger.info("Violence predictor initialized successfully")

    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            model_builder = ViolenceDetectionModel(self.config)
            self.model = model_builder.load_model(str(self.model_path))
            logger.info(f"Model loaded from: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict_video(
        self,
        video_path: Path,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Predict violence in a video file.

        Args:
            video_path: Path to the video file
            return_probabilities: Whether to return raw probabilities

        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        start_time = time.time()

        try:
            # Extract frames
            frames = self.frame_extractor.extract_frames(video_path)
            if frames is None:
                return {
                    "error": f"Failed to extract frames from {video_path}",
                    "video_path": str(video_path)
                }

            # Extract features
            features = self.feature_extractor.extract_features_from_frames(frames)
            if features is None:
                return {
                    "error": f"Failed to extract features from {video_path}",
                    "video_path": str(video_path)
                }

            # Reshape for model input (add batch dimension)
            features_reshaped = features.reshape(1, *features.shape)

            # Make prediction
            raw_prediction = self.model.predict(features_reshaped, verbose=0)[0]

            # Get predicted class
            predicted_class_idx = np.argmax(raw_prediction)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(raw_prediction[predicted_class_idx])

            # Determine if violence is detected
            violence_detected = predicted_class_idx == 0  # Violence is class 0

            prediction_time = time.time() - start_time

            result = {
                "video_path": str(video_path),
                "violence_detected": violence_detected,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "prediction_time_seconds": prediction_time,
                "timestamp": time.time()
            }

            if return_probabilities:
                result["probabilities"] = {
                    "Violence": float(raw_prediction[0]),
                    "No Violence": float(raw_prediction[1])
                }

            logger.info(f"Prediction: {predicted_class} ({confidence:.3f}) for {video_path.name}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed for {video_path}: {str(e)}")
            return {
                "error": str(e),
                "video_path": str(video_path)
            }

    def predict_video_batch(
        self,
        video_paths: List[Path],
        return_probabilities: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Predict violence for multiple videos.

        Args:
            video_paths: List of video file paths
            return_probabilities: Whether to return raw probabilities

        Returns:
            List of prediction results
        """
        results = []

        logger.info(f"Processing batch of {len(video_paths)} videos")

        for i, video_path in enumerate(video_paths):
            result = self.predict_video(video_path, return_probabilities)
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(video_paths)} videos")

        return results

    def predict_from_frames(
        self,
        frames: np.ndarray,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Predict violence from pre-extracted frames.

        Args:
            frames: Array of video frames
            return_probabilities: Whether to return raw probabilities

        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        start_time = time.time()

        try:
            # Extract features
            features = self.feature_extractor.extract_features_from_frames(frames)
            if features is None:
                return {"error": "Failed to extract features from frames"}

            # Reshape for model input
            features_reshaped = features.reshape(1, *features.shape)

            # Make prediction
            raw_prediction = self.model.predict(features_reshaped, verbose=0)[0]

            # Get predicted class
            predicted_class_idx = np.argmax(raw_prediction)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(raw_prediction[predicted_class_idx])

            # Determine if violence is detected
            violence_detected = predicted_class_idx == 0

            prediction_time = time.time() - start_time

            result = {
                "violence_detected": violence_detected,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "prediction_time_seconds": prediction_time,
                "timestamp": time.time()
            }

            if return_probabilities:
                result["probabilities"] = {
                    "Violence": float(raw_prediction[0]),
                    "No Violence": float(raw_prediction[1])
                }

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"error": str(e)}

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold for predictions.

        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold set to: {threshold}")


class RealTimeVideoProcessor:
    """Process video streams in real-time for violence detection."""

    def __init__(self, model_path: Path, config: Config = Config):
        """
        Initialize the real-time processor.

        Args:
            model_path: Path to the trained model
            config: Configuration object
        """
        self.config = config
        self.predictor = ViolencePredictor(model_path, config)
        self.frame_buffer: List[np.ndarray] = []
        self.buffer_size = config.FRAMES_PER_VIDEO

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single frame and return prediction when buffer is full.

        Args:
            frame: Video frame as numpy array

        Returns:
            Prediction result when buffer is full, None otherwise
        """
        # Resize and normalize frame
        resized_frame = cv2.resize(
            frame,
            self.config.IMG_SIZE_TUPLE,
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to RGB if necessary
        if len(resized_frame.shape) == 3 and resized_frame.shape[2] == 3:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Normalize
        normalized_frame = (resized_frame / 255.0).astype(np.float32)

        # Add to buffer
        self.frame_buffer.append(normalized_frame)

        # Check if buffer is full
        if len(self.frame_buffer) >= self.buffer_size:
            # Convert to numpy array
            frames_array = np.array(self.frame_buffer[-self.buffer_size:])

            # Make prediction
            result = self.predictor.predict_from_frames(frames_array)

            # Clear buffer (or slide window)
            self.frame_buffer = self.frame_buffer[self.buffer_size//2:]  # 50% overlap

            return result

        return None

    def process_video_stream(
        self,
        video_source: Union[str, int],
        output_callback: Optional[callable] = None,
        display: bool = False
    ) -> None:
        """
        Process a video stream in real-time.

        Args:
            video_source: Video source (file path or camera index)
            output_callback: Callback function for prediction results
            display: Whether to display video with predictions
        """
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            logger.error(f"Could not open video source: {video_source}")
            return

        logger.info(f"Starting real-time processing of: {video_source}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process frame
                result = self.process_frame(frame)

                if result is not None:
                    # Handle prediction result
                    if output_callback:
                        output_callback(result)

                    logger.info(
                        f"Frame {frame_count}: "
                        f"{result.get('predicted_class', 'Unknown')} "
                        f"({result.get('confidence', 0):.3f})"
                    )

                # Display frame if requested
                if display:
                    display_frame = frame.copy()

                    # Add prediction overlay if available
                    if result is not None:
                        violence_detected = result.get('violence_detected', False)
                        confidence = result.get('confidence', 0)

                        color = (0, 0, 255) if violence_detected else (0, 255, 0)
                        text = f"Violence: {violence_detected} ({confidence:.3f})"

                        cv2.putText(
                            display_frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
                        )

                    cv2.imshow('Violence Detection', display_frame)

                    # Break on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

            logger.info(f"Processed {frame_count} frames")


class InferenceAPI:
    """API wrapper for violence detection inference."""

    def __init__(self, model_path: Path, config: Config = Config):
        """
        Initialize the inference API.

        Args:
            model_path: Path to the trained model
            config: Configuration object
        """
        self.predictor = ViolencePredictor(model_path, config)
        self.config = config

    def predict_single_video(
        self,
        video_path: Union[str, Path],
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        API endpoint for single video prediction.

        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence for positive detection

        Returns:
            Prediction result with API-friendly format
        """
        video_path = Path(video_path)

        if not video_path.exists():
            return {
                "success": False,
                "error": f"Video file not found: {video_path}",
                "error_code": "FILE_NOT_FOUND"
            }

        # Set confidence threshold
        self.predictor.set_confidence_threshold(confidence_threshold)

        # Make prediction
        result = self.predictor.predict_video(video_path, return_probabilities=True)

        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "error_code": "PREDICTION_FAILED"
            }

        return {
            "success": True,
            "data": {
                "video_path": result["video_path"],
                "violence_detected": result["violence_detected"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "processing_time": result["prediction_time_seconds"]
            }
        }

    def predict_multiple_videos(
        self,
        video_paths: List[Union[str, Path]],
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        API endpoint for multiple video predictions.

        Args:
            video_paths: List of video file paths
            confidence_threshold: Minimum confidence for positive detection

        Returns:
            Batch prediction results
        """
        video_paths = [Path(p) for p in video_paths]

        # Validate paths
        missing_files = [p for p in video_paths if not p.exists()]
        if missing_files:
            return {
                "success": False,
                "error": f"Files not found: {[str(p) for p in missing_files]}",
                "error_code": "FILES_NOT_FOUND"
            }

        # Set confidence threshold
        self.predictor.set_confidence_threshold(confidence_threshold)

        # Make predictions
        results = self.predictor.predict_video_batch(video_paths, return_probabilities=True)

        # Process results
        successful_predictions = []
        failed_predictions = []

        for result in results:
            if "error" in result:
                failed_predictions.append(result)
            else:
                successful_predictions.append({
                    "video_path": result["video_path"],
                    "violence_detected": result["violence_detected"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "processing_time": result["prediction_time_seconds"]
                })

        return {
            "success": len(failed_predictions) == 0,
            "data": {
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "total_processed": len(results),
                "success_rate": len(successful_predictions) / len(results) if results else 0
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        return {
            "model_path": str(self.predictor.model_path),
            "model_type": "LSTM with Attention",
            "input_shape": [self.config.N_CHUNKS, self.config.CHUNK_SIZE],
            "num_classes": self.config.NUM_CLASSES,
            "class_names": self.predictor.class_names,
            "frames_per_video": self.config.FRAMES_PER_VIDEO,
            "image_size": self.config.IMG_SIZE_TUPLE
        }


def validate_inference_setup(model_path: Path, test_video_path: Path) -> Dict[str, Any]:
    """
    Validate that inference setup is working correctly.

    Args:
        model_path: Path to the trained model
        test_video_path: Path to a test video

    Returns:
        Validation results
    """
    validation_results = {
        "model_exists": False,
        "model_loadable": False,
        "test_video_exists": False,
        "prediction_successful": False,
        "errors": []
    }

    try:
        # Check model file
        if model_path.exists():
            validation_results["model_exists"] = True
        else:
            validation_results["errors"].append(f"Model file not found: {model_path}")
            return validation_results

        # Try loading model
        try:
            predictor = ViolencePredictor(model_path)
            validation_results["model_loadable"] = True
        except Exception as e:
            validation_results["errors"].append(f"Failed to load model: {str(e)}")
            return validation_results

        # Check test video
        if test_video_path.exists():
            validation_results["test_video_exists"] = True
        else:
            validation_results["errors"].append(f"Test video not found: {test_video_path}")
            return validation_results

        # Try prediction
        result = predictor.predict_video(test_video_path)
        if "error" not in result:
            validation_results["prediction_successful"] = True
            validation_results["test_prediction"] = result
        else:
            validation_results["errors"].append(f"Prediction failed: {result['error']}")

    except Exception as e:
        validation_results["errors"].append(f"Validation error: {str(e)}")

    return validation_results