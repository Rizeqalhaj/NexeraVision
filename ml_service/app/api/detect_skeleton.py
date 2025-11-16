"""
Skeleton-based violence detection API endpoints.

Provides pose estimation and skeleton-based violence detection endpoints.
"""
import base64
import logging
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field

from app.mediapipe_detector import SkeletonDetector, SkeletonViolenceClassifier, PoseFeatureExtractor
from app.models.violence_detector import ViolenceDetector
from app.mediapipe_detector.violence_classifier import EnsembleViolenceDetector
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances
skeleton_detector: Optional[SkeletonDetector] = None
skeleton_classifier: Optional[SkeletonViolenceClassifier] = None
ensemble_detector: Optional[EnsembleViolenceDetector] = None


def initialize_skeleton_detector():
    """Initialize skeleton detection components."""
    global skeleton_detector, skeleton_classifier

    if skeleton_detector is None:
        skeleton_detector = SkeletonDetector()
        logger.info("Skeleton detector initialized")

    if skeleton_classifier is None:
        skeleton_classifier = SkeletonViolenceClassifier()
        logger.info("Skeleton violence classifier initialized")


def initialize_ensemble_detector(cnn_detector: ViolenceDetector):
    """Initialize ensemble detector with CNN model."""
    global ensemble_detector

    if ensemble_detector is None:
        ensemble_detector = EnsembleViolenceDetector(
            cnn_detector=cnn_detector,
            skeleton_weight=0.3  # 70% CNN, 30% skeleton
        )
        logger.info("Ensemble detector initialized")


class PoseDetectionResponse(BaseModel):
    """Response model for pose detection."""
    success: bool
    pose_detected: bool
    landmarks: Optional[List[dict]] = None
    mean_visibility: Optional[float] = None
    visualization: Optional[str] = None


class SkeletonPredictionResponse(BaseModel):
    """Response model for skeleton-based violence prediction."""
    success: bool
    violence_probability: float
    confidence: str
    prediction: str
    pose_detected: bool
    patterns: Optional[dict] = None
    features: Optional[dict] = None


class EnsemblePredictionResponse(BaseModel):
    """Response model for ensemble prediction."""
    success: bool
    violence_probability: float
    confidence: str
    prediction: str
    cnn_probability: float
    skeleton_probability: float
    pose_detected: bool
    per_class_scores: dict


@router.post("/detect_pose", response_model=PoseDetectionResponse)
async def detect_pose(
    image: UploadFile = File(...),
    visualize: bool = Form(False)
):
    """
    Detect pose landmarks in image.

    Extracts 33 body keypoints using MediaPipe Pose.
    """
    initialize_skeleton_detector()

    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect pose
        pose_data = skeleton_detector.detect_pose(frame)

        response = {
            "success": True,
            "pose_detected": pose_data is not None
        }

        if pose_data:
            response["landmarks"] = pose_data['landmarks']
            response["mean_visibility"] = pose_data['mean_visibility']

            # Add visualization if requested
            if visualize:
                vis_frame = skeleton_detector.visualize_pose(frame, pose_data)
                _, buffer = cv2.imencode('.jpg', vis_frame)
                vis_base64 = base64.b64encode(buffer).decode('utf-8')
                response["visualization"] = vis_base64

        return response

    except Exception as e:
        logger.error(f"Pose detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect_skeleton", response_model=SkeletonPredictionResponse)
async def detect_violence_skeleton(
    image: UploadFile = File(...)
):
    """
    Detect violence using skeleton-based analysis.

    Analyzes pose patterns to detect violent actions.
    """
    initialize_skeleton_detector()

    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Predict violence
        prediction = skeleton_classifier.predict(frame)

        return {
            "success": True,
            **prediction
        }

    except Exception as e:
        logger.error(f"Skeleton violence detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect_ensemble", response_model=EnsemblePredictionResponse)
async def detect_violence_ensemble(
    video: UploadFile = File(...)
):
    """
    Detect violence using ensemble of CNN and skeleton models.

    Combines VGG19 CNN (70%) with skeleton-based detection (30%)
    for improved accuracy.
    """
    # Import here to avoid circular dependency
    from app.api.detect import detector as cnn_detector

    if cnn_detector is None:
        raise HTTPException(status_code=503, detail="CNN detector not initialized")

    initialize_skeleton_detector()
    initialize_ensemble_detector(cnn_detector)

    try:
        # Read video
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            contents = await video.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        # Extract frames
        from app.utils.frame_extraction import extract_frames

        frames = extract_frames(tmp_path, num_frames=20, target_size=(224, 224))

        # Cleanup temp file
        Path(tmp_path).unlink()

        if frames is None or len(frames) == 0:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # Ensemble prediction
        prediction = ensemble_detector.predict(frames)

        return {
            "success": True,
            **prediction
        }

    except Exception as e:
        logger.error(f"Ensemble detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract_features")
async def extract_pose_features(
    image: UploadFile = File(...)
):
    """
    Extract pose features for violence detection.

    Returns detailed features including joint angles, movement patterns,
    and violence indicators.
    """
    initialize_skeleton_detector()

    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect pose
        pose_data = skeleton_detector.detect_pose(frame)

        if pose_data is None:
            return {
                "success": True,
                "pose_detected": False,
                "features": {},
                "patterns": {}
            }

        # Extract features
        feature_extractor = PoseFeatureExtractor()
        features = feature_extractor.extract_features(pose_data)
        patterns = feature_extractor.detect_violent_patterns(features)

        return {
            "success": True,
            "pose_detected": True,
            "features": features,
            "patterns": patterns
        }

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skeleton/health")
async def skeleton_health_check():
    """Health check for skeleton detection service."""
    initialize_skeleton_detector()

    return {
        "status": "operational",
        "service": "skeleton_detection",
        "components": {
            "skeleton_detector": skeleton_detector is not None,
            "skeleton_classifier": skeleton_classifier is not None,
            "ensemble_detector": ensemble_detector is not None
        },
        "features": {
            "pose_estimation": True,
            "violence_classification": True,
            "ensemble_prediction": True,
            "feature_extraction": True
        }
    }


@router.post("/skeleton/visualize")
async def visualize_skeleton_prediction(
    image: UploadFile = File(...)
):
    """
    Visualize skeleton detection with violence prediction overlay.

    Returns annotated image with skeleton and violence indicators.
    """
    initialize_skeleton_detector()

    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Predict violence
        prediction = skeleton_classifier.predict(frame)

        # Visualize
        annotated = skeleton_classifier.visualize_prediction(frame, prediction)

        # Encode as base64
        _, buffer = cv2.imencode('.jpg', annotated)
        vis_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "visualization": vis_base64,
            "prediction": prediction
        }

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
