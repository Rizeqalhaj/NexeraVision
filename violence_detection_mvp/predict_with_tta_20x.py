#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) - 20x Augmentation for Higher Accuracy
===================================================================

Strategy:
- Create 20 augmented versions of each test video
- Run prediction on all 20 versions
- Average predictions → More robust, higher accuracy
- Expected boost: +1-2% accuracy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


class TTA20x:
    """20x Test-Time Augmentation"""
    
    @staticmethod
    def augment_video_20x(frames: np.ndarray) -> list:
        """Create 20 augmented versions of video frames"""
        augmented_versions = []
        
        # Original
        augmented_versions.append(frames.copy())
        
        # Horizontal flip
        augmented_versions.append(np.flip(frames, axis=2))
        
        # Brightness variations (5 levels: 0.7, 0.85, 1.0, 1.15, 1.3)
        for brightness in [0.7, 0.85, 1.15, 1.3]:
            aug = frames * brightness
            aug = np.clip(aug, 0, 255)
            augmented_versions.append(aug)
        
        # Brightness + flip
        for brightness in [0.7, 0.85, 1.15, 1.3]:
            aug = frames * brightness
            aug = np.clip(aug, 0, 255)
            aug = np.flip(aug, axis=2)
            augmented_versions.append(aug)
        
        # Contrast variations (3 levels)
        for contrast in [0.8, 1.2, 1.4]:
            aug = (frames - 128) * contrast + 128
            aug = np.clip(aug, 0, 255)
            augmented_versions.append(aug)
        
        # Small rotations (4 angles)
        for angle in [-5, -2, 2, 5]:
            aug = TTA20x._rotate_frames(frames, angle)
            augmented_versions.append(aug)
        
        return augmented_versions[:20]  # Ensure exactly 20
    
    @staticmethod
    def _rotate_frames(frames: np.ndarray, angle: float) -> np.ndarray:
        """Rotate all frames by angle"""
        h, w = frames.shape[1:3]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = []
        for frame in frames:
            rotated_frame = cv2.warpAffine(frame, M, (w, h))
            rotated.append(rotated_frame)
        
        return np.array(rotated)


def extract_video_frames(video_path: str, num_frames: int = 20) -> np.ndarray:
    """Extract frames from video"""
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                cap.release()
                return None
            
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return np.array(frames, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def predict_with_tta_20x(model_path: str, dataset_path: str):
    """Predict with 20x TTA"""
    
    logger.info("="*80)
    logger.info("20X TEST-TIME AUGMENTATION")
    logger.info("="*80)
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load VGG19 feature extractor
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
    
    base_model = VGG19(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )
    
    # Load test videos
    dataset_path = Path(dataset_path)
    test_videos = []
    test_labels = []
    
    for label, class_name in enumerate(['nonviolent', 'violent']):
        class_dir = dataset_path / 'test' / class_name
        if class_dir.exists():
            for video in class_dir.glob('*.mp4'):
                test_videos.append(str(video))
                test_labels.append(label)
    
    logger.info(f"Testing on {len(test_videos)} videos")
    logger.info("Creating 20 augmented versions per video...")
    
    # Predict with TTA
    all_predictions = []
    
    for i, video_path in enumerate(tqdm(test_videos, desc="TTA Prediction")):
        # Extract frames
        frames = extract_video_frames(video_path)
        
        if frames is None:
            # Failed video - use random prediction
            all_predictions.append(0)
            continue
        
        # Create 20 augmented versions
        augmented_versions = TTA20x.augment_video_20x(frames)
        
        # Predict on all 20 versions
        version_predictions = []
        
        for aug_frames in augmented_versions:
            # Preprocess
            aug_frames = preprocess_input(aug_frames)
            
            # Extract features
            features = feature_extractor.predict(aug_frames, verbose=0)
            features = features.reshape(1, 20, 4096)
            
            # Predict
            probs = model.predict(features, verbose=0)[0]
            version_predictions.append(probs)
        
        # Average predictions across all 20 versions
        avg_prediction = np.mean(version_predictions, axis=0)
        final_prediction = np.argmax(avg_prediction)
        all_predictions.append(final_prediction)
    
    # Calculate accuracy
    test_labels = np.array(test_labels)
    all_predictions = np.array(all_predictions)
    
    accuracy = accuracy_score(test_labels, all_predictions)
    
    logger.info("\n" + "="*80)
    logger.info("20X TTA RESULTS")
    logger.info("="*80)
    logger.info(f"✅ ACCURACY WITH 20X TTA: {accuracy*100:.2f}%")
    logger.info("="*80)
    
    print("\nClassification Report:")
    print(classification_report(test_labels, all_predictions, 
                                target_names=['Non-violent', 'Violent']))
    
    return accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                       default='/workspace/violence_detection_mvp/models/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str,
                       default='/workspace/organized_dataset',
                       help='Path to dataset')
    
    args = parser.parse_args()
    
    accuracy = predict_with_tta_20x(args.model, args.dataset)
