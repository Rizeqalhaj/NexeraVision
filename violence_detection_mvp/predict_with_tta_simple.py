#!/usr/bin/env python3
"""
SIMPLE TEST-TIME AUGMENTATION - Minimal augmentations that preserve content
"""

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SimpleTTA:
    """Simple TTA - only flip + 2 minor brightness"""

    @staticmethod
    def augment_video_simple(frames: np.ndarray) -> list:
        """Create 3 augmented versions - conservative approach"""
        augmented_versions = []

        # 1. Original
        augmented_versions.append(frames.copy())

        # 2. Horizontal flip
        augmented_versions.append(np.flip(frames.copy(), axis=2))

        # 3. Slight brightness increase (helps with dark videos)
        aug = frames.astype(np.float32) * 1.1
        aug = np.clip(aug, 0, 255).astype(np.float32)
        augmented_versions.append(aug)

        return augmented_versions


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


def predict_with_simple_tta(model_path: str, dataset_path: str):
    """Predict with simple TTA (3x)"""

    logger.info("="*80)
    logger.info("SIMPLE TEST-TIME AUGMENTATION (3x)")
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
    logger.info("Using 3 augmentations: original + flip + brightness")

    # Predict with TTA
    all_predictions = []

    for i, video_path in enumerate(tqdm(test_videos, desc="TTA Prediction")):
        # Extract frames
        frames = extract_video_frames(video_path)

        if frames is None:
            # Failed video - use random prediction
            all_predictions.append(0)
            continue

        # Create 3 augmented versions
        augmented_versions = SimpleTTA.augment_video_simple(frames)

        # Predict on all 3 versions
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

        # Average predictions across all 3 versions
        avg_prediction = np.mean(version_predictions, axis=0)
        final_prediction = np.argmax(avg_prediction)
        all_predictions.append(final_prediction)

    # Calculate accuracy
    test_labels = np.array(test_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(test_labels, all_predictions)

    logger.info("\n" + "="*80)
    logger.info("SIMPLE TTA RESULTS (3x)")
    logger.info("="*80)
    logger.info(f"✅ ACCURACY WITH SIMPLE TTA: {accuracy*100:.2f}%")
    logger.info("="*80)

    print("\nClassification Report:")
    print(classification_report(test_labels, all_predictions,
                                target_names=['Non-violent', 'Violent']))

    cm = confusion_matrix(test_labels, all_predictions)
    nonviolent_acc = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    violent_acc = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0

    logger.info(f"\n✅ Non-violent Accuracy: {nonviolent_acc*100:.2f}%")
    logger.info(f"✅ Violent Accuracy: {violent_acc*100:.2f}%")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    args = parser.parse_args()

    predict_with_simple_tta(args.model, args.dataset)
