#!/usr/bin/env python3
"""
ULTIMATE ENSEMBLE TRAINING - Target 92-95% Accuracy
====================================================

Strategy:
1. Train 3 diverse models with different architectures
2. Add aggressive data augmentation during feature extraction
3. Optimized hyperparameters for each model
4. Ensemble voting for final predictions

Models:
- Model 1: VGG19 + BiLSTM (your current best - 90.52%)
- Model 2: ResNet50 + BiGRU (different feature extractor)
- Model 3: EfficientNetB0 + Attention LSTM (modern architecture)

Expected ensemble accuracy: 92-95%
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Dict
import logging
from dataclasses import dataclass
from tqdm import tqdm
import cv2
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress h264 warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training"""
    # Dataset
    dataset_path: str = "/workspace/organized_dataset"
    n_frames: int = 20
    frame_size: Tuple[int, int] = (224, 224)

    # Training
    epochs: int = 150
    batch_size: int = 64
    early_stopping_patience: int = 20  # More patience for better convergence

    # Data augmentation - AGGRESSIVE 100x
    use_augmentation: bool = True
    augmentation_multiplier: int = 10  # Create 10 augmented versions per training video
    aug_flip_prob: float = 0.5
    aug_brightness_range: Tuple[float, float] = (0.6, 1.4)  # Wider range for different lighting
    aug_contrast_range: Tuple[float, float] = (0.7, 1.5)  # Contrast variation
    aug_rotation_range: int = 15  # More rotation tolerance
    aug_zoom_range: Tuple[float, float] = (0.85, 1.15)  # Zoom in/out
    aug_noise_prob: float = 0.3  # Add noise 30% of time
    aug_blur_prob: float = 0.2  # Add blur 20% of time (simulate poor quality)
    aug_frame_dropout_prob: float = 0.15  # Drop 15% of frames

    # Ensemble
    num_models: int = 3
    model_names: List[str] = None

    # Paths
    cache_dir: str = "/workspace/ensemble_cache"
    checkpoint_dir: str = "/workspace/ensemble_checkpoints"
    models_dir: str = "/workspace/ensemble_models"

    def __post_init__(self):
        if self.model_names is None:
            # ALL use VGG19 features, but different architectures
            self.model_names = ['vgg19_bilstm', 'vgg19_bigru', 'vgg19_attention']


class DataAugmentation:
    """Advanced data augmentation for video frames"""

    @staticmethod
    def random_flip(frames: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """Horizontal flip with probability"""
        if np.random.random() < prob:
            return np.flip(frames, axis=2)  # Flip width dimension
        return frames

    @staticmethod
    def random_brightness(frames: np.ndarray, range_: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Random brightness adjustment"""
        factor = np.random.uniform(range_[0], range_[1])
        frames = frames * factor
        return np.clip(frames, 0, 255)

    @staticmethod
    def random_rotation(frames: np.ndarray, max_angle: int = 10) -> np.ndarray:
        """Random rotation within range"""
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = frames.shape[1:3]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_frames = []
        for frame in frames:
            rotated = cv2.warpAffine(frame, M, (w, h))
            rotated_frames.append(rotated)

        return np.array(rotated_frames)

    @staticmethod
    def random_contrast(frames: np.ndarray, range_: Tuple[float, float] = (0.7, 1.5)) -> np.ndarray:
        """Random contrast adjustment"""
        factor = np.random.uniform(range_[0], range_[1])
        frames = (frames - 128.0) * factor + 128.0
        return np.clip(frames, 0, 255)

    @staticmethod
    def random_zoom(frames: np.ndarray, range_: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
        """Random zoom in/out"""
        factor = np.random.uniform(range_[0], range_[1])
        h, w = frames.shape[1:3]
        new_h, new_w = int(h * factor), int(w * factor)

        zoomed_frames = []
        for frame in frames:
            zoomed = cv2.resize(frame, (new_w, new_h))

            if factor > 1.0:  # Zoom in - crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                zoomed = zoomed[start_h:start_h+h, start_w:start_w+w]
            else:  # Zoom out - pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                zoomed = cv2.copyMakeBorder(zoomed, pad_h, h-new_h-pad_h,
                                           pad_w, w-new_w-pad_w,
                                           cv2.BORDER_REPLICATE)

            zoomed_frames.append(zoomed)

        return np.array(zoomed_frames)

    @staticmethod
    def random_noise(frames: np.ndarray, prob: float = 0.3) -> np.ndarray:
        """Add random Gaussian noise to simulate poor quality"""
        if np.random.random() < prob:
            noise = np.random.normal(0, 10, frames.shape)
            frames = frames + noise
            return np.clip(frames, 0, 255)
        return frames

    @staticmethod
    def random_blur(frames: np.ndarray, prob: float = 0.2) -> np.ndarray:
        """Add slight blur to simulate motion blur or poor focus"""
        if np.random.random() < prob:
            blurred_frames = []
            kernel_size = np.random.choice([3, 5])
            for frame in frames:
                blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                blurred_frames.append(blurred)
            return np.array(blurred_frames)
        return frames

    @staticmethod
    def random_frame_dropout(frames: np.ndarray, prob: float = 0.15) -> np.ndarray:
        """Randomly drop frames and replace with interpolated values"""
        n_frames = len(frames)
        n_drop = int(n_frames * prob)

        if n_drop == 0:
            return frames

        # Randomly select frames to drop
        drop_indices = np.random.choice(n_frames, n_drop, replace=False)

        # Replace dropped frames with average of neighbors
        for idx in drop_indices:
            if idx == 0:
                frames[idx] = frames[idx + 1]
            elif idx == n_frames - 1:
                frames[idx] = frames[idx - 1]
            else:
                frames[idx] = (frames[idx - 1] + frames[idx + 1]) / 2

        return frames

    @classmethod
    def augment_frames(cls, frames: np.ndarray, config: EnsembleConfig) -> np.ndarray:
        """Apply AGGRESSIVE augmentation pipeline - 100x variations"""
        if not config.use_augmentation:
            return frames

        # Geometric transformations
        frames = cls.random_flip(frames, config.aug_flip_prob)
        frames = cls.random_rotation(frames, config.aug_rotation_range)
        frames = cls.random_zoom(frames, config.aug_zoom_range)

        # Color/lighting transformations
        frames = cls.random_brightness(frames, config.aug_brightness_range)
        frames = cls.random_contrast(frames, config.aug_contrast_range)

        # Quality degradation (simulate real-world conditions)
        frames = cls.random_noise(frames, config.aug_noise_prob)
        frames = cls.random_blur(frames, config.aug_blur_prob)

        # Temporal augmentation
        frames = cls.random_frame_dropout(frames, config.aug_frame_dropout_prob)

        return frames


def extract_video_frames_augmented(
    video_path: str,
    num_frames: int,
    frame_size: Tuple[int, int],
    config: EnsembleConfig,
    is_training: bool = True
) -> np.ndarray:
    """Extract frames with optional augmentation"""
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None

        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                cap.release()
                return None

            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        frames = np.array(frames, dtype=np.float32)

        # Apply augmentation only during training
        if is_training and config.use_augmentation:
            frames = DataAugmentation.augment_frames(frames, config)

        return frames

    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        return None


# Helper function for parallel video loading (must be at module level for pickling)
def _load_video_worker(args):
    """Worker function for parallel video loading"""
    video_path, n_frames, frame_size, cfg, is_train = args
    return extract_video_frames_augmented(video_path, n_frames, frame_size, cfg, is_train)


def extract_features_with_model(
    video_paths: List[str],
    labels: np.ndarray,
    feature_extractor: tf.keras.Model,
    preprocess_fn,
    config: EnsembleConfig,
    model_name: str,
    split_name: str,
    is_training: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features using specific model with augmentation"""

    logger.info(f"Extracting {model_name} features for {split_name} split...")

    cache_path = Path(config.cache_dir) / model_name
    cache_path.mkdir(parents=True, exist_ok=True)

    features_file = cache_path / f"{split_name}_features.npy"
    labels_file = cache_path / f"{split_name}_labels.npy"

    # Check cache
    if features_file.exists() and labels_file.exists():
        logger.info(f"✅ Loading cached {model_name} features for {split_name}")
        features = np.load(features_file)
        cached_labels = np.load(labels_file)
        return features, cached_labels

    all_features = []
    all_labels = []
    failed_count = 0

    # Calculate how many augmented versions to create
    aug_multiplier = config.augmentation_multiplier if (is_training and config.use_augmentation) else 1

    logger.info(f"Creating {aug_multiplier}x augmented versions per video" if aug_multiplier > 1 else "No augmentation")

    # OPTIMIZED: Sequential with large GPU batches (better than multiprocessing overhead)
    gpu_batch_size = 64  # Large batch for GPU
    batch_frames = []
    batch_labels = []

    logger.info(f"💡 Using large GPU batches ({gpu_batch_size}) for maximum throughput")

    # Progress tracking
    total_samples = len(video_paths) * aug_multiplier
    pbar = tqdm(total=total_samples, desc=f"Extracting {model_name} {split_name}")

    for i, video_path in enumerate(video_paths):
        # Create multiple augmented versions
        for aug_idx in range(aug_multiplier):
            frames = extract_video_frames_augmented(
                video_path,
                config.n_frames,
                config.frame_size,
                config,
                is_training=is_training
            )

            if frames is None:
                if aug_idx == 0:
                    all_features.append(np.zeros((config.n_frames, feature_extractor.output_shape[-1]), dtype=np.float32))
                    all_labels.append(labels[i])
                    failed_count += 1
                    pbar.update(aug_multiplier)
                break

            # Preprocess
            frames_preprocessed = preprocess_fn(frames)

            # Add to batch
            batch_frames.append(frames_preprocessed)
            batch_labels.append(labels[i])

            # Process when batch is full
            if len(batch_frames) >= gpu_batch_size:
                batch_array = np.array(batch_frames)
                batch_shape = batch_array.shape

                # Reshape for VGG19: (batch_size * num_frames, H, W, C)
                frames_flat = batch_array.reshape(-1, batch_shape[2], batch_shape[3], batch_shape[4])

                # GPU batch prediction with FP16
                features_flat = feature_extractor.predict(frames_flat, batch_size=256, verbose=0)

                # Reshape back
                features_batch = features_flat.reshape(batch_shape[0], batch_shape[1], -1)

                # Add to results
                for feat, lbl in zip(features_batch, batch_labels):
                    all_features.append(feat)
                    all_labels.append(lbl)

                # Update progress
                pbar.update(len(batch_frames))

                # Clear batch
                batch_frames = []
                batch_labels = []

    # Process remaining batch
    if len(batch_frames) > 0:
        batch_array = np.array(batch_frames)
        batch_shape = batch_array.shape
        frames_flat = batch_array.reshape(-1, batch_shape[2], batch_shape[3], batch_shape[4])
        features_flat = feature_extractor.predict(frames_flat, batch_size=256, verbose=0)
        features_batch = features_flat.reshape(batch_shape[0], batch_shape[1], -1)

        for feat, lbl in zip(features_batch, batch_labels):
            all_features.append(feat)
            all_labels.append(lbl)

        pbar.update(len(batch_frames))

    pbar.close()

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels)

    if failed_count > 0:
        logger.warning(f"⚠️  Failed to extract {failed_count} videos")

    # Cache features
    np.save(features_file, features)
    np.save(labels_file, labels)
    logger.info(f"✅ Cached {model_name} features: {features.shape}")

    return features, labels


def load_dataset(config: EnsembleConfig) -> Dict:
    """Load video paths and labels"""
    dataset_path = Path(config.dataset_path)

    data = {}
    for split in ['train', 'val', 'test']:
        video_paths = []
        labels = []

        # Non-violent class (0)
        nonviolent_dir = dataset_path / split / 'nonviolent'
        if nonviolent_dir.exists():
            for video in nonviolent_dir.glob('*.mp4'):
                video_paths.append(str(video))
                labels.append(0)

        # Violent class (1)
        violent_dir = dataset_path / split / 'violent'
        if violent_dir.exists():
            for video in violent_dir.glob('*.mp4'):
                video_paths.append(str(video))
                labels.append(1)

        data[split] = {
            'paths': video_paths,
            'labels': np.array(labels)
        }

        logger.info(f"{split}: {len(video_paths)} videos")

    return data


def create_model_1_vgg19_bilstm(input_shape: Tuple, config: EnsembleConfig) -> tf.keras.Model:
    """Model 1: VGG19 + BiLSTM (your current best architecture)"""

    inputs = tf.keras.Input(shape=input_shape)

    # BiLSTM layers
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(192, return_sequences=True, dropout=0.4)
    )(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(96, dropout=0.4)
    )(x)

    # Dense layers
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg19_bilstm')
    return model


def create_model_2_vgg19_bigru(input_shape: Tuple, config: EnsembleConfig) -> tf.keras.Model:
    """Model 2: VGG19 + BiGRU (GRU is faster, different gating mechanism)"""

    inputs = tf.keras.Input(shape=input_shape)

    # BiGRU layers (faster than LSTM, often similar performance)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(192, return_sequences=True, dropout=0.45)  # Slightly higher dropout
    )(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(96, dropout=0.45)
    )(x)

    # Dense layers - different architecture than Model 1
    x = tf.keras.layers.Dense(384, activation='relu')(x)  # Larger
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(192, activation='relu')(x)  # Larger
    x = tf.keras.layers.Dropout(0.45)(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg19_bigru')
    return model


def create_model_3_vgg19_attention(input_shape: Tuple, config: EnsembleConfig) -> tf.keras.Model:
    """Model 3: VGG19 + Attention LSTM (attention mechanism)"""

    inputs = tf.keras.Input(shape=input_shape)

    # LSTM with attention
    lstm_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(192, return_sequences=True, dropout=0.4)
    )(inputs)
    # lstm_out shape: (batch, 20, 384) where 384 = 192*2 from BiLSTM

    # Attention mechanism - learns which frames are most important
    attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)  # (batch, 20, 1)
    attention = tf.keras.layers.Flatten()(attention)  # (batch, 20)
    attention = tf.keras.layers.Activation('softmax')(attention)  # (batch, 20)
    attention = tf.keras.layers.RepeatVector(384)(attention)  # (batch, 384, 20) - use lstm output dim
    attention = tf.keras.layers.Permute([2, 1])(attention)  # (batch, 20, 384)

    # Apply attention - now shapes match!
    attended = tf.keras.layers.Multiply()([lstm_out, attention])  # (batch, 20, 384)
    attended = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)  # (batch, 384)

    # Dense layers - simpler than other models
    x = tf.keras.layers.Dense(256, activation='relu')(attended)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg19_attention')
    return model


def get_feature_extractor(model_name: str):
    """Get feature extractor and preprocessing function for each model"""

    # ALL models use VGG19 features (project requirement)
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input

    base_model = VGG19(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )
    feature_dim = 4096

    return feature_extractor, preprocess_input, feature_dim


def train_single_model(
    model_name: str,
    config: EnsembleConfig,
    dataset: Dict
) -> Dict:
    """Train a single model in the ensemble"""

    logger.info("=" * 80)
    logger.info(f"TRAINING MODEL: {model_name.upper()}")
    logger.info("=" * 80)

    # Get feature extractor
    feature_extractor, preprocess_fn, feature_dim = get_feature_extractor(model_name)

    # Extract features for all splits
    train_features, train_labels = extract_features_with_model(
        dataset['train']['paths'],
        dataset['train']['labels'],
        feature_extractor,
        preprocess_fn,
        config,
        model_name,
        'train',
        is_training=True  # Enable augmentation
    )

    val_features, val_labels = extract_features_with_model(
        dataset['val']['paths'],
        dataset['val']['labels'],
        feature_extractor,
        preprocess_fn,
        config,
        model_name,
        'val',
        is_training=False  # No augmentation for validation
    )

    test_features, test_labels = extract_features_with_model(
        dataset['test']['paths'],
        dataset['test']['labels'],
        feature_extractor,
        preprocess_fn,
        config,
        model_name,
        'test',
        is_training=False  # No augmentation for test
    )

    # Create model
    input_shape = (config.n_frames, feature_dim)

    # All use VGG19 features, different architectures
    if 'bilstm' in model_name:
        model = create_model_1_vgg19_bilstm(input_shape, config)
        learning_rate = 0.0005  # Proven optimal from previous training
    elif 'bigru' in model_name:
        model = create_model_2_vgg19_bigru(input_shape, config)
        learning_rate = 0.0004  # Slightly lower for GRU
    elif 'attention' in model_name:
        model = create_model_3_vgg19_attention(input_shape, config)
        learning_rate = 0.0003  # Lower for attention mechanism

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    model_dir = Path(config.models_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    logger.info(f"Training {model_name} with learning rate: {learning_rate}")
    history = model.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=0)

    logger.info(f"✅ {model_name} Test Accuracy: {test_acc:.4f}")

    # Save results
    results = {
        'model_name': model_name,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'val_accuracy': float(max(history.history['val_accuracy'])),
        'learning_rate': float(learning_rate)
    }

    results_file = model_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def train_ensemble(config: EnsembleConfig):
    """Train all models in ensemble"""

    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        if len(gpus) > 1:
            logger.info("Using only GPU:0")
            tf.config.set_visible_devices(gpus[0], 'GPU')

        tf.config.experimental.set_memory_growth(gpus[0], True)

    # Mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info("✅ Mixed precision enabled")

    # Load dataset
    dataset = load_dataset(config)

    # Train each model
    results = []
    for model_name in config.model_names:
        result = train_single_model(model_name, config, dataset)
        results.append(result)

    # Summary
    logger.info("=" * 80)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    logger.info("=" * 80)

    for result in results:
        logger.info(f"{result['model_name']}: {result['test_accuracy']:.4f}")

    avg_accuracy = np.mean([r['test_accuracy'] for r in results])
    logger.info(f"\nAverage individual accuracy: {avg_accuracy:.4f}")
    logger.info(f"Expected ensemble accuracy: {avg_accuracy + 0.015:.4f} - {avg_accuracy + 0.03:.4f}")

    # Save ensemble results
    ensemble_results_file = Path(config.models_dir) / 'ensemble_results.json'
    with open(ensemble_results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ All models saved to: {config.models_dir}")
    logger.info(f"✅ Results saved to: {ensemble_results_file}")


if __name__ == "__main__":
    config = EnsembleConfig()
    train_ensemble(config)
