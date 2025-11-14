#!/usr/bin/env python3
"""
Multi-GPU Training Script for 2x RTX 4080
Uses TensorFlow MirroredStrategy for data parallelism
"""

import os
import sys

# GPU Configuration - Enable both GPUs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# GPU Setup
print("=" * 80)
print("MULTI-GPU CONFIGURATION")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        print(f"✅ Mixed Precision Training: ENABLED (float16)")
        print(f"✅ Multi-GPU Strategy: MirroredStrategy")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️ No GPU detected - will use CPU")

print("=" * 80)

# Import configuration
from src.config import (
    SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    NUM_CLASSES, RNN_SIZE, DROPOUT_RATE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Multi-GPU batch size (automatically scaled)
GLOBAL_BATCH_SIZE = BATCH_SIZE * len(gpus) if gpus else BATCH_SIZE
logger.info(f"Batch size per GPU: {BATCH_SIZE}")
logger.info(f"Global batch size: {GLOBAL_BATCH_SIZE}")


def extract_features_with_augmentation(
    data_dir: str,
    subset: str = 'train',
    max_videos: int = None,
    use_augmentation: bool = True
):
    """
    Extract VGG19 features using first GPU only (feature extraction doesn't parallelize well).

    Returns:
        X: Feature array with shape (num_videos, SEQUENCE_LENGTH, 4096)
        y: Label array with shape (num_videos,)
    """
    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import cv2

    logger.info(f"Extracting features for {subset} set...")
    logger.info(f"Using GPU:0 for feature extraction")

    # Initialize VGG19 on first GPU only
    with tf.device('/GPU:0'):
        vgg19 = VGG19(
            weights='imagenet',
            include_top=True,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
        feature_extractor = tf.keras.Model(
            inputs=vgg19.input,
            outputs=vgg19.get_layer('fc2').output
        )

    logger.info("✅ VGG19 feature extractor loaded on GPU:0")

    # Data augmentation
    if use_augmentation and subset == 'train':
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator()

    # Get video paths
    data_path = Path(data_dir) / subset
    violence_videos = sorted(list((data_path / 'Fight').glob('*.avi')))
    nonviolence_videos = sorted(list((data_path / 'NonFight').glob('*.avi')))

    if max_videos:
        violence_videos = violence_videos[:max_videos//2]
        nonviolence_videos = nonviolence_videos[:max_videos//2]

    all_videos = violence_videos + nonviolence_videos
    labels = [1] * len(violence_videos) + [0] * len(nonviolence_videos)

    logger.info(f"Processing {len(all_videos)} videos ({len(violence_videos)} violence, {len(nonviolence_videos)} non-violence)")

    # Extract features
    features_list = []
    labels_list = []

    for idx, (video_path, label) in enumerate(zip(all_videos, labels)):
        try:
            # Extract frames
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.warning(f"Skipping {video_path.name}: no frames")
                cap.release()
                continue

            frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) != SEQUENCE_LENGTH:
                logger.warning(f"Skipping {video_path.name}: only got {len(frames)}/{SEQUENCE_LENGTH} frames")
                continue

            # Convert to numpy array
            frames = np.array(frames, dtype=np.uint8)

            # Apply augmentation if enabled
            if use_augmentation and subset == 'train':
                augmented_frames = []
                for frame in frames:
                    frame = frame.reshape((1,) + frame.shape)
                    aug_iter = datagen.flow(frame, batch_size=1)
                    augmented_frame = next(aug_iter)[0].astype('uint8')
                    augmented_frames.append(augmented_frame)
                frames = np.array(augmented_frames)

            # Preprocess for VGG19
            frames_preprocessed = preprocess_input(frames.astype('float32'))

            # Extract features using GPU:0
            with tf.device('/GPU:0'):
                video_features = feature_extractor.predict(
                    frames_preprocessed,
                    batch_size=SEQUENCE_LENGTH,
                    verbose=0
                )

            # Verify shape
            assert video_features.shape == (SEQUENCE_LENGTH, 4096), \
                f"Expected shape ({SEQUENCE_LENGTH}, 4096), got {video_features.shape}"

            features_list.append(video_features)
            labels_list.append(label)

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(all_videos)} videos")

        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            continue

    # Convert to numpy arrays
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)

    # Verify shapes
    logger.info(f"✅ Feature extraction complete:")
    logger.info(f"   X.shape = {X.shape} (expected: (num_videos, {SEQUENCE_LENGTH}, 4096))")
    logger.info(f"   y.shape = {y.shape} (expected: (num_videos,))")

    assert len(X.shape) == 3, f"X should be 3D, got shape {X.shape}"
    assert X.shape[1] == SEQUENCE_LENGTH, f"Sequence length mismatch"
    assert X.shape[2] == 4096, f"Feature dimension mismatch"

    return X, y


def build_lstm_model(input_shape, num_classes=2):
    """
    Build LSTM model for multi-GPU training.

    Args:
        input_shape: Tuple (sequence_length, features) = (20, 4096)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    logger.info(f"Building LSTM model with input_shape={input_shape}")

    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=input_shape),

        # LSTM layers (CuDNN optimized on GPU)
        tf.keras.layers.LSTM(
            RNN_SIZE,
            return_sequences=True,
            dropout=DROPOUT_RATE,
            name='lstm_1'
        ),
        tf.keras.layers.LSTM(
            RNN_SIZE // 2,
            return_sequences=False,
            dropout=DROPOUT_RATE,
            name='lstm_2'
        ),

        # Dense layers
        tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(DROPOUT_RATE, name='dropout_1'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(DROPOUT_RATE, name='dropout_2'),

        # Output layer (float32 for numerical stability)
        tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("✅ Model built successfully")
    return model


def train_multi_gpu(
    data_dir: str,
    output_dir: str = './models',
    max_train_videos: int = None,
    max_val_videos: int = None
):
    """
    Main training function optimized for multi-GPU (2x RTX 4080).
    Uses MirroredStrategy for data parallelism.
    """
    logger.info("=" * 80)
    logger.info("STARTING MULTI-GPU TRAINING")
    logger.info("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Phase 1: Extract training features (single GPU)
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: FEATURE EXTRACTION - TRAINING SET")
    logger.info("=" * 80)

    X_train, y_train = extract_features_with_augmentation(
        data_dir=data_dir,
        subset='train',
        max_videos=max_train_videos,
        use_augmentation=True
    )

    logger.info(f"Training data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    # Phase 2: Extract validation features (single GPU)
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: FEATURE EXTRACTION - VALIDATION SET")
    logger.info("=" * 80)

    X_val, y_val = extract_features_with_augmentation(
        data_dir=data_dir,
        subset='val',
        max_videos=max_val_videos,
        use_augmentation=False
    )

    logger.info(f"Validation data shapes: X_val={X_val.shape}, y_val={y_val.shape}")

    # Phase 3: Build and train model with multi-GPU strategy
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: MULTI-GPU MODEL TRAINING")
    logger.info("=" * 80)

    # Create MirroredStrategy for multi-GPU training
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"✅ Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()
        logger.info("Using default strategy (single GPU)")

    # Build model within strategy scope
    with strategy.scope():
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape, num_classes=NUM_CLASSES)

        logger.info(f"Model input shape: {input_shape}")
        model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_path / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(output_path / 'training_log.csv')
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_path / 'tensorboard_logs'),
            histogram_freq=1
        )
    ]

    # Train with multi-GPU
    logger.info(f"Starting multi-GPU training:")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Batch size per GPU: {BATCH_SIZE}")
    logger.info(f"  Global batch size: {GLOBAL_BATCH_SIZE}")
    logger.info(f"  Training samples: {X_train.shape[0]}")
    logger.info(f"  Validation samples: {X_val.shape[0]}")
    logger.info(f"  Mixed precision: {policy.name}")
    logger.info(f"  Number of GPUs: {len(gpus)}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=GLOBAL_BATCH_SIZE,  # Automatically distributed across GPUs
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(str(output_path / 'final_model.h5'))
    logger.info(f"✅ Model saved to {output_path}")

    # Evaluation
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    logger.info(f"Training Accuracy: {train_acc:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    return history, model


def main():
    """Main entry point for multi-GPU training"""

    DATA_DIR = "/workspace/data/RWF-2000/RWF-2000"
    OUTPUT_DIR = "./models"

    logger.info("=" * 80)
    logger.info("VIOLENCE DETECTION TRAINING - MULTI-GPU (2x RTX 4080)")
    logger.info("=" * 80)
    logger.info(f"Dataset: {DATA_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Sequence Length: {SEQUENCE_LENGTH}")
    logger.info(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
    logger.info(f"Batch Size per GPU: {BATCH_SIZE}")
    logger.info(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Number of GPUs: {len(gpus)}")
    logger.info(f"Mixed Precision: {policy.name}")
    logger.info("=" * 80)

    if not Path(DATA_DIR).exists():
        logger.error(f"Dataset not found at: {DATA_DIR}")
        sys.exit(1)

    try:
        history, model = train_multi_gpu(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            max_train_videos=None,  # Use all videos
            max_val_videos=None
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ MULTI-GPU TRAINING COMPLETE!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
