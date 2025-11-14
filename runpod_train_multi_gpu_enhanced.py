#!/usr/bin/env python3
"""
Enhanced Multi-GPU Training with Aggressive Data Augmentation
Improves accuracy through synthetic data expansion
"""

import os
import sys

# GPU Configuration
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

# Mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} Physical GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Import configuration
from src.config import (
    SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    NUM_CLASSES, RNN_SIZE, DROPOUT_RATE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GLOBAL_BATCH_SIZE = BATCH_SIZE * len(gpus) if gpus else BATCH_SIZE


def extract_features_with_aggressive_augmentation(
    data_dir: str,
    subset: str = 'train',
    max_videos: int = None,
    augmentation_factor: int = 3  # NEW: Create 3x more data
):
    """
    Extract features with AGGRESSIVE augmentation.
    Creates multiple augmented versions of each video.

    Args:
        augmentation_factor: How many augmented versions per video (2-5 recommended)
    """
    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import cv2

    logger.info(f"Extracting features for {subset} set with {augmentation_factor}x augmentation...")

    # Initialize VGG19
    with tf.device('/GPU:0'):
        vgg19 = VGG19(weights='imagenet', include_top=True, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        feature_extractor = tf.keras.Model(
            inputs=vgg19.input,
            outputs=vgg19.get_layer('fc2').output
        )

    # AGGRESSIVE Data Augmentation
    if subset == 'train':
        datagen = ImageDataGenerator(
            rotation_range=20,          # Increased from 10
            width_shift_range=0.2,      # Increased from 0.1
            height_shift_range=0.2,     # Increased from 0.1
            horizontal_flip=True,
            vertical_flip=False,        # NEW: No vertical flip (unnatural)
            zoom_range=0.15,            # Increased from 0.1
            brightness_range=[0.7, 1.3], # NEW: Brightness variation
            fill_mode='nearest',
            shear_range=0.1             # NEW: Shear transformation
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
    if subset == 'train':
        logger.info(f"Will create {len(all_videos) * augmentation_factor} total samples with augmentation")

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

            frames = np.array(frames, dtype=np.uint8)

            # Create multiple augmented versions
            num_versions = augmentation_factor if subset == 'train' else 1

            for aug_idx in range(num_versions):
                if subset == 'train' and aug_idx > 0:
                    # Apply augmentation
                    augmented_frames = []
                    for frame in frames:
                        frame_reshaped = frame.reshape((1,) + frame.shape)
                        aug_iter = datagen.flow(frame_reshaped, batch_size=1)
                        augmented_frame = next(aug_iter)[0].astype('uint8')
                        augmented_frames.append(augmented_frame)
                    current_frames = np.array(augmented_frames)
                else:
                    # Original frames (no augmentation)
                    current_frames = frames.copy()

                # Preprocess for VGG19
                frames_preprocessed = preprocess_input(current_frames.astype('float32'))

                # Extract features
                with tf.device('/GPU:0'):
                    video_features = feature_extractor.predict(
                        frames_preprocessed,
                        batch_size=SEQUENCE_LENGTH,
                        verbose=0
                    )

                assert video_features.shape == (SEQUENCE_LENGTH, 4096)

                features_list.append(video_features)
                labels_list.append(label)

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(all_videos)} videos")

        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            continue

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)

    logger.info(f"✅ Feature extraction complete:")
    logger.info(f"   X.shape = {X.shape}")
    logger.info(f"   y.shape = {y.shape}")

    return X, y


# Keep the same build_lstm_model and train_multi_gpu functions from original
# (Copy from runpod_train_multi_gpu.py)

def build_lstm_model(input_shape, num_classes=2):
    """Build LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.LSTM(RNN_SIZE, return_sequences=True, dropout=DROPOUT_RATE, name='lstm_1'),
        tf.keras.layers.LSTM(RNN_SIZE // 2, return_sequences=False, dropout=DROPOUT_RATE, name='lstm_2'),
        tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(DROPOUT_RATE, name='dropout_1'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(DROPOUT_RATE, name='dropout_2'),
        tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_multi_gpu(data_dir: str, output_dir: str = './models_enhanced', augmentation_factor: int = 3):
    """Main training with aggressive augmentation."""

    logger.info("=" * 80)
    logger.info(f"ENHANCED TRAINING (with {augmentation_factor}x augmentation)")
    logger.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract features with augmentation
    logger.info("\nPHASE 1: FEATURE EXTRACTION - TRAINING SET (with augmentation)")
    X_train, y_train = extract_features_with_aggressive_augmentation(
        data_dir=data_dir,
        subset='train',
        augmentation_factor=augmentation_factor
    )

    logger.info("\nPHASE 2: FEATURE EXTRACTION - VALIDATION SET (no augmentation)")
    X_val, y_val = extract_features_with_aggressive_augmentation(
        data_dir=data_dir,
        subset='val',
        augmentation_factor=1  # No augmentation for validation
    )

    # Build and train model
    logger.info("\nPHASE 3: MULTI-GPU MODEL TRAINING")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"✅ Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape, num_classes=NUM_CLASSES)

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
            patience=20,  # Increased patience for larger dataset
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,  # Increased patience
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(filename=str(output_path / 'training_log.csv')),
        tf.keras.callbacks.TensorBoard(log_dir=str(output_path / 'tensorboard_logs'))
    ]

    logger.info(f"Training samples: {X_train.shape[0]} ({X_train.shape[0] // augmentation_factor} original videos)")
    logger.info(f"Validation samples: {X_val.shape[0]}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=GLOBAL_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    model.save(str(output_path / 'final_model.h5'))
    logger.info(f"✅ Model saved to {output_path}")

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    logger.info("\nFINAL RESULTS:")
    logger.info(f"Training Accuracy: {train_acc:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    return history, model


def main():
    # Update with combined dataset path
    DATA_DIR = "/workspace/data/combined"  # Or use RWF-2000 with more augmentation
    OUTPUT_DIR = "./models_enhanced"
    AUGMENTATION_FACTOR = 3  # Create 3x more training data

    logger.info("=" * 80)
    logger.info("ENHANCED VIOLENCE DETECTION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {DATA_DIR}")
    logger.info(f"Augmentation Factor: {AUGMENTATION_FACTOR}x")
    logger.info(f"Expected improvement: 3-5% accuracy boost")
    logger.info("=" * 80)

    if not Path(DATA_DIR).exists():
        logger.error(f"Dataset not found at: {DATA_DIR}")
        sys.exit(1)

    try:
        history, model = train_multi_gpu(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            augmentation_factor=AUGMENTATION_FACTOR
        )
        logger.info("\n✅ ENHANCED TRAINING COMPLETE!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
