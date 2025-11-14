#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED Multi-GPU Training for Maximum Accuracy
- Full epoch training (no early stopping unless specified)
- Maximum GPU utilization (2× RTX 4080)
- Aggressive data augmentation (5x)
- Optimized hyperparameters for best results
"""

import os
import sys

# GPU Configuration - MAXIMUM PERFORMANCE
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Better multi-GPU performance
os.environ['TF_GPU_THREAD_COUNT'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime

# Enable mixed precision for 2x speed
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# GPU Setup
print("=" * 80)
print("ULTRA-OPTIMIZED MULTI-GPU TRAINING")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        print(f"✅ Mixed Precision: ENABLED (float16)")
        print(f"✅ Multi-GPU Strategy: MirroredStrategy")
        print(f"✅ GPU Threading: Optimized")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️ No GPU detected")

print("=" * 80)

# Import configuration
from src.config import (
    SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH,
    NUM_CLASSES, RNN_SIZE, DROPOUT_RATE
)

# OPTIMIZED HYPERPARAMETERS
BATCH_SIZE = 128  # Increased from 64 for better GPU utilization
EPOCHS = 500  # Full training - let it run!
LEARNING_RATE = 0.0001
AUGMENTATION_FACTOR = 5  # 5x data augmentation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GLOBAL_BATCH_SIZE = BATCH_SIZE * len(gpus) if gpus else BATCH_SIZE
logger.info(f"Batch size per GPU: {BATCH_SIZE}")
logger.info(f"Global batch size: {GLOBAL_BATCH_SIZE}")


def extract_features_ultra_augmented(
    data_dir: str,
    subset: str = 'train',
    max_videos: int = None,
    augmentation_factor: int = AUGMENTATION_FACTOR
):
    """
    Extract features with ULTRA-AGGRESSIVE augmentation.
    Creates 5x more training data.
    """
    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import cv2

    logger.info(f"Extracting features for {subset} set...")
    logger.info(f"Augmentation factor: {augmentation_factor}x")

    # Initialize VGG19 on GPU:0
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

    logger.info("✅ VGG19 feature extractor loaded")

    # ULTRA-AGGRESSIVE Data Augmentation
    if subset == 'train':
        datagen = ImageDataGenerator(
            rotation_range=25,          # More rotation
            width_shift_range=0.25,     # More shifting
            height_shift_range=0.25,
            horizontal_flip=True,
            zoom_range=0.2,             # More zoom
            brightness_range=[0.6, 1.4],  # More brightness variation
            fill_mode='nearest',
            shear_range=0.15,           # More shear
            channel_shift_range=20.0    # Color variation
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

    num_versions = augmentation_factor if subset == 'train' else 1
    if subset == 'train':
        logger.info(f"Will create {len(all_videos) * num_versions} total samples with {num_versions}x augmentation")

    features_list = []
    labels_list = []

    for idx, (video_path, label) in enumerate(zip(all_videos, labels)):
        try:
            # Extract frames
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
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
                continue

            frames = np.array(frames, dtype=np.uint8)

            # Create multiple augmented versions
            for aug_idx in range(num_versions):
                if subset == 'train' and aug_idx > 0:
                    # Apply strong augmentation
                    augmented_frames = []
                    for frame in frames:
                        frame_reshaped = frame.reshape((1,) + frame.shape)
                        aug_iter = datagen.flow(frame_reshaped, batch_size=1)
                        augmented_frame = next(aug_iter)[0].astype('uint8')
                        augmented_frames.append(augmented_frame)
                    current_frames = np.array(augmented_frames)
                else:
                    current_frames = frames.copy()

                # Preprocess
                frames_preprocessed = preprocess_input(current_frames.astype('float32'))

                # Extract features
                with tf.device('/GPU:0'):
                    video_features = feature_extractor.predict(
                        frames_preprocessed,
                        batch_size=SEQUENCE_LENGTH,
                        verbose=0
                    )

                features_list.append(video_features)
                labels_list.append(label)

            if (idx + 1) % 100 == 0:
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


def build_optimized_lstm_model(input_shape, num_classes=2):
    """
    Build OPTIMIZED LSTM model with more capacity.
    """
    logger.info(f"Building OPTIMIZED LSTM model with input_shape={input_shape}")

    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.InputLayer(input_shape=input_shape),

        # LSTM layers - INCREASED CAPACITY
        tf.keras.layers.LSTM(
            256,  # Increased from 128
            return_sequences=True,
            dropout=0.3,  # Reduced dropout for more capacity
            recurrent_dropout=0.3,
            name='lstm_1'
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LSTM(
            128,  # Increased from 64
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.3,
            name='lstm_2'
        ),
        tf.keras.layers.BatchNormalization(),

        # Dense layers - MORE CAPACITY
        tf.keras.layers.Dense(512, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.4, name='dropout_1'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.4, name='dropout_2'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(128, activation='relu', name='dense_3'),
        tf.keras.layers.Dropout(0.3, name='dropout_3'),

        # Output
        tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')
    ])

    # Optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("✅ Optimized model built successfully")
    return model


def train_ultra_optimized(
    data_dir: str,
    output_dir: str = './models_ultra',
    augmentation_factor: int = AUGMENTATION_FACTOR,
    run_full_epochs: bool = True
):
    """
    ULTRA-OPTIMIZED training with maximum GPU utilization.

    Args:
        run_full_epochs: If True, run all 500 epochs (no early stopping)
    """
    logger.info("=" * 80)
    logger.info("ULTRA-OPTIMIZED TRAINING - MAXIMUM PERFORMANCE")
    logger.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # PHASE 1: Extract training features
    logger.info("\nPHASE 1: FEATURE EXTRACTION - TRAINING SET")
    X_train, y_train = extract_features_ultra_augmented(
        data_dir=data_dir,
        subset='train',
        augmentation_factor=augmentation_factor
    )

    logger.info(f"Training data: X_train={X_train.shape}, y_train={y_train.shape}")

    # PHASE 2: Extract validation features
    logger.info("\nPHASE 2: FEATURE EXTRACTION - VALIDATION SET")
    X_val, y_val = extract_features_ultra_augmented(
        data_dir=data_dir,
        subset='val',
        augmentation_factor=1  # No augmentation for validation
    )

    logger.info(f"Validation data: X_val={X_val.shape}, y_val={y_val.shape}")

    # PHASE 3: Build and train model
    logger.info("\nPHASE 3: ULTRA-OPTIMIZED MODEL TRAINING")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"✅ Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_optimized_lstm_model(input_shape, num_classes=NUM_CLASSES)
        model.summary()

    # Callbacks - OPTIMIZED FOR FULL TRAINING
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_path / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,  # More patience
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(output_path / 'training_log.csv')
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_path / 'tensorboard_logs'),
            histogram_freq=1,
            write_graph=True
        )
    ]

    # Add early stopping only if not running full epochs
    if not run_full_epochs:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,  # Very patient
                restore_best_weights=True,
                verbose=1
            )
        )
    else:
        logger.info("⚡ RUNNING FULL EPOCHS - NO EARLY STOPPING")

    # Train
    logger.info("\nStarting ULTRA-OPTIMIZED training:")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Batch size per GPU: {BATCH_SIZE}")
    logger.info(f"  Global batch size: {GLOBAL_BATCH_SIZE}")
    logger.info(f"  Training samples: {X_train.shape[0]}")
    logger.info(f"  Validation samples: {X_val.shape[0]}")
    logger.info(f"  Mixed precision: {policy.name}")
    logger.info(f"  Augmentation factor: {augmentation_factor}x")
    logger.info(f"  Early stopping: {'DISABLED' if run_full_epochs else 'ENABLED (patience=30)'}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=GLOBAL_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(str(output_path / 'final_model.h5'))
    logger.info(f"✅ Model saved to {output_path}")

    # Final evaluation
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    logger.info(f"Training Accuracy:   {train_acc:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Improvement over baseline: +{(val_acc - 0.7575) * 100:.2f}%")

    return history, model


def main():
    """Main entry point for ULTRA-OPTIMIZED training"""

    DATA_DIR = "/workspace/data/combined"  # Combined dataset
    OUTPUT_DIR = "./models_ultra"
    RUN_FULL_EPOCHS = True  # Set to False to enable early stopping

    logger.info("=" * 80)
    logger.info("ULTRA-OPTIMIZED VIOLENCE DETECTION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {DATA_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch Size per GPU: {BATCH_SIZE}")
    logger.info(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")
    logger.info(f"Augmentation: {AUGMENTATION_FACTOR}x")
    logger.info(f"GPUs: {len(gpus)}")
    logger.info(f"Mixed Precision: {policy.name}")
    logger.info(f"Full Epochs: {RUN_FULL_EPOCHS}")
    logger.info("=" * 80)

    if not Path(DATA_DIR).exists():
        logger.error(f"Dataset not found at: {DATA_DIR}")
        sys.exit(1)

    try:
        history, model = train_ultra_optimized(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            augmentation_factor=AUGMENTATION_FACTOR,
            run_full_epochs=RUN_FULL_EPOCHS
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ ULTRA-OPTIMIZED TRAINING COMPLETE!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
