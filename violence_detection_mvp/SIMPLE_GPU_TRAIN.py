#!/usr/bin/env python3
"""
SIMPLE GPU-ACCELERATED TRAINING (Standalone Version)
Just upload this file + gpu_video_loader.py and run!

This is a self-contained version that doesn't need train_ensemble_ultimate.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Tuple, List
from datetime import datetime

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Import GPU video loader
from gpu_video_loader import GPUVideoLoader

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("🚀 SIMPLE GPU-ACCELERATED TRAINING")
print("=" * 80)
print("✅ GPU:1 selected")
print("✅ Mixed Precision (FP16) enabled")
print("=" * 80 + "\n")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths
DATASET_PATH = "/workspace/organized_dataset"
CACHE_DIR = "/workspace/simple_gpu_cache"
MODELS_DIR = "/workspace/simple_gpu_models"

# Training params
N_FRAMES = 20
FRAME_SIZE = (224, 224)
EPOCHS = 150
BATCH_SIZE = 64
AUGMENTATION_MULTIPLIER = 10  # 10x augmentation for robustness

# Create directories
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_frames(frames: np.ndarray) -> np.ndarray:
    """Apply aggressive augmentation"""
    # Flip
    if np.random.random() < 0.5:
        frames = np.flip(frames, axis=2)

    # Brightness
    factor = np.random.uniform(0.6, 1.4)
    frames = frames * factor
    frames = np.clip(frames, 0, 255)

    # Contrast
    factor = np.random.uniform(0.7, 1.5)
    frames = (frames - 128.0) * factor + 128.0
    frames = np.clip(frames, 0, 255)

    # Rotation
    if np.random.random() < 0.5:
        angle = np.random.uniform(-15, 15)
        h, w = frames.shape[1:3]
        center = (w // 2, h // 2)
        import cv2
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = []
        for frame in frames:
            rotated.append(cv2.warpAffine(frame, M, (w, h)))
        frames = np.array(rotated)

    # Noise
    if np.random.random() < 0.3:
        noise = np.random.normal(0, 10, frames.shape)
        frames = frames + noise
        frames = np.clip(frames, 0, 255)

    return frames


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset():
    """Load dataset paths and labels"""
    dataset_path = Path(DATASET_PATH)
    data = {}

    for split in ['train', 'val', 'test']:
        video_paths = []
        labels = []

        split_dir = dataset_path / split

        # Nonviolent (class 0)
        nonviolent_dir = split_dir / 'nonviolent'
        if nonviolent_dir.exists():
            for video in nonviolent_dir.glob('*.mp4'):
                video_paths.append(str(video))
                labels.append(0)

        # Violent (class 1)
        violent_dir = split_dir / 'violent'
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


# ============================================================================
# GPU FEATURE EXTRACTION
# ============================================================================

def extract_features_gpu(video_paths, labels, feature_extractor, split_name, is_training=True):
    """Extract features using GPU video loading"""

    logger.info(f"\n{'='*80}")
    logger.info(f"🎬 GPU Feature Extraction: {split_name}")
    logger.info(f"{'='*80}")

    gpu_loader = GPUVideoLoader(backend='auto')
    logger.info(f"✅ Backend: {gpu_loader.selected_backend}")

    aug_mult = AUGMENTATION_MULTIPLIER if is_training else 1
    total_samples = len(video_paths) * aug_mult

    all_features = []
    all_labels = []

    batch_frames = []
    batch_labels = []
    gpu_batch_size = 8  # Reduced from 32 to avoid OOM

    from tqdm import tqdm
    pbar = tqdm(total=total_samples, desc=f"GPU {split_name}", unit="video")

    for i, video_path in enumerate(video_paths):
        for aug_idx in range(aug_mult):
            # Load video on GPU
            frames = gpu_loader.load_video_gpu(video_path, N_FRAMES, FRAME_SIZE)

            if frames is None:
                continue

            # Augment if training
            if is_training:
                frames = augment_frames(frames)

            # Normalize
            frames = frames / 255.0

            batch_frames.append(frames)
            batch_labels.append(labels[i])

            # Process batch
            if len(batch_frames) >= gpu_batch_size:
                batch_array = np.array(batch_frames)
                frames_flat = batch_array.reshape(-1, FRAME_SIZE[0], FRAME_SIZE[1], 3)

                # Extract features on GPU
                features_flat = feature_extractor.predict(frames_flat, batch_size=32, verbose=0)
                features_batch = features_flat.reshape(len(batch_frames), N_FRAMES, 4096)

                all_features.append(features_batch)
                all_labels.extend(batch_labels)

                pbar.update(len(batch_frames))

                batch_frames = []
                batch_labels = []

    # Process remaining
    if batch_frames:
        batch_array = np.array(batch_frames)
        frames_flat = batch_array.reshape(-1, FRAME_SIZE[0], FRAME_SIZE[1], 3)
        features_flat = feature_extractor.predict(frames_flat, batch_size=32, verbose=0)
        features_batch = features_flat.reshape(len(batch_frames), N_FRAMES, 4096)

        all_features.append(features_batch)
        all_labels.extend(batch_labels)
        pbar.update(len(batch_frames))

    pbar.close()

    X = np.concatenate(all_features, axis=0)
    y = np.array(all_labels)

    logger.info(f"✅ Features: {X.shape}, Labels: {y.shape}")
    logger.info(f"   Normal: {np.sum(y==0)}, Violent: {np.sum(y==1)}")

    return X, y


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_bilstm_model(input_shape):
    """Build BiLSTM model"""
    inputs = tf.keras.Input(shape=input_shape)

    # BiLSTM layers
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.3)
    )(inputs)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, dropout=0.3)
    )(x)

    # Dense layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("SIMPLE GPU-ACCELERATED TRAINING")
    logger.info("=" * 80)
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")

    # Load dataset
    logger.info("📂 Loading dataset...")
    dataset = load_dataset()

    train_paths = dataset['train']['paths']
    train_labels = dataset['train']['labels']
    val_paths = dataset['val']['paths']
    val_labels = dataset['val']['labels']
    test_paths = dataset['test']['paths']
    test_labels = dataset['test']['labels']

    # Load VGG19
    logger.info("\n🔧 Loading VGG19 feature extractor...")
    base_model = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(*FRAME_SIZE, 3),
        pooling='avg'
    )
    base_model.trainable = False

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.output
    )
    logger.info("✅ VGG19 loaded")

    # Extract features
    logger.info("\n" + "=" * 80)
    logger.info("🚀 GPU FEATURE EXTRACTION")
    logger.info("=" * 80)

    import time
    start = time.time()

    X_train, y_train = extract_features_gpu(train_paths, train_labels, feature_extractor, "TRAIN", is_training=True)
    X_val, y_val = extract_features_gpu(val_paths, val_labels, feature_extractor, "VAL", is_training=False)
    X_test, y_test = extract_features_gpu(test_paths, test_labels, feature_extractor, "TEST", is_training=False)

    elapsed = time.time() - start
    total_videos = len(train_paths) * AUGMENTATION_MULTIPLIER + len(val_paths) + len(test_paths)
    logger.info(f"\n✅ Feature extraction complete: {elapsed/60:.1f} min ({total_videos/elapsed:.1f} videos/sec)")

    # Save features
    logger.info("\n💾 Caching features...")
    np.save(Path(CACHE_DIR) / "X_train.npy", X_train)
    np.save(Path(CACHE_DIR) / "y_train.npy", y_train)
    np.save(Path(CACHE_DIR) / "X_val.npy", X_val)
    np.save(Path(CACHE_DIR) / "y_val.npy", y_val)
    np.save(Path(CACHE_DIR) / "X_test.npy", X_test)
    np.save(Path(CACHE_DIR) / "y_test.npy", y_test)
    logger.info(f"✅ Cached to {CACHE_DIR}")

    # Build model
    logger.info("\n🏗️  Building BiLSTM model...")
    model = build_bilstm_model(input_shape=(N_FRAMES, 4096))
    logger.info("✅ Model built")

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("🎯 TRAINING")
    logger.info("=" * 80)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(MODELS_DIR) / "best_model_{epoch:03d}_{val_accuracy:.4f}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("📊 EVALUATION")
    logger.info("=" * 80)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"✅ Test Accuracy: {test_acc*100:.2f}%")

    # Save final model
    final_model_path = Path(MODELS_DIR) / "final_model.h5"
    model.save(final_model_path)
    logger.info(f"💾 Saved: {final_model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model: {final_model_path}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)
