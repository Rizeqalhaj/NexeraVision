"""
Optimized training script with GPU memory management and feature caching.
Automatically adapts to available GPU memory.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.violence_detection_model import ViolenceDetectionModel
from src.config import (
    SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH,
    BATCH_SIZE, LEARNING_RATE, EPOCHS
)


def check_gpu_memory() -> Tuple[bool, float]:
    """Check available GPU memory in GB."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return False, 0.0

    # Get GPU memory info
    try:
        # This will show total memory
        gpu_info = tf.config.experimental.get_memory_info('GPU:0')
        total_gb = gpu_info['peak'] / (1024**3)
        return True, total_gb
    except:
        # Estimate based on GPU model
        return True, 8.0  # Default assumption


def get_optimal_batch_size(gpu_memory_gb: float) -> int:
    """Determine optimal batch size based on GPU memory."""
    if gpu_memory_gb <= 4:
        return 8
    elif gpu_memory_gb <= 6:
        return 16
    elif gpu_memory_gb <= 8:
        return 24
    elif gpu_memory_gb <= 12:
        return 32
    elif gpu_memory_gb <= 16:
        return 48
    else:
        return 64


def extract_vgg19_features(
    video_paths: list,
    labels: np.ndarray,
    output_dir: str,
    batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract VGG19 features from videos and cache to disk.
    This is a one-time operation that saves memory during training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "vgg19_features.npy"
    labels_path = output_dir / "labels.npy"

    # Check if features already extracted
    if features_path.exists() and labels_path.exists():
        print("📦 Loading cached VGG19 features...")
        features = np.load(features_path)
        cached_labels = np.load(labels_path)
        return features, cached_labels

    print("🔍 Extracting VGG19 features (one-time operation)...")

    # Load VGG19 without top layers
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
    import cv2

    vgg19 = VGG19(
        weights='imagenet',
        include_top=True,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    # Extract fc2 layer (4096 dimensions)
    feature_extractor = tf.keras.Model(
        inputs=vgg19.input,
        outputs=vgg19.get_layer('fc2').output
    )

    all_features = []

    for i, video_path in enumerate(video_paths):
        if i % 10 == 0:
            print(f"Processing video {i+1}/{len(video_paths)}...")

        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and preprocess
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # Pad if necessary
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))

        frames = frames[:SEQUENCE_LENGTH]
        frames_array = np.array(frames)
        frames_array = preprocess_input(frames_array)

        # Extract features
        features = feature_extractor.predict(frames_array, verbose=0)
        all_features.append(features)

    all_features = np.array(all_features)

    # Save to disk
    print(f"💾 Saving features to {features_path}")
    np.save(features_path, all_features)
    np.save(labels_path, labels)

    return all_features, labels


def create_feature_only_model(
    sequence_length: int = SEQUENCE_LENGTH,
    feature_dim: int = 4096
) -> tf.keras.Model:
    """
    Create model that takes pre-extracted features as input.
    This model has only LSTM + Attention layers.
    """
    from tensorflow.keras import layers, Model

    # Input: pre-extracted features
    inputs = layers.Input(shape=(sequence_length, feature_dim))

    # LSTM layers
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(128, return_sequences=True)(x)

    # Multi-head self-attention
    attention = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=128,
        dropout=0.3
    )(x, x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(attention)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_with_feature_caching(
    train_dir: str,
    val_dir: str,
    cache_dir: str = "data/processed/features",
    epochs: int = EPOCHS
):
    """Train using cached features - memory efficient approach."""

    print("=" * 60)
    print("FEATURE CACHING MODE (Memory Efficient)")
    print("=" * 60)

    # Get video paths and labels
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    train_fight = list((train_dir / "Fight").glob("*.avi"))
    train_nonviolence = list((train_dir / "NonFight").glob("*.avi"))
    val_fight = list((val_dir / "Fight").glob("*.avi"))
    val_nonviolence = list((val_dir / "NonFight").glob("*.avi"))

    train_videos = train_fight + train_nonviolence
    val_videos = val_fight + val_nonviolence

    train_labels = np.array([1] * len(train_fight) + [0] * len(train_nonviolence))
    val_labels = np.array([1] * len(val_fight) + [0] * len(val_nonviolence))

    # Extract features (cached)
    X_train, y_train = extract_vgg19_features(
        train_videos, train_labels, f"{cache_dir}/train"
    )
    X_val, y_val = extract_vgg19_features(
        val_videos, val_labels, f"{cache_dir}/val"
    )

    print(f"\n✅ Features extracted:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")

    # Create lightweight model
    model = create_feature_only_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("\n📊 Model Summary:")
    model.summary()

    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/violence_detector_cached.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]

    # Determine batch size based on GPU memory
    has_gpu, gpu_memory = check_gpu_memory()
    batch_size = get_optimal_batch_size(gpu_memory) if has_gpu else 8

    print(f"\n🎮 Training Configuration:")
    print(f"   GPU: {'Yes' if has_gpu else 'No'}")
    print(f"   GPU Memory: {gpu_memory:.1f} GB" if has_gpu else "   GPU Memory: N/A")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    results = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✅ Final Validation Results:")
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.4f}")
    print(f"   Precision: {results[2]:.4f}")
    print(f"   Recall: {results[3]:.4f}")

    return model, history


def train_standard(
    train_dir: str,
    val_dir: str,
    epochs: int = EPOCHS
):
    """Standard training - requires more GPU memory."""

    print("=" * 60)
    print("STANDARD TRAINING MODE")
    print("=" * 60)

    # Check GPU
    has_gpu, gpu_memory = check_gpu_memory()
    batch_size = get_optimal_batch_size(gpu_memory) if has_gpu else 8

    print(f"\n🎮 Configuration:")
    print(f"   GPU: {'Yes' if has_gpu else 'No (CPU mode - will be slow!)'}")
    if has_gpu:
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
    print(f"   Batch Size: {batch_size}")

    if not has_gpu:
        print("\n⚠️  WARNING: No GPU detected!")
        print("   Training will be VERY slow on CPU.")
        print("   Consider using feature caching mode or cloud GPU.")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return None, None

    # Create model
    model = ViolenceDetectionModel.create_model()

    # TODO: Implement data loading and training
    # This requires data pipeline implementation

    print("\n⚠️  Standard training pipeline not yet implemented.")
    print("   Use feature caching mode instead:")
    print("   python train_optimized.py --mode cached")

    return None, None


def main():
    parser = argparse.ArgumentParser(description='Optimized Violence Detection Training')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['cached', 'standard'],
        default='cached',
        help='Training mode: cached (memory efficient) or standard (requires more GPU memory)'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/raw/sample_training/train',
        help='Training data directory'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default='data/raw/sample_training/val',
        help='Validation data directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/processed/features',
        help='Directory to cache extracted features'
    )

    args = parser.parse_args()

    # Enable mixed precision if GPU available (2x memory efficiency)
    if tf.config.list_physical_devices('GPU'):
        try:
            from tensorflow.keras.mixed_precision import set_global_policy
            set_global_policy('mixed_float16')
            print("✅ Mixed precision enabled (FP16)")
        except:
            print("⚠️  Mixed precision not available")

    # Train based on mode
    if args.mode == 'cached':
        model, history = train_with_feature_caching(
            args.train_dir,
            args.val_dir,
            args.cache_dir,
            args.epochs
        )
    else:
        model, history = train_standard(
            args.train_dir,
            args.val_dir,
            args.epochs
        )

    if model:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed or cancelled.")


if __name__ == "__main__":
    main()
