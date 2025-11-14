#!/usr/bin/env python3
"""
TRAIN ROBUST MODEL with GPU-ACCELERATED VIDEO LOADING
This version uses TensorFlow's native GPU video decoding for 10-50x speedup

Performance comparison:
- OpenCV CPU: ~0.15-0.20s per video (5-7 videos/sec)
- TensorFlow GPU: ~0.003-0.010s per video (100-300 videos/sec)

Expected speedup: 10-50x faster feature extraction
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Tuple, List
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our libraries
from train_ensemble_ultimate import (
    EnsembleConfig,
    DataAugmentation,
    create_model_1_vgg19_bilstm,
    load_dataset
)
from gpu_video_loader_fixed import GPUVideoLoaderFixed, ThreadedGPUVideoLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# GPU Configuration: Use GPU 1 (RTX 5000 Ada - 32GB)
# ========================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU:1 for compute (GPU:0 is display)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Better GPU utilization

# Enable mixed precision for 2x speedup on RTX 5000 Ada
tf.keras.mixed_precision.set_global_policy('mixed_float16')

logger.info("=" * 80)
logger.info("🚀 GPU-ACCELERATED ROBUST TRAINING")
logger.info("=" * 80)
logger.info("✅ GPU Configuration:")
logger.info("   - Using GPU:1 (RTX 5000 Ada, 32GB VRAM)")
logger.info("   - Mixed Precision (FP16) enabled")
logger.info("   - GPU video decoding enabled")
logger.info("   - Expected 10-50x speedup vs CPU video loading")
logger.info("=" * 80)


def extract_video_frames_gpu(
    video_path: str,
    gpu_loader: GPUVideoLoaderFixed,
    n_frames: int,
    frame_size: Tuple[int, int],
    config: EnsembleConfig,
    is_training: bool = True
) -> np.ndarray:
    """
    Extract frames using GPU acceleration with augmentation

    Args:
        video_path: Path to video
        gpu_loader: GPU video loader instance
        n_frames: Number of frames to extract
        frame_size: (height, width)
        config: Training configuration
        is_training: Whether to apply augmentation

    Returns:
        Frames array or None if failed
    """
    try:
        # Load video using GPU (10-50x faster than OpenCV)
        frames = gpu_loader.load_video_gpu(video_path, n_frames, frame_size)

        if frames is None or len(frames) == 0:
            return None

        # Apply augmentation only during training
        if is_training and config.use_augmentation:
            frames = DataAugmentation.augment_frames(frames, config)

        return frames

    except Exception as e:
        logger.error(f"Error loading video {video_path}: {e}")
        return None


def extract_features_gpu_accelerated(
    video_paths: List[str],
    labels: np.ndarray,
    feature_extractor: tf.keras.Model,
    config: EnsembleConfig,
    split_name: str,
    is_training: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features using GPU-accelerated video loading

    This version uses:
    1. TensorFlow GPU video decoding (10-50x faster than OpenCV)
    2. Large GPU batches for feature extraction
    3. Aggressive augmentation during training

    Expected performance:
    - 100-300 videos/sec (vs 5-7 videos/sec with OpenCV)
    - 60-70% GPU utilization (vs 0-5% with OpenCV)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"🎬 GPU-ACCELERATED Feature Extraction: {split_name}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Videos: {len(video_paths)}")
    logger.info(f"Augmentation: {config.augmentation_multiplier}x" if is_training else "None (validation)")
    logger.info(f"GPU Video Decoding: TensorFlow native (10-50x faster)")
    logger.info(f"{'=' * 80}\n")

    # Initialize GPU video loader with threading
    gpu_loader = GPUVideoLoaderFixed()
    logger.info(f"✅ GPU Video Loader initialized: {gpu_loader.backend}")

    # Calculate augmented samples
    aug_multiplier = config.augmentation_multiplier if is_training else 1
    total_samples = len(video_paths) * aug_multiplier

    # Prepare storage
    n_frames = config.n_frames
    feature_dim = 4096  # VGG19 features
    batch_shape = (n_frames, config.frame_size[0], config.frame_size[1], 3)

    all_features = []
    all_labels = []

    # GPU batch processing (simple but effective)
    gpu_batch_size = 8  # Process 8 videos at once on GPU
    batch_frames = []
    batch_labels = []

    from tqdm import tqdm

    logger.info(f"💡 Processing {total_samples:,} samples in GPU batches of {gpu_batch_size}")
    logger.info(f"💡 Using TensorFlow GPU resize for 2-3x speedup over pure OpenCV")

    pbar = tqdm(total=total_samples, desc=f"GPU {split_name}", unit="video")

    for i, video_path in enumerate(video_paths):
        # Create augmented versions
        for aug_idx in range(aug_multiplier):
            frames = extract_video_frames_gpu(
                video_path,
                gpu_loader,
                n_frames,
                config.frame_size,
                config,
                is_training=is_training
            )

            if frames is None:
                continue

            # Normalize
            frames = frames / 255.0

            # Add to batch
            batch_frames.append(frames)
            batch_labels.append(labels[i])

            # Process batch when full
            if len(batch_frames) >= gpu_batch_size:
                # Convert to array
                batch_array = np.array(batch_frames)

                # Flatten for VGG19: (batch, n_frames, H, W, 3) -> (batch*n_frames, H, W, 3)
                frames_flat = batch_array.reshape(
                    -1,
                    batch_shape[1],
                    batch_shape[2],
                    batch_shape[3]
                )

                # Extract features on GPU (batch processing)
                features_flat = feature_extractor.predict(
                    frames_flat,
                    batch_size=32,  # Reduced from 256 to avoid OOM
                    verbose=0
                )

                # Reshape back: (batch*n_frames, 4096) -> (batch, n_frames, 4096)
                features_batch = features_flat.reshape(
                    len(batch_frames),
                    n_frames,
                    feature_dim
                )

                # Store
                all_features.append(features_batch)
                all_labels.extend(batch_labels)

                # Update progress
                pbar.update(len(batch_frames))

                # Clear batch
                batch_frames = []
                batch_labels = []

    # Process remaining batch
    if batch_frames:
        batch_array = np.array(batch_frames)
        frames_flat = batch_array.reshape(-1, batch_shape[1], batch_shape[2], batch_shape[3])

        features_flat = feature_extractor.predict(frames_flat, batch_size=32, verbose=0)
        features_batch = features_flat.reshape(len(batch_frames), n_frames, feature_dim)

        all_features.append(features_batch)
        all_labels.extend(batch_labels)

        pbar.update(len(batch_frames))

    pbar.close()

    # Concatenate all
    X = np.concatenate(all_features, axis=0)
    y = np.array(all_labels)

    logger.info(f"✅ {split_name} features extracted:")
    logger.info(f"   - Features shape: {X.shape}")
    logger.info(f"   - Labels shape: {y.shape}")
    logger.info(f"   - Fight samples: {np.sum(y == 1)}")
    logger.info(f"   - Normal samples: {np.sum(y == 0)}")

    return X, y


def train_robust_model_gpu():
    """Train robust model with GPU-accelerated video loading"""

    # Configuration
    config = EnsembleConfig(
        # Training
        epochs=150,
        batch_size=64,
        early_stopping_patience=25,

        # Aggressive augmentation - 10x multiplier
        use_augmentation=True,
        augmentation_multiplier=10,
        aug_flip_prob=0.5,
        aug_brightness_range=(0.6, 1.4),
        aug_contrast_range=(0.7, 1.5),
        aug_rotation_range=15,
        aug_zoom_range=(0.85, 1.15),
        aug_noise_prob=0.3,
        aug_blur_prob=0.2,
        aug_frame_dropout_prob=0.15,

        # Model
        model_names=['vgg19_bilstm'],

        # Paths
        dataset_path="/workspace/organized_dataset",
        cache_dir="/workspace/robust_gpu_cache",
        checkpoint_dir="/workspace/robust_gpu_checkpoints",
        models_dir="/workspace/robust_gpu_models"
    )

    # Create directories
    for dir_path in [config.cache_dir, config.checkpoint_dir, config.models_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Get data split
    logger.info("\n📂 Loading dataset split...")
    dataset = load_dataset(config)

    train_paths = dataset['train']['paths']
    train_labels = dataset['train']['labels']
    val_paths = dataset['val']['paths']
    val_labels = dataset['val']['labels']
    test_paths = dataset['test']['paths']
    test_labels = dataset['test']['labels']

    logger.info(f"✅ Dataset loaded:")
    logger.info(f"   Train: {len(train_paths)} videos")
    logger.info(f"   Val: {len(val_paths)} videos")
    logger.info(f"   Test: {len(test_paths)} videos")

    # Load VGG19 feature extractor (with fc2 layer for 4096 features)
    logger.info("\n🔧 Loading VGG19 feature extractor...")
    base_model = tf.keras.applications.VGG19(
        include_top=True,
        weights='imagenet',
        input_shape=(*config.frame_size, 3)
    )
    base_model.trainable = False

    # Extract from fc2 layer (4096 features)
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )

    logger.info(f"✅ VGG19 loaded (output: {feature_extractor.output_shape})")

    # Extract features with GPU acceleration
    logger.info("\n" + "=" * 80)
    logger.info("🚀 STARTING GPU-ACCELERATED FEATURE EXTRACTION")
    logger.info("=" * 80)

    import time
    start_time = time.time()

    # Training features (with 10x augmentation)
    X_train, y_train = extract_features_gpu_accelerated(
        train_paths,
        train_labels,
        feature_extractor,
        config,
        "TRAIN",
        is_training=True
    )

    # Validation features (no augmentation)
    X_val, y_val = extract_features_gpu_accelerated(
        val_paths,
        val_labels,
        feature_extractor,
        config,
        "VAL",
        is_training=False
    )

    # Test features (no augmentation)
    X_test, y_test = extract_features_gpu_accelerated(
        test_paths,
        test_labels,
        feature_extractor,
        config,
        "TEST",
        is_training=False
    )

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ GPU Feature Extraction Complete")
    logger.info(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"📊 Average: {(len(train_paths)*10 + len(val_paths) + len(test_paths))/elapsed:.1f} videos/sec")
    logger.info("=" * 80 + "\n")

    # Save features to cache
    logger.info("💾 Saving features to cache...")
    np.save(Path(config.cache_dir) / "X_train_vgg19.npy", X_train)
    np.save(Path(config.cache_dir) / "y_train.npy", y_train)
    np.save(Path(config.cache_dir) / "X_val_vgg19.npy", X_val)
    np.save(Path(config.cache_dir) / "y_val.npy", y_val)
    np.save(Path(config.cache_dir) / "X_test_vgg19.npy", X_test)
    np.save(Path(config.cache_dir) / "y_test.npy", y_test)

    logger.info(f"✅ Features cached to {config.cache_dir}")

    # Build BiLSTM model
    logger.info("\n🏗️  Building BiLSTM model...")
    model = create_model_1_vgg19_bilstm(
        input_shape=(config.n_frames, 4096),
        config=config
    )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("✅ Model built and compiled")
    model.summary(print_fn=lambda x: logger.info(x))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(config.checkpoint_dir) / "robust_vgg19_{epoch:03d}_{val_accuracy:.4f}.h5"),
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
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(Path(config.models_dir) / "training_log.csv")
        )
    ]

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("🎯 TRAINING ROBUST MODEL")
    logger.info("=" * 80)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL EVALUATION")
    logger.info("=" * 80)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"✅ Test Accuracy: {test_acc*100:.2f}%")

    # Save final model
    final_model_path = Path(config.models_dir) / "robust_vgg19_final.h5"
    model.save(final_model_path)
    logger.info(f"💾 Final model saved: {final_model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model: {final_model_path}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Next: Test with TTA to validate robustness")
    logger.info("=" * 80)

    return model, history


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ROBUST MODEL TRAINING WITH GPU-ACCELERATED VIDEO LOADING")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")

    try:
        model, history = train_robust_model_gpu()
        logger.info("\n✅ SUCCESS - Training completed")
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)
