#!/usr/bin/env python3
"""
TRAIN ROBUST MODEL 1 with 10x Aggressive Augmentation
Goal: Model that works in real-world scenarios
"""

import os
import sys
import logging
from pathlib import Path
import tensorflow as tf

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))

from train_ensemble_ultimate import (
    EnsembleConfig,
    load_dataset,
    extract_features_with_model,
    get_feature_extractor,
    create_model_1_vgg19_bilstm,
    train_single_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU Configuration: Use GPU 1 (best for single-model training)
# GPU 0 (979MB) is for display, GPU 1 (32GB) is for compute
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Enable mixed precision - RTX 5000 Ada has Tensor Cores
tf.keras.mixed_precision.set_global_policy('mixed_float16')

logger.info("✅ Using GPU 1 (RTX 5000 Ada, 32GB VRAM)")
logger.info("✅ Mixed Precision (FP16) enabled - 2x faster with Tensor Cores")
logger.info("✅ 12 CPU cores for parallel video loading")


def train_robust_model():
    """Train single robust model with aggressive augmentation"""

    logger.info("="*80)
    logger.info("TRAINING ROBUST MODEL 1 - 10x AUGMENTATION")
    logger.info("="*80)

    # Configure with aggressive augmentation
    config = EnsembleConfig(
        # Training
        epochs=150,
        batch_size=64,
        early_stopping_patience=25,  # More patience for augmented data

        # Aggressive augmentation - 10x multiplier (maximum robustness)
        use_augmentation=True,
        augmentation_multiplier=10,  # 10 versions per video (176K total samples)
        aug_flip_prob=0.5,
        aug_brightness_range=(0.6, 1.4),
        aug_contrast_range=(0.7, 1.5),
        aug_rotation_range=15,
        aug_zoom_range=(0.85, 1.15),
        aug_noise_prob=0.3,
        aug_blur_prob=0.2,
        aug_frame_dropout_prob=0.15,

        # Train only Model 1
        model_names=['vgg19_bilstm'],

        # Paths
        cache_dir="/workspace/robust_cache",
        checkpoint_dir="/workspace/robust_checkpoints",
        models_dir="/workspace/robust_models"
    )

    logger.info(f"\n📊 Augmentation Strategy:")
    logger.info(f"  - 10x multiplier: Each video → 10 augmented versions")
    logger.info(f"  - Brightness: {config.aug_brightness_range}")
    logger.info(f"  - Contrast: {config.aug_contrast_range}")
    logger.info(f"  - Rotation: ±{config.aug_rotation_range}°")
    logger.info(f"  - Zoom: {config.aug_zoom_range}")
    logger.info(f"  - Noise: {config.aug_noise_prob*100}%")
    logger.info(f"  - Blur: {config.aug_blur_prob*100}%")
    logger.info(f"  - Frame dropout: {config.aug_frame_dropout_prob*100}%")

    # Load dataset
    logger.info("\n" + "="*80)
    logger.info("LOADING DATASET")
    logger.info("="*80)
    dataset = load_dataset(config)

    original_train_size = len(dataset['train']['paths'])
    augmented_train_size = original_train_size * config.augmentation_multiplier

    logger.info(f"\n📈 Dataset Expansion:")
    logger.info(f"  Original training videos: {original_train_size:,}")
    logger.info(f"  After 10x augmentation: {augmented_train_size:,}")
    logger.info(f"  Validation videos: {len(dataset['val']['paths']):,} (no augmentation)")
    logger.info(f"  Test videos: {len(dataset['test']['paths']):,} (no augmentation)")

    # Extract features with augmentation
    model_name = 'vgg19_bilstm'
    logger.info(f"\n" + "="*80)
    logger.info(f"EXTRACTING FEATURES FOR {model_name.upper()}")
    logger.info("="*80)

    feature_extractor, preprocess_fn, input_shape = get_feature_extractor(model_name)

    # Training features (with 10x augmentation)
    logger.info(f"\n🔄 Creating 10x augmented training features...")
    logger.info("🚀 Using 12 CPU cores + 2 GPUs for maximum speed!")
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

    logger.info(f"\n✅ Augmented training features shape: {train_features.shape}")
    logger.info(f"✅ Original: {original_train_size} → Augmented: {len(train_features)} samples")

    # Validation features (no augmentation)
    val_features, val_labels = extract_features_with_model(
        dataset['val']['paths'],
        dataset['val']['labels'],
        feature_extractor,
        preprocess_fn,
        config,
        model_name,
        'val',
        is_training=False
    )

    # Test features (no augmentation)
    test_features, test_labels = extract_features_with_model(
        dataset['test']['paths'],
        dataset['test']['labels'],
        feature_extractor,
        preprocess_fn,
        config,
        model_name,
        'test',
        is_training=False
    )

    # Create and train model
    logger.info(f"\n" + "="*80)
    logger.info(f"TRAINING ROBUST MODEL")
    logger.info("="*80)

    model = create_model_1_vgg19_bilstm(input_shape, config)

    # Train
    train_single_model(
        model,
        model_name,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        config
    )

    logger.info("\n" + "="*80)
    logger.info("✅ ROBUST MODEL TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Model saved to: {config.models_dir}/{model_name}/best_model.h5")
    logger.info("\nNext step: Test with TTA to validate robustness!")


if __name__ == "__main__":
    train_robust_model()
