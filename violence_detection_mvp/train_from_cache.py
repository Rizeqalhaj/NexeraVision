#!/usr/bin/env python3
"""
Train model from CACHED features (no re-extraction needed!)
Use this to continue training after feature extraction completed
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Import
sys.path.append(str(Path(__file__).parent))
from train_ensemble_ultimate import EnsembleConfig, create_model_1_vgg19_bilstm

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("🚀 TRAINING FROM CACHED FEATURES")
print("=" * 80)
print("✅ No feature extraction needed - loading from cache")
print("=" * 80 + "\n")


def train_from_cache():
    """Train model using cached features"""

    # Paths
    cache_dir = Path("/workspace/robust_gpu_cache")
    models_dir = Path("/workspace/robust_gpu_models")
    checkpoint_dir = Path("/workspace/robust_gpu_checkpoints")

    # Check cache exists
    if not cache_dir.exists():
        logger.error(f"❌ Cache directory not found: {cache_dir}")
        logger.error("Run train_robust_gpu_accelerated.py first to extract features")
        sys.exit(1)

    # Load cached features
    logger.info("📂 Loading cached features...")
    logger.info(f"   Cache: {cache_dir}")

    try:
        X_train = np.load(cache_dir / "X_train_vgg19.npy")
        y_train = np.load(cache_dir / "y_train.npy")
        X_val = np.load(cache_dir / "X_val_vgg19.npy")
        y_val = np.load(cache_dir / "y_val.npy")
        X_test = np.load(cache_dir / "X_test_vgg19.npy")
        y_test = np.load(cache_dir / "y_test.npy")

        logger.info("✅ Features loaded successfully!")
        logger.info(f"   Train: {X_train.shape}, Labels: {y_train.shape}")
        logger.info(f"   Val:   {X_val.shape}, Labels: {y_val.shape}")
        logger.info(f"   Test:  {X_test.shape}, Labels: {y_test.shape}")

    except FileNotFoundError as e:
        logger.error(f"❌ Cache file not found: {e}")
        logger.error("Make sure feature extraction completed successfully")
        sys.exit(1)

    # Build model
    logger.info("\n🏗️  Building BiLSTM model...")

    config = EnsembleConfig(
        n_frames=20,
        epochs=150,
        batch_size=64,
        early_stopping_patience=25,
        model_names=['vgg19_bilstm'],
        dataset_path="/workspace/organized_dataset",
        cache_dir=str(cache_dir),
        checkpoint_dir=str(checkpoint_dir),
        models_dir=str(models_dir)
    )

    model = create_model_1_vgg19_bilstm(
        input_shape=(config.n_frames, 4096),
        config=config
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("✅ Model built and compiled")
    model.summary()

    # Callbacks
    logger.info("\n🔧 Setting up callbacks...")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "robust_vgg19_{epoch:03d}_{val_accuracy:.4f}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=25,
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
            str(models_dir / "training_log.csv")
        )
    ]

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("🎯 TRAINING ROBUST MODEL")
    logger.info("=" * 80)
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Early stopping patience: {config.early_stopping_patience}")
    logger.info("=" * 80 + "\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL EVALUATION")
    logger.info("=" * 80)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"✅ Test Loss: {test_loss:.4f}")
    logger.info(f"✅ Test Accuracy: {test_acc*100:.2f}%")

    # Save final model
    final_model_path = models_dir / "robust_vgg19_final.h5"
    model.save(final_model_path)
    logger.info(f"💾 Final model saved: {final_model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {final_model_path}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info("\nNext: Test with TTA to validate robustness")
    logger.info("  python3 predict_with_tta_simple.py --model {final_model_path}")
    logger.info("=" * 80)

    return model, history


if __name__ == "__main__":
    try:
        model, history = train_from_cache()
        logger.info("\n✅ SUCCESS")
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)
