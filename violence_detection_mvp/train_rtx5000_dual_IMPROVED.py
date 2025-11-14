#!/usr/bin/env python3
"""
RTX 5000 Ada Training with Memory-Efficient Data Loading
Uses cached features with TensorFlow data pipeline (no OOM errors)
Strong regularization to prevent overfitting

GPU Setup: Uses GPU:1 only (GPU:0 is for desktop)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime

# Use ONLY GPU:1 (GPU:0 is for desktop)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

sys.path.append(str(Path(__file__).parent))
from train_ensemble_ultimate import EnsembleConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("🚀 RTX 5000 Ada TRAINING (Memory-Efficient + Anti-Overfitting)")
print("=" * 80)
print("✅ Using GPU:1 (32GB) with data pipeline (no OOM)")
print("✅ Strong regularization to prevent overfitting")
print("=" * 80 + "\n")


def create_dataset_from_cached_features(cache_dir, split, batch_size, shuffle=True):
    """
    Create TensorFlow dataset from cached numpy arrays
    This loads data in batches to avoid OOM errors
    """
    X_path = cache_dir / f"X_{split}_vgg19.npy"
    y_path = cache_dir / f"y_{split}.npy"

    # Load metadata only (not full data)
    X = np.load(X_path, mmap_mode='r')  # Memory-mapped (doesn't load into RAM)
    y = np.load(y_path)

    num_samples = len(y)
    logger.info(f"   {split}: {num_samples} samples, shape: {X.shape}")

    # Create dataset that loads in batches
    def data_generator():
        """Generator that yields batches"""
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            yield X[idx], y[idx]

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(20, 4096), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    )

    # Optimize dataset pipeline
    dataset = dataset.repeat()  # Repeat indefinitely to avoid end of sequence
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_samples


def build_robust_bilstm(input_shape):
    """
    Build BiLSTM with STRONG regularization to prevent overfitting

    Regularization techniques:
    - Dropout: 0.5-0.6 (high)
    - L2 regularization on all layers
    - Batch normalization
    - Smaller hidden units to reduce capacity
    """
    from tensorflow.keras import regularizers

    inputs = tf.keras.Input(shape=input_shape)

    # BiLSTM with L2 regularization and high dropout
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,  # Reduced from 192 to prevent overfitting
            return_sequences=True,
            dropout=0.5,
            recurrent_dropout=0.3,
            kernel_regularizer=regularizers.l2(0.01)
        )
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,  # Reduced from 96
            dropout=0.5,
            recurrent_dropout=0.3,
            kernel_regularizer=regularizers.l2(0.01)
        )
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Dense layers with strong dropout
    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # High dropout
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='robust_vgg19_bilstm')
    return model


def train_from_cache():
    """Train using GPU:1 with memory-efficient loading"""

    cache_dir = Path("/workspace/robust_gpu_cache")
    models_dir = Path("/workspace/robust_gpu_models")
    checkpoint_dir = Path("/workspace/robust_gpu_checkpoints")

    # Create directories
    models_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Check cache
    if not cache_dir.exists():
        logger.error(f"❌ Cache not found: {cache_dir}")
        sys.exit(1)

    # Config
    batch_size = 32  # Smaller batch for stability
    epochs = 150
    learning_rate = 0.0001

    logger.info("📂 Loading cached features with memory-efficient pipeline...")

    # Create datasets (memory-efficient - loads in batches)
    train_dataset, train_samples = create_dataset_from_cached_features(
        cache_dir, 'train', batch_size, shuffle=True
    )
    val_dataset, val_samples = create_dataset_from_cached_features(
        cache_dir, 'val', batch_size, shuffle=False
    )
    test_dataset, test_samples = create_dataset_from_cached_features(
        cache_dir, 'test', batch_size, shuffle=False
    )

    logger.info("✅ Datasets created (memory-efficient loading)")

    # Build model
    logger.info("\n🏗️  Building robust BiLSTM model...")
    model = build_robust_bilstm(input_shape=(20, 4096))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("✅ Model built and compiled")
    model.summary()

    # Callbacks with aggressive regularization
    callbacks = [
        # Early stopping - stop if no improvement
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),

        # Checkpoint best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model_{epoch:03d}_{val_accuracy:.4f}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),

        # Log training
        tf.keras.callbacks.CSVLogger(
            str(models_dir / "training_log.csv")
        ),

        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(models_dir / "tensorboard_logs"),
            histogram_freq=1
        )
    ]

    # Calculate steps
    steps_per_epoch = train_samples // batch_size
    validation_steps = val_samples // batch_size

    logger.info("\n" + "=" * 80)
    logger.info("🎯 TRAINING ROBUST MODEL (Anti-Overfitting)")
    logger.info("=" * 80)
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU: RTX 5000 Ada (32GB)")
    logger.info(f"Training samples: {train_samples:,}")
    logger.info(f"Validation samples: {val_samples:,}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info("\n📊 Anti-Overfitting Measures:")
    logger.info("   ✅ High dropout (0.5-0.6)")
    logger.info("   ✅ L2 regularization (0.01)")
    logger.info("   ✅ Batch normalization")
    logger.info("   ✅ Early stopping (patience=20)")
    logger.info("   ✅ Learning rate reduction")
    logger.info("   ✅ Smaller model capacity")
    logger.info("   ✅ Recurrent dropout (0.3)")
    logger.info("=" * 80 + "\n")

    # Train
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL EVALUATION")
    logger.info("=" * 80)

    test_loss, test_acc = model.evaluate(test_dataset, steps=test_samples // batch_size, verbose=0)

    logger.info(f"✅ Test Loss: {test_loss:.4f}")
    logger.info(f"✅ Test Accuracy: {test_acc*100:.2f}%")

    # Check for overfitting
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    gap = train_acc - val_acc

    logger.info(f"\n📈 Overfitting Check:")
    logger.info(f"   Train Accuracy: {train_acc*100:.2f}%")
    logger.info(f"   Val Accuracy:   {val_acc*100:.2f}%")
    logger.info(f"   Gap:            {gap*100:.2f}%")

    if gap < 0.05:
        logger.info("   ✅ Good! Model is NOT overfitting")
    elif gap < 0.10:
        logger.info("   ⚠️  Slight overfitting (acceptable)")
    else:
        logger.info("   ❌ Warning: Significant overfitting detected")

    # Save final model
    final_model_path = models_dir / "robust_final.h5"
    model.save(final_model_path)
    logger.info(f"\n💾 Final model saved: {final_model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {final_model_path}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info("\n🎯 Next: Test with TTA to validate robustness")
    logger.info(f"  python3 predict_with_tta_simple.py --model {final_model_path}")
    logger.info("=" * 80)

    return model, history


if __name__ == "__main__":
    try:
        model, history = train_from_cache()
        logger.info("\n✅ SUCCESS")
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)
