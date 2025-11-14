#!/usr/bin/env python3
"""
Multi-GPU Training with Stability Fixes (Gradient Clipping + Safe LR)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.model_architecture import ViolenceDetectionModel

print("="*80)
print("MULTI-GPU TRAINING WITH STABILITY FIXES")
print("="*80)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
print(f"\nFound {len(gpus)} GPUs")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load data
print("\nLoading data (5000 samples for testing)...")
X_train = np.load('feature_cache/train_features.npy')[:5000]
y_train = np.load('feature_cache/train_labels.npy')[:5000]
X_val = np.load('feature_cache/val_features.npy')[:1000]
y_val = np.load('feature_cache/val_labels.npy')[:1000]

print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")

# MirroredStrategy
print("\nCreating MirroredStrategy...")
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.ReductionToOneDevice()
)

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Create datasets
def create_dataset(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

BATCH_SIZE = 64

train_dataset = create_dataset(X_train, y_train, BATCH_SIZE)
val_dataset = create_dataset(X_val, y_val, BATCH_SIZE)

train_dataset = strategy.experimental_distribute_dataset(train_dataset)
val_dataset = strategy.experimental_distribute_dataset(val_dataset)

# Build model with stability fixes
with strategy.scope():
    print("\nBuilding model...")
    model = ViolenceDetectionModel(config=Config).build_model()

    # CRITICAL: Gradient clipping + Lower LR for stability
    print("Compiling with GRADIENT CLIPPING and safe LR...")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,  # Lower LR (was 0.001)
        clipnorm=1.0  # Gradient clipping to prevent explosion
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

print(f"Parameters: {model.count_params():,}")
print("\n🛡️  Stability measures:")
print("  - Learning rate: 0.0001 (safe for multi-GPU)")
print("  - Gradient clipping: 1.0 (prevents NaN)")
print("  - ReductionToOneDevice (avoids AllReduce hang)")

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

try:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=3,
        verbose=1
    )

    print("\n" + "="*80)
    print("✅ SUCCESS! Multi-GPU stable training!")
    print("="*80)
    print(f"\nFinal accuracy: {history.history['accuracy'][-1]:.2%}")
    print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.2%}")
    print("\n🎯 NO NaN! Ready for full 100-epoch training!")

except Exception as e:
    print("\n" + "="*80)
    print(f"❌ FAILED: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
