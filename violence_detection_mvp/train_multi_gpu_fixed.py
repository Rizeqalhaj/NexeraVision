#!/usr/bin/env python3
"""
Multi-GPU Training with AllReduce Deadlock Fixes
"""

import os
# CRITICAL: Set these BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# NCCL fixes for AllReduce hang
os.environ['NCCL_DEBUG'] = 'WARN'  # Reduce noise
os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes timeout
os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Force blocking wait
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Better error handling

# TensorFlow multi-GPU fixes
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Separate threads per GPU
os.environ['TF_GPU_THREAD_COUNT'] = '2'  # One thread per GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Memory growth

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.model_architecture import ViolenceDetectionModel

print("="*80)
print("MULTI-GPU TRAINING WITH FIXES")
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

# MirroredStrategy with fixes
print("\nCreating MirroredStrategy with communication options...")
communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL,
    timeout_seconds=1800.0  # 30 minutes
)

strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.ReductionToOneDevice()  # Use reduction instead of AllReduce
)

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Create datasets OUTSIDE strategy scope
def create_dataset(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Global batch size
BATCH_SIZE = 64

train_dataset = create_dataset(X_train, y_train, BATCH_SIZE)
val_dataset = create_dataset(X_val, y_val, BATCH_SIZE)

# Distribute datasets
train_dataset = strategy.experimental_distribute_dataset(train_dataset)
val_dataset = strategy.experimental_distribute_dataset(val_dataset)

# Build model inside strategy scope
with strategy.scope():
    print("\nBuilding model...")
    model = ViolenceDetectionModel(config=Config).build_model()

    print("Compiling with ReductionToOneDevice...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

print(f"Parameters: {model.count_params():,}")

print("\n" + "="*80)
print("STARTING TRAINING")
print("Testing if AllReduce hang is fixed...")
print("="*80 + "\n")

try:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2,
        verbose=1
    )

    print("\n" + "="*80)
    print("✅ SUCCESS! Multi-GPU works with fixes!")
    print("="*80)
    print(f"\nAccuracy: {history.history['accuracy'][-1]:.2%}")
    print("\n🎯 Ready for full 100-epoch training!")

except Exception as e:
    print("\n" + "="*80)
    print(f"❌ FAILED: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
