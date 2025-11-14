#!/usr/bin/env python3
"""Progressive Multi-GPU Testing - Find the exact breaking point"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# NCCL debugging
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.model_architecture import ViolenceDetectionModel

print("="*80)
print("PROGRESSIVE MULTI-GPU DEBUG TEST")
print("="*80)

# GPU setup - BOTH GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"\nFound {len(gpus)} GPUs")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load data (smaller subset for quick testing)
print("\nLoading data (5000 samples for quick test)...")
X_train = np.load('feature_cache/train_features.npy')[:5000]
y_train = np.load('feature_cache/train_labels.npy')[:5000]
X_val = np.load('feature_cache/val_features.npy')[:1000]
y_val = np.load('feature_cache/val_labels.npy')[:1000]

print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")

# TEST 1: Multi-GPU, NO mixed precision, NO callbacks
print("\n" + "="*80)
print("TEST 1: Multi-GPU + Basic Settings")
print("="*80)

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = ViolenceDetectionModel(config=Config).build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

print(f"Model parameters: {model.count_params():,}")

# Simple dataset - NO prefetch, NO cache
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(64)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Add minimal prefetch

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(64)

print("\nStarting training (2 epochs)...")
print("Watching for hangs at step 1...")

try:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2,
        verbose=1
    )

    print("\n" + "="*80)
    print("✅ TEST 1 PASSED: Multi-GPU works!")
    print("="*80)
    print(f"Accuracy: {history.history['accuracy'][-1]:.2%}")

    # TEST 2: Add mixed precision
    print("\n" + "="*80)
    print("TEST 2: Multi-GPU + Mixed Precision")
    print("="*80)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    with strategy.scope():
        model2 = ViolenceDetectionModel(config=Config).build_model()
        model2.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    history2 = model2.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2,
        verbose=1
    )

    print("\n" + "="*80)
    print("✅ TEST 2 PASSED: Multi-GPU + Mixed Precision works!")
    print("="*80)

except Exception as e:
    print("\n" + "="*80)
    print(f"❌ TEST FAILED: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
