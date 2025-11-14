#!/usr/bin/env python3
"""
MINIMAL TEST - NO CALLBACKS
Tests if callbacks are causing the hang
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.model_architecture import ViolenceDetectionModel

print("\n" + "="*80)
print("MINIMAL TRAINING TEST (NO CALLBACKS)")
print("="*80 + "\n")

# GPU setup - SINGLE GPU ONLY
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.list_physical_devices('GPU')
print(f"Found {len(gpus)} GPU(s)")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load features
print("\nLoading features...")
train_features = np.load('feature_cache/train_features.npy')
train_labels = np.load('feature_cache/train_labels.npy')
val_features = np.load('feature_cache/val_features.npy')
val_labels = np.load('feature_cache/val_labels.npy')

print(f"Train: {train_features.shape}")
print(f"Val: {val_features.shape}")

# Create datasets (SIMPLE - no prefetch, no cache, no repeat)
def create_simple_dataset(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

train_ds = create_simple_dataset(train_features, train_labels, 64)
val_ds = create_simple_dataset(val_features, val_labels, 64)

# Build model
print("\nBuilding model...")
model = ViolenceDetectionModel(
    sequence_length=Config.FRAMES_PER_VIDEO,
    feature_dim=Config.CHUNK_SIZE,
    rnn_units=Config.RNN_SIZE,
    dropout_rate=Config.DROPOUT_RATE
)

model.build(input_shape=(None, Config.FRAMES_PER_VIDEO, Config.CHUNK_SIZE))
print(f"Model parameters: {model.count_params():,}")

# Compile (SIMPLE - no focal loss, no class weights, no mixed precision)
print("\nCompiling...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train (NO CALLBACKS AT ALL)
print("\n" + "="*80)
print("STARTING TRAINING (3 EPOCHS, NO CALLBACKS, SINGLE GPU)")
print("If this hangs, callbacks are NOT the problem")
print("="*80 + "\n")

try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        verbose=1
        # NO CALLBACKS!
    )

    print("\n" + "="*80)
    print("✅ SUCCESS! Training completed without hanging!")
    print("="*80)
    print(f"\nFinal accuracy: {history.history['accuracy'][-1]:.2%}")
    print(f"Final val_accuracy: {history.history['val_accuracy'][-1]:.2%}")
    print("\n🎯 Callbacks are NOT the problem - the issue is elsewhere")

except Exception as e:
    print("\n" + "="*80)
    print(f"❌ FAILED: {e}")
    print("="*80)
