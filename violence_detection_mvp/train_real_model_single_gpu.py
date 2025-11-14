#!/usr/bin/env python3
"""Test with REAL model architecture but single GPU, no callbacks"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force single GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.model_architecture import ViolenceDetectionModel

print("="*80)
print("TEST: Real Model + Single GPU + No Callbacks")
print("="*80)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs: {len(gpus)}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load FULL dataset
print("\nLoading FULL dataset...")
X_train = np.load('feature_cache/train_features.npy')
y_train = np.load('feature_cache/train_labels.npy')
X_val = np.load('feature_cache/val_features.npy')
y_val = np.load('feature_cache/val_labels.npy')

print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")

# Build REAL model
print("\nBuilding REAL 3-layer LSTM + Attention model...")
model_builder = ViolenceDetectionModel(config=Config)
model = model_builder.build_model()

print(f"Parameters: {model.count_params():,}")

# Compile (SIMPLE - no focal loss, no mixed precision)
print("\nCompiling with basic settings...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train (3 epochs, NO callbacks, batch=64)
print("\n" + "="*80)
print("TRAINING: 3 epochs, batch=64, NO callbacks")
print("If this works, multi-GPU or callbacks are the problem")
print("If this hangs, the model architecture is the problem")
print("="*80 + "\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    batch_size=64,
    verbose=1
)

print("\n" + "="*80)
print("✅ SUCCESS! Real model works on single GPU!")
print("="*80)
print(f"\nFinal train accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.2%}")
print("\n🎯 Problem is: Multi-GPU or Callbacks or Mixed Precision")
