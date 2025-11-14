#!/usr/bin/env python3
"""ULTRA MINIMAL - Just test if training can complete at all"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Single GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

print("="*80)
print("ULTRA MINIMAL TRAINING TEST")
print("="*80)

# Load data
print("\nLoading features...")
X_train = np.load('feature_cache/train_features.npy')[:1000]  # Only 1000 samples
y_train = np.load('feature_cache/train_labels.npy')[:1000]
X_val = np.load('feature_cache/val_features.npy')[:200]       # Only 200 samples
y_val = np.load('feature_cache/val_labels.npy')[:200]

print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")

# Simple model
print("\nBuilding simple LSTM...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20, 4096)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Parameters: {model.count_params():,}")

# Train
print("\n" + "="*80)
print("TRAINING (2 epochs, batch=32, NO callbacks)")
print("="*80 + "\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=32,
    verbose=1
)

print("\n" + "="*80)
print("✅ SUCCESS!")
print("="*80)
print(f"Final accuracy: {history.history['accuracy'][-1]:.2%}")
