#!/usr/bin/env python3
"""
SIMPLE 3-MODEL ENSEMBLE
Train 3 models with different random seeds for diversity
Each uses the SAME successful architecture and training approach
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("3-MODEL ENSEMBLE TRAINING")
print("="*80)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load data
print("\nLoading data...")
X_train = np.load('feature_cache/train_features.npy')
y_train = np.load('feature_cache/train_labels.npy')
X_val = np.load('feature_cache/val_features.npy')
y_val = np.load('feature_cache/val_labels.npy')

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")

# Simple augmentation
def augment(features):
    augmented = []
    for video in features:
        if np.random.random() > 0.5:
            num_frames = video.shape[0]
            indices = np.arange(num_frames)
            for i in range(0, num_frames, 4):
                end = min(i + 4, num_frames)
                np.random.shuffle(indices[i:end])
            video = video[indices]

        if np.random.random() > 0.5:
            video = video + np.random.normal(0, 0.05, video.shape)

        if np.random.random() > 0.5:
            mask = np.random.random(video.shape) > 0.03
            video = video * mask

        augmented.append(video)
    return np.array(augmented, dtype=np.float32)

# Augment training data
print("\nAugmenting training data...")
X_train_aug = augment(X_train)
X_train_combined = np.concatenate([X_train, X_train_aug])
y_train_combined = np.concatenate([y_train, y_train])

print(f"Augmented: {X_train.shape[0]:,} → {X_train_combined.shape[0]:,}")

# Dataset creation
def create_dataset(X, y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(20000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Model architecture
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, name='attention', **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(1, use_bias=False, name='dense')
        super().build(input_shape)

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        return tf.reduce_sum(inputs * attention_weights, axis=1)

def build_model():
    inputs = tf.keras.Input(shape=(20, 4096), name='video_input')

    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)

    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_2')(x)

    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_3')(x)

    x = AttentionLayer(name='attention')(x)

    x = tf.keras.layers.Dense(256, name='dense_1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_4')(x)
    x = tf.keras.layers.Activation('relu', name='relu_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_4')(x)

    x = tf.keras.layers.Dense(128, name='dense_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_5')(x)
    x = tf.keras.layers.Activation('relu', name='relu_2')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_5')(x)

    x = tf.keras.layers.Dense(64, name='dense_3')(x)
    x = tf.keras.layers.Activation('relu', name='relu_3')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_6')(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32', name='output')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Training function
def train_model(model_id, seed, epochs=100):
    print(f"\n{'='*80}")
    print(f"MODEL {model_id}/3 (Seed: {seed})")
    print(f"{'='*80}\n")

    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Shuffle with this seed
    indices = np.random.permutation(len(X_train_combined))
    X_train_shuffled = X_train_combined[indices]
    y_train_shuffled = y_train_combined[indices]

    # Datasets
    train_ds = create_dataset(X_train_shuffled, y_train_shuffled, 64, True)
    val_ds = create_dataset(X_val, y_val, 64, False)

    # Build model
    model = build_model()

    # LR schedule
    def lr_schedule(epoch):
        warmup = 5
        if epoch < warmup:
            return 0.001 * (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / (epochs - warmup)
            return 0.001 * (0.5 * (1 + np.cos(np.pi * progress))) + 1e-7

    # Focal loss
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, 2)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true_oh * tf.math.log(y_pred)
        weight = 0.25 * y_true_oh * tf.pow(1 - y_pred, 2.0)
        return tf.reduce_sum(weight * ce, axis=-1)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0), clipnorm=1.0),
        loss=focal_loss,
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'checkpoints/ensemble_m{model_id}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
        tf.keras.callbacks.CSVLogger(f'checkpoints/ensemble_m{model_id}_history.csv')
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    best_val = max(history.history['val_accuracy'])
    print(f"\n✅ Model {model_id} best val accuracy: {best_val*100:.2f}%")

    return best_val

# Train 3 models
print("\nTraining 3 models with different seeds...")
print("Expected time: ~9-12 hours (3 models × 3-4 hours each)\n")

seeds = [42, 123, 456]
results = []

for i, seed in enumerate(seeds, 1):
    val_acc = train_model(model_id=i, seed=seed, epochs=100)
    results.append({'model_id': i, 'seed': seed, 'val_accuracy': val_acc})

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

for r in results:
    print(f"Model {r['model_id']} (seed {r['seed']:3d}): {r['val_accuracy']*100:.2f}%")

avg_val = np.mean([r['val_accuracy'] for r in results])
print(f"\nAverage val accuracy: {avg_val*100:.2f}%")
print(f"Expected ensemble:    {(avg_val + 0.015)*100:.2f}% (+1.5% boost)")
print(f"Expected with TTA:    {(avg_val + 0.020)*100:.2f}% (+2.0% boost)")

print("\n✅ Models saved:")
for i in range(1, 4):
    print(f"   checkpoints/ensemble_m{i}_best.h5")

print("\n✅ Next: Run ensemble prediction script to combine all 3 models!")
print("="*80)
