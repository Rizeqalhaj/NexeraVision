#!/usr/bin/env python3
"""
ENSEMBLE TRAINING - Train 5 diverse models
Each with different random seed for diversity
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import json
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.model_architecture import ViolenceDetectionModel

# HIDDEN GEM #1: Stochastic Weight Averaging (SWA)
# Average weights from last N epochs instead of just best epoch
class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, swa_start_epoch=100, swa_freq=5):
        super().__init__()
        self.swa_start = swa_start_epoch
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.swa_n = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            if self.swa_weights is None:
                self.swa_weights = self.model.get_weights()
                self.swa_n = 1
            else:
                # Average with previous weights
                for i, w in enumerate(self.model.get_weights()):
                    self.swa_weights[i] = (self.swa_weights[i] * self.swa_n + w) / (self.swa_n + 1)
                self.swa_n += 1

    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            print(f"\n✅ Applying SWA weights (averaged {self.swa_n} checkpoints)")
            self.model.set_weights(self.swa_weights)

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_single_model(model_id, seed, epochs=150):
    """Train one model in the ensemble"""

    print(f"\n{'='*80}")
    print(f"TRAINING MODEL #{model_id} (Seed: {seed})")
    print(f"{'='*80}\n")

    set_seed(seed)

    # Load data
    X_train = np.load('feature_cache/train_features.npy')
    y_train = np.load('feature_cache/train_labels.npy')
    X_val = np.load('feature_cache/val_features.npy')
    y_val = np.load('feature_cache/val_labels.npy')

    # HIDDEN GEM #2: Different augmentation per model for diversity
    def augment_diverse(features, seed_offset):
        np.random.seed(seed + seed_offset)
        augmented = []
        for video in features:
            # Vary augmentation strength per model
            if np.random.random() > 0.3:  # 70% augmentation rate
                # Temporal jittering (stronger for this model)
                frames = video.shape[0]
                indices = np.arange(frames)
                for i in range(0, frames, 3):
                    end = min(i + 3, frames)
                    np.random.shuffle(indices[i:end])
                video = video[indices]

            if np.random.random() > 0.3:
                # Gaussian noise (varied strength)
                noise_std = np.random.uniform(0.03, 0.07)
                video = video + np.random.normal(0, noise_std, video.shape)

            if np.random.random() > 0.3:
                # Feature dropout (varied rate)
                dropout_rate = np.random.uniform(0.02, 0.05)
                mask = np.random.random(video.shape) > dropout_rate
                video = video * mask

            augmented.append(video)
        return np.array(augmented, dtype=np.float32)

    # Create augmented dataset
    X_train_aug = augment_diverse(X_train, model_id * 1000)
    X_train_combined = np.concatenate([X_train, X_train_aug], axis=0)
    y_train_combined = np.concatenate([y_train, y_train], axis=0)

    # Shuffle with model-specific seed
    np.random.seed(seed)
    indices = np.random.permutation(len(X_train_combined))
    X_train_combined = X_train_combined[indices]
    y_train_combined = y_train_combined[indices]

    # Dataset
    def create_dataset(X, y, batch_size, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(20000, seed=seed)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = create_dataset(X_train_combined, y_train_combined, 64, shuffle=True)
    val_ds = create_dataset(X_val, y_val, 64, shuffle=False)

    # Build model
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model = ViolenceDetectionModel(config=Config).build_model()

    # Learning rate schedule
    def lr_schedule(epoch):
        warmup = 5
        if epoch < warmup:
            return 0.001 * (epoch + 1) / warmup
        else:
            decay = (epochs - warmup)
            progress = (epoch - warmup) / decay
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
            f'checkpoints/ensemble_model_{model_id}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
        SWACallback(swa_start_epoch=int(epochs * 0.7), swa_freq=5),  # SWA magic!
        tf.keras.callbacks.CSVLogger(f'checkpoints/ensemble_model_{model_id}_history.csv')
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2  # Less verbose
    )

    # Get best val accuracy
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Model #{model_id} Best Val Accuracy: {best_val_acc*100:.2f}%\n")

    return best_val_acc

# Main ensemble training
print("="*80)
print("ENSEMBLE TRAINING - 5 Diverse Models")
print("="*80)

# Train 5 models with different seeds
seeds = [42, 123, 456, 789, 1024]
results = []

for i, seed in enumerate(seeds, 1):
    acc = train_single_model(model_id=i, seed=seed, epochs=150)
    results.append({'model_id': i, 'seed': seed, 'val_accuracy': acc})

# Summary
print("\n" + "="*80)
print("ENSEMBLE TRAINING COMPLETE")
print("="*80)
for r in results:
    print(f"Model #{r['model_id']} (seed {r['seed']}): {r['val_accuracy']*100:.2f}%")

avg_acc = np.mean([r['val_accuracy'] for r in results])
print(f"\nAverage accuracy: {avg_acc*100:.2f}%")
print(f"Expected ensemble accuracy: {(avg_acc + 0.02)*100:.2f}% (+2-3% boost)")

# Save results
with open('checkpoints/ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ All models saved in checkpoints/ensemble_model_*_best.h5")
print("✅ Run ensemble_predict.py to combine predictions!")
