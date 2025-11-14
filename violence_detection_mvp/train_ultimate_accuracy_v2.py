#!/usr/bin/env python3
"""
ULTIMATE ACCURACY TRAINING V2
- Simpler, working augmentation
- Extended epochs (200)
- Single GPU (stable)
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

print("="*80)
print("ULTIMATE ACCURACY TRAINING V2")
print("="*80)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs: {len(gpus)}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision: FP16 enabled")

# Load data
print("\nLoading features...")
X_train = np.load('feature_cache/train_features.npy')
y_train = np.load('feature_cache/train_labels.npy')
X_val = np.load('feature_cache/val_features.npy')
y_val = np.load('feature_cache/val_labels.npy')
X_test = np.load('feature_cache/test_features.npy')
y_test = np.load('feature_cache/test_labels.npy')

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

# SIMPLE NUMPY-BASED AUGMENTATION (works!)
def augment_features(features):
    """Augment video features using numpy (called before batching)"""
    augmented = []

    for video_features in features:
        # Random choice: augment or not (50% chance)
        if np.random.random() > 0.5:
            # Temporal jittering: shuffle frames slightly
            num_frames = video_features.shape[0]
            indices = np.arange(num_frames)
            # Shuffle within small windows
            for i in range(0, num_frames, 4):
                end = min(i + 4, num_frames)
                np.random.shuffle(indices[i:end])
            video_features = video_features[indices]

        if np.random.random() > 0.5:
            # Add Gaussian noise (5% std)
            noise = np.random.normal(0, 0.05, video_features.shape)
            video_features = video_features + noise

        if np.random.random() > 0.5:
            # Feature dropout (randomly zero 3% of features)
            mask = np.random.random(video_features.shape) > 0.03
            video_features = video_features * mask

        augmented.append(video_features)

    return np.array(augmented, dtype=np.float32)

# Create augmented training data (one-time augmentation)
print("\nApplying data augmentation to training set...")
print("This will take a few minutes...")

# Augment by creating 2x dataset (original + augmented)
X_train_augmented = augment_features(X_train)
X_train_combined = np.concatenate([X_train, X_train_augmented], axis=0)
y_train_combined = np.concatenate([y_train, y_train], axis=0)

print(f"Original training: {X_train.shape}")
print(f"Augmented training: {X_train_combined.shape}")

# Create datasets
def create_dataset(features, labels, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(20000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

BATCH_SIZE = 64
EPOCHS = 200

train_dataset = create_dataset(X_train_combined, y_train_combined, BATCH_SIZE, shuffle=True)
val_dataset = create_dataset(X_val, y_val, BATCH_SIZE, shuffle=False)

# Build model
print("\nBuilding model...")
model = ViolenceDetectionModel(config=Config).build_model()

# Start fresh with augmented data
initial_epoch = 0
print("Starting from scratch with 2x augmented dataset (21,556 samples)")

print(f"Parameters: {model.count_params():,}")

# LR schedule
initial_lr = 0.001
warmup_epochs = 5
total_epochs = EPOCHS

def lr_schedule(epoch):
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        decay_epochs = total_epochs - warmup_epochs
        epoch_in_decay = epoch - warmup_epochs
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
        return initial_lr * cosine_decay + 1e-7

# Focal loss
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=2)

    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
    weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)

    focal_loss_value = weight * cross_entropy
    return tf.reduce_sum(focal_loss_value, axis=-1)

# Compile
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule(initial_epoch),
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss=focal_loss,
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc', dtype=tf.float32),
             tf.keras.metrics.Precision(name='precision', dtype=tf.float32),
             tf.keras.metrics.Recall(name='recall', dtype=tf.float32)]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/ultimate_best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/ultimate_epoch_{epoch:03d}.h5',
        save_freq='epoch',
        verbose=0
    ),
    # Early stopping DISABLED - train all 200 epochs!
    # tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=50,
    #     restore_best_weights=True,
    #     verbose=1
    # ),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
    tf.keras.callbacks.CSVLogger(
        'checkpoints/ultimate_training_history.csv',
        append=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='checkpoints/tensorboard_ultimate',
        histogram_freq=1
    )
]

print("\n" + "="*80)
print(f"TRAINING: {EPOCHS} epochs with DATA AUGMENTATION")
print("="*80)
print("\n🔥 Active improvements:")
print(f"  ✅ Dataset augmented: {len(X_train):,} → {len(X_train_combined):,} samples (2x)")
print("  ✅ Temporal jittering")
print("  ✅ Gaussian noise")
print("  ✅ Feature dropout")
print("  ✅ Extended epochs (200)")
print("  ✅ Focal loss")
print("  ✅ Gradient clipping")
print("  ✅ Mixed precision")
print(f"\nStarting from epoch: {initial_epoch}")
print()

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    verbose=1
)

# Test
print("\n" + "="*80)
print("FINAL TEST EVALUATION")
print("="*80)

test_dataset = create_dataset(X_test, y_test, BATCH_SIZE, shuffle=False)
test_results = model.evaluate(test_dataset, verbose=1)

print("\n📊 Final Results:")
print(f"  Accuracy:  {test_results[1]*100:.2f}%")
print(f"  AUC:       {test_results[2]*100:.2f}%")
print(f"  Precision: {test_results[3]*100:.2f}%")
print(f"  Recall:    {test_results[4]*100:.2f}%")

results = {
    'timestamp': datetime.now().isoformat(),
    'epochs_trained': EPOCHS,
    'test_accuracy': float(test_results[1]),
    'test_auc': float(test_results[2]),
    'test_precision': float(test_results[3]),
    'test_recall': float(test_results[4]),
    'augmentation': 'enabled_2x_dataset',
    'model_path': 'checkpoints/ultimate_best_model.h5'
}

with open('checkpoints/ultimate_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Model: checkpoints/ultimate_best_model.h5")
print(f"✅ Results: checkpoints/ultimate_results.json")
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
