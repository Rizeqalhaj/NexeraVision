#!/usr/bin/env python3
"""
ULTIMATE ACCURACY TRAINING
- Data augmentation
- Extended epochs (200)
- Fine-tuning schedule
- Single GPU (stable)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Single GPU
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
print("ULTIMATE ACCURACY TRAINING")
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

# DATA AUGMENTATION FOR VIDEO FEATURES
class VideoAugmentation(tf.keras.layers.Layer):
    """Augmentation for pre-extracted video features"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        if not training:
            return inputs

        # 1. Temporal jittering: randomly drop and repeat frames
        if tf.random.uniform([]) > 0.5:
            # Drop 1-2 random frames and repeat others
            num_frames = tf.shape(inputs)[1]
            indices = tf.range(num_frames)
            # Randomly shuffle frame order slightly
            noise = tf.random.uniform([num_frames]) * 0.3
            shuffled_indices = tf.argsort(tf.cast(indices, tf.float32) + noise)
            inputs = tf.gather(inputs, shuffled_indices, axis=1)

        # 2. Feature dropout: randomly zero out 5% of features
        if tf.random.uniform([]) > 0.5:
            dropout_mask = tf.cast(
                tf.random.uniform(tf.shape(inputs)) > 0.05,
                inputs.dtype
            )
            inputs = inputs * dropout_mask

        # 3. Gaussian noise: add small random noise
        if tf.random.uniform([]) > 0.5:
            noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=0.01)
            inputs = inputs + noise

        return inputs

# Create augmented dataset
def create_dataset_with_augmentation(features, labels, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)

    if augment:
        # Apply augmentation in map function
        augmentation = VideoAugmentation()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

BATCH_SIZE = 64
EPOCHS = 200

train_dataset = create_dataset_with_augmentation(X_train, y_train, BATCH_SIZE, augment=True)
val_dataset = create_dataset_with_augmentation(X_val, y_val, BATCH_SIZE, augment=False)

# Build model
print("\nBuilding model...")
model = ViolenceDetectionModel(config=Config).build_model()

# Load from best checkpoint
checkpoint_path = Path('checkpoints/best_model.h5')
if checkpoint_path.exists():
    print(f"Loading from: {checkpoint_path}")
    model.load_weights(str(checkpoint_path))
    initial_epoch = 34  # Continue from where we left off
else:
    print("Starting from scratch")
    initial_epoch = 0

print(f"Parameters: {model.count_params():,}")

# Warmup + Cosine decay schedule (extended)
initial_lr = 0.001
warmup_epochs = 5
total_epochs = EPOCHS

def lr_schedule(epoch):
    if epoch < warmup_epochs:
        # Warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        decay_epochs = total_epochs - warmup_epochs
        epoch_in_decay = epoch - warmup_epochs
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
        return initial_lr * cosine_decay + 1e-7

# Compile with gradient clipping
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule(initial_epoch),
    clipnorm=1.0
)

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

model.compile(
    optimizer=optimizer,
    loss=focal_loss,
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Callbacks
callbacks = [
    # Best model
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/ultimate_best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # Periodic checkpoints
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/ultimate_epoch_{epoch:03d}.h5',
        save_freq='epoch',
        verbose=0
    ),

    # Early stopping (very patient)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,  # Very patient!
        restore_best_weights=True,
        verbose=1
    ),

    # Learning rate scheduler
    tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),

    # CSV logger
    tf.keras.callbacks.CSVLogger(
        'checkpoints/ultimate_training_history.csv',
        append=True
    ),

    # TensorBoard
    tf.keras.callbacks.TensorBoard(
        log_dir='checkpoints/tensorboard_ultimate',
        histogram_freq=1
    )
]

print("\n" + "="*80)
print(f"TRAINING: {EPOCHS} epochs with DATA AUGMENTATION")
print("="*80)
print("\n🔥 Improvements active:")
print("  ✅ Temporal jittering augmentation")
print("  ✅ Feature dropout augmentation")
print("  ✅ Gaussian noise augmentation")
print("  ✅ Extended epochs (200)")
print("  ✅ Warmup + Cosine LR schedule")
print("  ✅ Focal loss")
print("  ✅ Gradient clipping")
print("  ✅ Mixed precision (FP16)")
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

# Test evaluation
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80)

test_dataset = create_dataset_with_augmentation(X_test, y_test, BATCH_SIZE, augment=False)
test_results = model.evaluate(test_dataset, verbose=1)

print("\n📊 Final Test Results:")
print(f"  Test Accuracy:  {test_results[1]*100:.2f}%")
print(f"  Test AUC:       {test_results[2]*100:.2f}%")
print(f"  Test Precision: {test_results[3]*100:.2f}%")
print(f"  Test Recall:    {test_results[4]*100:.2f}%")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'epochs_trained': EPOCHS,
    'test_accuracy': float(test_results[1]),
    'test_auc': float(test_results[2]),
    'test_precision': float(test_results[3]),
    'test_recall': float(test_results[4]),
    'augmentation': 'enabled',
    'model_path': 'checkpoints/ultimate_best_model.h5'
}

with open('checkpoints/ultimate_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Best model saved: checkpoints/ultimate_best_model.h5")
print(f"✅ Results saved: checkpoints/ultimate_results.json")
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
