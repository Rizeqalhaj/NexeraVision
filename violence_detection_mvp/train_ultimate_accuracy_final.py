#!/usr/bin/env python3
"""
ULTIMATE ACCURACY TRAINING - FINAL VERSION
Mixed precision compatible with proper metric handling
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
print("ULTIMATE ACCURACY TRAINING - FINAL")
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

# Data augmentation
def augment_features(features):
    """Augment video features"""
    augmented = []
    for video_features in features:
        if np.random.random() > 0.5:
            # Temporal jittering
            num_frames = video_features.shape[0]
            indices = np.arange(num_frames)
            for i in range(0, num_frames, 4):
                end = min(i + 4, num_frames)
                np.random.shuffle(indices[i:end])
            video_features = video_features[indices]

        if np.random.random() > 0.5:
            # Gaussian noise
            noise = np.random.normal(0, 0.05, video_features.shape)
            video_features = video_features + noise

        if np.random.random() > 0.5:
            # Feature dropout
            mask = np.random.random(video_features.shape) > 0.03
            video_features = video_features * mask

        augmented.append(video_features)
    return np.array(augmented, dtype=np.float32)

print("\nAugmenting training data...")
X_train_augmented = augment_features(X_train)
X_train_combined = np.concatenate([X_train, X_train_augmented], axis=0)
y_train_combined = np.concatenate([y_train, y_train], axis=0)

print(f"Original: {X_train.shape[0]:,} → Augmented: {X_train_combined.shape[0]:,} samples")

# Datasets
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

initial_epoch = 0
print("Starting from scratch with 2x augmented dataset")
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

# CUSTOM METRICS - Mixed precision compatible
class MixedPrecisionMetric(tf.keras.metrics.Metric):
    """Base class for mixed precision compatible metrics"""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, dtype=tf.float32, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast to float32 for stable computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._update_state(y_true, y_pred, sample_weight)

    def _update_state(self, y_true, y_pred, sample_weight):
        raise NotImplementedError

class BinaryAccuracy(MixedPrecisionMetric):
    def __init__(self, **kwargs):
        super().__init__(name='binary_accuracy', **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros', dtype=tf.float32)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float32)

    def _update_state(self, y_true, y_pred, sample_weight):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        matches = tf.cast(tf.equal(y_true, y_pred_class), tf.float32)
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.divide(self.correct, self.total + tf.keras.backend.epsilon())

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class SimplePrecision(MixedPrecisionMetric):
    def __init__(self, **kwargs):
        super().__init__(name='precision', **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros', dtype=tf.float32)

    def _update_state(self, y_true, y_pred, sample_weight):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        true_pos = tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_class, 1)), tf.float32)
        pred_pos = tf.cast(tf.equal(y_pred_class, 1), tf.float32)

        self.true_positives.assign_add(tf.reduce_sum(true_pos))
        self.predicted_positives.assign_add(tf.reduce_sum(pred_pos))

    def result(self):
        return tf.divide(self.true_positives, self.predicted_positives + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.predicted_positives.assign(0.0)

class SimpleRecall(MixedPrecisionMetric):
    def __init__(self, **kwargs):
        super().__init__(name='recall', **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.actual_positives = self.add_weight(name='ap', initializer='zeros', dtype=tf.float32)

    def _update_state(self, y_true, y_pred, sample_weight):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        true_pos = tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_class, 1)), tf.float32)
        actual_pos = tf.cast(tf.equal(y_true, 1), tf.float32)

        self.true_positives.assign_add(tf.reduce_sum(true_pos))
        self.actual_positives.assign_add(tf.reduce_sum(actual_pos))

    def result(self):
        return tf.divide(self.true_positives, self.actual_positives + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.actual_positives.assign(0.0)

# Compile with custom metrics
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule(initial_epoch),
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss=focal_loss,
    metrics=[
        BinaryAccuracy(),
        SimplePrecision(),
        SimpleRecall()
    ]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/ultimate_best_model.h5',
        monitor='val_binary_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/ultimate_epoch_{epoch:03d}.h5',
        save_freq='epoch',
        verbose=0
    ),
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
print(f"TRAINING: {EPOCHS} epochs")
print("="*80)
print("\n🔥 ALL IMPROVEMENTS ACTIVE:")
print(f"  ✅ 2x augmented data: {X_train_combined.shape[0]:,} samples")
print("  ✅ Temporal jittering + Gaussian noise + Feature dropout")
print("  ✅ 200 epochs (NO early stopping)")
print("  ✅ Focal loss (α=0.25, γ=2.0)")
print("  ✅ Warmup + Cosine LR schedule")
print("  ✅ Gradient clipping (1.0)")
print("  ✅ Mixed precision (FP16) - 2-3× speed boost")
print("  ✅ Custom metrics (mixed precision compatible)")
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

# Final test
print("\n" + "="*80)
print("FINAL TEST EVALUATION")
print("="*80)

test_dataset = create_dataset(X_test, y_test, BATCH_SIZE, shuffle=False)
test_results = model.evaluate(test_dataset, verbose=1)

print("\n📊 FINAL RESULTS:")
print(f"  Test Accuracy:  {test_results[1]*100:.2f}%")
print(f"  Test Precision: {test_results[2]*100:.2f}%")
print(f"  Test Recall:    {test_results[3]*100:.2f}%")

results = {
    'timestamp': datetime.now().isoformat(),
    'epochs_trained': EPOCHS,
    'test_accuracy': float(test_results[1]),
    'test_precision': float(test_results[2]),
    'test_recall': float(test_results[3]),
    'augmentation': '2x_dataset',
    'model_path': 'checkpoints/ultimate_best_model.h5'
}

with open('checkpoints/ultimate_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Best model: checkpoints/ultimate_best_model.h5")
print(f"✅ Results: checkpoints/ultimate_results.json")
print("\n" + "="*80)
print("🎯 TRAINING COMPLETE!")
print("="*80)
