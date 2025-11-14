#!/usr/bin/env python3
"""
STRONGER REGULARIZATION TRAINING
Combat overfitting with aggressive regularization techniques
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

print("="*80)
print("STRONGER REGULARIZATION TRAINING")
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

# HIDDEN GEM #4: Mixup augmentation
def mixup(features1, features2, labels1, labels2, alpha=0.2):
    """Mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)"""
    lam = np.random.beta(alpha, alpha)
    mixed_features = lam * features1 + (1 - lam) * features2
    mixed_labels = lam * labels1 + (1 - lam) * labels2
    return mixed_features, mixed_labels

# AGGRESSIVE AUGMENTATION
def augment_aggressive(features):
    """More aggressive augmentation to force generalization"""
    augmented = []

    for video_features in features:
        # Temporal jittering (stronger - 80% chance)
        if np.random.random() > 0.2:
            num_frames = video_features.shape[0]
            indices = np.arange(num_frames)
            # Larger shuffle windows
            for i in range(0, num_frames, 3):
                end = min(i + 3, num_frames)
                np.random.shuffle(indices[i:end])
            video_features = video_features[indices]

        # Gaussian noise (stronger - higher std)
        if np.random.random() > 0.2:
            noise_std = np.random.uniform(0.07, 0.12)  # Increased from 0.05
            noise = np.random.normal(0, noise_std, video_features.shape)
            video_features = video_features + noise

        # Feature dropout (stronger - up to 8%)
        if np.random.random() > 0.2:
            dropout_rate = np.random.uniform(0.05, 0.08)  # Increased from 0.03
            mask = np.random.random(video_features.shape) > dropout_rate
            video_features = video_features * mask

        # NEW: Random frame dropping
        if np.random.random() > 0.3:
            num_frames = video_features.shape[0]
            keep_ratio = np.random.uniform(0.8, 1.0)
            keep_frames = int(num_frames * keep_ratio)
            keep_indices = np.random.choice(num_frames, keep_frames, replace=False)
            keep_indices = np.sort(keep_indices)
            # Repeat last frame to maintain shape
            while len(keep_indices) < num_frames:
                keep_indices = np.append(keep_indices, keep_indices[-1])
            video_features = video_features[keep_indices]

        augmented.append(video_features)

    return np.array(augmented, dtype=np.float32)

print("\nApplying aggressive augmentation...")
X_train_aug = augment_aggressive(X_train)
X_train_combined = np.concatenate([X_train, X_train_aug], axis=0)
y_train_combined = np.concatenate([y_train, y_train], axis=0)

print(f"Augmented: {len(X_train):,} → {len(X_train_combined):,} samples")

# Shuffle
indices = np.random.permutation(len(X_train_combined))
X_train_combined = X_train_combined[indices]
y_train_combined = y_train_combined[indices]

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

# Build model with STRONGER REGULARIZATION
print("\nBuilding model with aggressive regularization...")

# L2 regularization strength
L2_STRENGTH = 0.01  # Strong L2 penalty

def build_regularized_model(config):
    """Build model with strong L2 regularization"""

    inputs = tf.keras.Input(shape=(20, 4096), name='input_features')

    # Bidirectional LSTM layers with L2 regularization
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.6,  # Increased from 0.5
            recurrent_dropout=0.3,  # Added recurrent dropout
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
            recurrent_regularizer=tf.keras.regularizers.l2(L2_STRENGTH)
        ),
        name='bilstm_1'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.6,
            recurrent_dropout=0.3,
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
            recurrent_regularizer=tf.keras.regularizers.l2(L2_STRENGTH)
        ),
        name='bilstm_2'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.6,
            recurrent_dropout=0.3,
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
            recurrent_regularizer=tf.keras.regularizers.l2(L2_STRENGTH)
        ),
        name='bilstm_3'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Attention mechanism
    attention = tf.keras.layers.Dense(
        1,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        name='attention_score'
    )(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax', name='attention_weights')(attention)
    attention = tf.keras.layers.RepeatVector(256)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)

    attended = tf.keras.layers.Multiply(name='attended_features')([x, attention])
    attended = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)

    # Dense layers with strong regularization
    x = tf.keras.layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        name='dense_1'
    )(attended)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # Increased from 0.5

    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        name='dense_2'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)

    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        name='dense_3'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output with label smoothing in loss
    outputs = tf.keras.layers.Dense(
        2,
        activation='softmax',
        dtype='float32',
        name='output'
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='violence_detection_regularized')
    return model

model = build_regularized_model(Config)
print(f"Parameters: {model.count_params():,}")

# LR schedule
initial_lr = 0.0008  # Slightly lower due to stronger regularization
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

# HIDDEN GEM #5: Label smoothing in focal loss
def focal_loss_with_label_smoothing(y_true, y_pred, alpha=0.25, gamma=2.0, label_smoothing=0.15):
    """Focal loss with label smoothing for better generalization"""
    y_true = tf.cast(y_true, tf.int32)
    y_true_oh = tf.one_hot(y_true, 2)

    # Apply label smoothing
    y_true_smooth = y_true_oh * (1 - label_smoothing) + label_smoothing / 2

    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    ce = -y_true_smooth * tf.math.log(y_pred)
    weight = alpha * y_true_smooth * tf.pow(1 - y_pred, gamma)

    return tf.reduce_sum(weight * ce, axis=-1)

# Custom metrics (mixed precision compatible)
class BinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name='binary_accuracy', dtype=tf.float32, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros', dtype=tf.float32)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
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

# Compile
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule(0),
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss=focal_loss_with_label_smoothing,
    metrics=[BinaryAccuracy()]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/regularized_best_model.h5',
        monitor='val_binary_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/regularized_epoch_{epoch:03d}.h5',
        save_freq='epoch',
        verbose=0
    ),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
    tf.keras.callbacks.CSVLogger(
        'checkpoints/regularized_training_history.csv',
        append=False
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='checkpoints/tensorboard_regularized',
        histogram_freq=1
    )
]

print("\n" + "="*80)
print(f"TRAINING: {EPOCHS} epochs with STRONG REGULARIZATION")
print("="*80)
print("\n🔥 Regularization techniques active:")
print(f"  ✅ L2 weight decay: {L2_STRENGTH}")
print("  ✅ Increased dropout: 0.6 (was 0.5)")
print("  ✅ Recurrent dropout: 0.3")
print("  ✅ Label smoothing: 0.15")
print("  ✅ Aggressive augmentation (2x dataset)")
print("  ✅ Stronger Gaussian noise: 0.07-0.12")
print("  ✅ Stronger feature dropout: 5-8%")
print("  ✅ Random frame dropping")
print()

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Test
print("\n" + "="*80)
print("FINAL TEST EVALUATION")
print("="*80)

test_dataset = create_dataset(X_test, y_test, BATCH_SIZE, shuffle=False)
test_results = model.evaluate(test_dataset, verbose=1)

print("\n📊 FINAL RESULTS:")
print(f"  Test Accuracy: {test_results[1]*100:.2f}%")

results = {
    'timestamp': datetime.now().isoformat(),
    'epochs_trained': EPOCHS,
    'test_accuracy': float(test_results[1]),
    'regularization': 'strong_l2_dropout_augmentation',
    'l2_strength': L2_STRENGTH,
    'dropout': 0.6,
    'label_smoothing': 0.15,
    'model_path': 'checkpoints/regularized_best_model.h5'
}

with open('checkpoints/regularized_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Best model: checkpoints/regularized_best_model.h5")
print(f"✅ Results: checkpoints/regularized_results.json")
print("\n" + "="*80)
print("🎯 TRAINING COMPLETE!")
print("="*80)
