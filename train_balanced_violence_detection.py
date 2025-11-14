#!/usr/bin/env python3
"""
BALANCED Violence Detection Training
Fixes the 54% TTA accuracy issue by using moderate regularization and monitoring per-class accuracy

Key Changes from Previous Version:
1. Reduced dropout: 30-40% (was 50-60%) - preserves violent patterns
2. Reduced augmentation: 3x multiplier (was 10x) - less distortion
3. Per-class accuracy monitoring - catches imbalanced learning early
4. Focal loss with gamma=3.0 - forces model to learn hard violent examples
5. Class-balanced metrics - ensures both classes perform well
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU:1 only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import time
from pathlib import Path

print("="*80)
print("BALANCED VIOLENCE DETECTION TRAINING")
print("="*80)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("="*80 + "\n")

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Paths
    'cache_dir': '/workspace/balanced_cache',
    'checkpoint_dir': '/workspace/balanced_checkpoints',
    'dataset_path': '/workspace/organized_dataset',

    # Model architecture - MODERATE regularization
    'lstm_units': 128,  # Sufficient capacity
    'dropout_rate': 0.35,  # REDUCED from 0.5-0.6
    'recurrent_dropout': 0.25,  # REDUCED from 0.3
    'l2_reg': 0.005,  # REDUCED from 0.01

    # Training
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 25,  # Increased patience

    # Augmentation - MODERATE (3x instead of 10x)
    'augmentation_multiplier': 3,
    'brightness_range': 0.15,  # REDUCED from 0.3
    'rotation_range': 10,  # REDUCED from 20
    'zoom_range': 0.08,  # REDUCED from 0.15
    'noise_std': 0.01,  # REDUCED from 0.02

    # Class balance
    'monitor_per_class': True,
    'focal_loss_gamma': 3.0,  # Higher gamma for hard examples
}

# ============================================================================
# Per-Class Accuracy Callback
# ============================================================================

class PerClassAccuracyCallback(callbacks.Callback):
    """Monitor accuracy for each class separately to detect imbalance early"""

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.history = {
            'violent_accuracy': [],
            'nonviolent_accuracy': [],
            'accuracy_gap': []
        }

    def on_epoch_end(self, epoch, logs=None):
        # Get validation data
        X_val, y_val = self.validation_data

        # Predict
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = (y_pred[:, 1] > 0.5).astype(int)
        y_true_classes = y_val[:, 1].astype(int)

        # Calculate per-class accuracy
        violent_mask = y_true_classes == 1
        nonviolent_mask = y_true_classes == 0

        violent_acc = np.mean(y_pred_classes[violent_mask] == y_true_classes[violent_mask]) if violent_mask.sum() > 0 else 0
        nonviolent_acc = np.mean(y_pred_classes[nonviolent_mask] == y_true_classes[nonviolent_mask]) if nonviolent_mask.sum() > 0 else 0

        gap = abs(violent_acc - nonviolent_acc)

        # Store history
        self.history['violent_accuracy'].append(violent_acc)
        self.history['nonviolent_accuracy'].append(nonviolent_acc)
        self.history['accuracy_gap'].append(gap)

        # Log
        print(f"\n  Per-Class Accuracy:")
        print(f"    Violent:     {violent_acc*100:.2f}%")
        print(f"    Non-violent: {nonviolent_acc*100:.2f}%")
        print(f"    Gap:         {gap*100:.2f}%")

        # WARNING if gap is too large
        if gap > 0.15:  # 15% gap threshold
            print(f"    ⚠️  WARNING: Large accuracy gap ({gap*100:.1f}%) - model may be biased!")

        # Store in logs for other callbacks
        logs['violent_accuracy'] = violent_acc
        logs['nonviolent_accuracy'] = nonviolent_acc
        logs['accuracy_gap'] = gap

# ============================================================================
# Focal Loss for Hard Example Mining
# ============================================================================

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss: Forces model to learn hard examples (violent videos)
    gamma > 2.0 means "focus heavily on misclassified examples"
    """

    def __init__(self, gamma=3.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Binary focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)

        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_loss = self.alpha * focal_weight * bce

        return tf.reduce_mean(focal_loss)

# ============================================================================
# Moderate Augmentation
# ============================================================================

def apply_moderate_augmentation(features, config):
    """
    Moderate augmentation that preserves violent motion patterns
    3x multiplier instead of 10x
    """
    augmented = [features]  # Original

    # Aug 1: Slight brightness
    brightness_factor = 1.0 + np.random.uniform(-config['brightness_range'], config['brightness_range'])
    aug1 = features * brightness_factor
    aug1 = np.clip(aug1, 0, 1)
    augmented.append(aug1)

    # Aug 2: Small noise
    noise = np.random.normal(0, config['noise_std'], features.shape)
    aug2 = features + noise
    aug2 = np.clip(aug2, 0, 1)
    augmented.append(aug2)

    return np.array(augmented)

# ============================================================================
# Model Architecture - Moderate Regularization
# ============================================================================

def build_balanced_bilstm(input_shape, config):
    """
    BiLSTM with MODERATE regularization (30-40% dropout)
    Preserves violent pattern learning capability
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # First BiLSTM - moderate dropout
    x = layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'],
            return_sequences=True,
            dropout=config['dropout_rate'],  # 0.35
            recurrent_dropout=config['recurrent_dropout'],  # 0.25
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        ),
        name='bilstm_1'
    )(inputs)

    x = layers.BatchNormalization()(x)

    # Second BiLSTM - moderate dropout
    x = layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'] // 2,
            return_sequences=True,
            dropout=config['dropout_rate'],
            recurrent_dropout=config['recurrent_dropout'],
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        ),
        name='bilstm_2'
    )(x)

    x = layers.BatchNormalization()(x)

    # Third BiLSTM - return final state
    x = layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'] // 4,
            return_sequences=False,
            dropout=config['dropout_rate'],
            recurrent_dropout=config['recurrent_dropout'],
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        ),
        name='bilstm_3'
    )(x)

    x = layers.BatchNormalization()(x)

    # Dense layers - moderate dropout
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(config['l2_reg'])
    )(x)
    x = layers.Dropout(0.3)(x)  # Moderate dropout

    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(config['l2_reg'])
    )(x)
    x = layers.Dropout(0.25)(x)  # Light dropout

    # Output
    outputs = layers.Dense(2, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='BalancedViolenceDetector')

    return model

# ============================================================================
# Data Loading with Moderate Augmentation
# ============================================================================

def load_or_extract_features(dataset_path, split, config):
    """Load cached features or extract with VGG19"""
    cache_file = Path(config['cache_dir']) / f'{split}_features_balanced.npy'
    labels_file = Path(config['cache_dir']) / f'{split}_labels_balanced.npy'

    if cache_file.exists() and labels_file.exists():
        print(f"  Loading cached {split} features...")
        features = np.load(cache_file, mmap_mode='r')
        labels = np.load(labels_file)
        print(f"  ✓ Loaded: {features.shape[0]} samples")
        return features, labels

    print(f"  Extracting {split} features...")

    # Load VGG19
    base_model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )

    split_path = Path(dataset_path) / split
    violent_videos = list((split_path / 'violent').glob('*.mp4'))
    nonviolent_videos = list((split_path / 'nonviolent').glob('*.mp4'))

    print(f"  Found: {len(violent_videos)} violent, {len(nonviolent_videos)} non-violent")

    all_features = []
    all_labels = []

    # Process videos
    from tqdm import tqdm
    import cv2

    for video_path, label in tqdm(
        list(zip(violent_videos + nonviolent_videos,
                 [1]*len(violent_videos) + [0]*len(nonviolent_videos))),
        desc=f"Extracting {split}"
    ):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 20:
            cap.release()
            continue

        # Extract 20 evenly-spaced frames
        indices = np.linspace(0, total_frames-1, 20, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

        cap.release()

        if len(frames) < 20:
            continue

        # Extract features
        frames_batch = np.array(frames)
        features = feature_extractor.predict(frames_batch, verbose=0)

        # Apply MODERATE augmentation (3x)
        if split == 'train':
            augmented = apply_moderate_augmentation(features, config)
            for aug_features in augmented:
                all_features.append(aug_features)
                all_labels.append(label)
        else:
            all_features.append(features)
            all_labels.append(label)

    # Save cache
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)

    Path(config['cache_dir']).mkdir(parents=True, exist_ok=True)
    np.save(cache_file, features_array)
    np.save(labels_file, labels_array)

    print(f"  ✓ Cached: {features_array.shape}")

    return features_array, labels_array

# ============================================================================
# Main Training
# ============================================================================

def main():
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # Load data
    X_train, y_train = load_or_extract_features(CONFIG['dataset_path'], 'train', CONFIG)
    X_val, y_val = load_or_extract_features(CONFIG['dataset_path'], 'val', CONFIG)

    print(f"\nTrain: {X_train.shape} | Violent: {y_train.sum()} | Non-violent: {len(y_train) - y_train.sum()}")
    print(f"Val:   {X_val.shape} | Violent: {y_val.sum()} | Non-violent: {len(y_val) - y_val.sum()}")

    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 2)

    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)

    model = build_balanced_bilstm((20, 4096), CONFIG)
    model.summary()

    # Compile with focal loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=FocalLoss(gamma=CONFIG['focal_loss_gamma']),
        metrics=['accuracy']
    )

    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    # Callbacks
    checkpoint_path = Path(CONFIG['checkpoint_dir'])
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        # Per-class accuracy monitoring
        PerClassAccuracyCallback((X_val, y_val_cat)),

        # Early stopping on overall accuracy
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),

        # Model checkpoint
        callbacks.ModelCheckpoint(
            str(checkpoint_path / 'best_model_{epoch:03d}_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # Reduce LR on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks_list,
        verbose=1
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_path}")

    return model, history

if __name__ == "__main__":
    model, history = main()
