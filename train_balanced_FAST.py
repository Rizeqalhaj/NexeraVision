#!/usr/bin/env python3
"""
FAST Balanced Training - Reuses existing VGG19 features, only re-applies augmentation

FAST because:
- Reuses base features from /workspace/robust_gpu_cache/ (saves 20+ hours)
- Only re-applies moderate augmentation (3x instead of 10x) - takes 10 minutes
- Starts training immediately

Changes from previous model:
1. Reduced dropout: 30-40% (was 50-60%)
2. Reduced augmentation: 3x (was 10x)
3. Per-class accuracy monitoring
4. Focal loss for hard examples
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
print("FAST BALANCED VIOLENCE DETECTION TRAINING")
print("="*80)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("="*80 + "\n")

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Paths - REUSE existing cache!
    'old_cache_dir': '/workspace/robust_gpu_cache',  # OLD features (10x aug)
    'new_cache_dir': '/workspace/balanced_cache',     # NEW features (3x aug)
    'checkpoint_dir': '/workspace/balanced_checkpoints',

    # Model architecture - MODERATE regularization
    'lstm_units': 128,
    'dropout_rate': 0.35,  # REDUCED from 0.5-0.6
    'recurrent_dropout': 0.25,  # REDUCED from 0.3
    'l2_reg': 0.005,  # REDUCED from 0.01

    # Training
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 25,

    # Augmentation - MODERATE (3x instead of 10x)
    'augmentation_multiplier': 3,
    'brightness_range': 0.15,
    'noise_std': 0.01,
}

# ============================================================================
# Per-Class Accuracy Callback
# ============================================================================

class PerClassAccuracyCallback(callbacks.Callback):
    """Monitor accuracy for each class separately"""

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.history = {
            'violent_accuracy': [],
            'nonviolent_accuracy': [],
            'accuracy_gap': []
        }

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = (y_pred[:, 1] > 0.5).astype(int)
        y_true_classes = y_val[:, 1].astype(int)

        violent_mask = y_true_classes == 1
        nonviolent_mask = y_true_classes == 0

        violent_acc = np.mean(y_pred_classes[violent_mask] == y_true_classes[violent_mask]) if violent_mask.sum() > 0 else 0
        nonviolent_acc = np.mean(y_pred_classes[nonviolent_mask] == y_true_classes[nonviolent_mask]) if nonviolent_mask.sum() > 0 else 0
        gap = abs(violent_acc - nonviolent_acc)

        self.history['violent_accuracy'].append(violent_acc)
        self.history['nonviolent_accuracy'].append(nonviolent_acc)
        self.history['accuracy_gap'].append(gap)

        print(f"\n  📊 Per-Class Accuracy:")
        print(f"    Violent:     {violent_acc*100:.2f}%")
        print(f"    Non-violent: {nonviolent_acc*100:.2f}%")
        print(f"    Gap:         {gap*100:.2f}%")

        if gap > 0.15:
            print(f"    ⚠️  WARNING: Large accuracy gap ({gap*100:.1f}%)!")
        else:
            print(f"    ✅ Good balance!")

        logs['violent_accuracy'] = violent_acc
        logs['nonviolent_accuracy'] = nonviolent_acc
        logs['accuracy_gap'] = gap

# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=3.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_loss = self.alpha * focal_weight * bce
        return tf.reduce_mean(focal_loss)

# ============================================================================
# FAST: Reuse Base Features, Re-apply Augmentation
# ============================================================================

def load_base_features_and_reaugment(split, config):
    """
    FAST: Load base features from old cache, apply NEW moderate augmentation

    This saves 20+ hours by reusing VGG19 feature extraction
    """
    new_cache_file = Path(config['new_cache_dir']) / f'{split}_features_3xaug.npy'
    new_labels_file = Path(config['new_cache_dir']) / f'{split}_labels_3xaug.npy'

    # Check if already re-augmented
    if new_cache_file.exists() and new_labels_file.exists():
        print(f"  ✅ Loading re-augmented {split} features...")
        features = np.load(new_cache_file, mmap_mode='r')
        labels = np.load(new_labels_file)
        print(f"     Loaded: {features.shape}")
        return features, labels

    print(f"\n  🔄 Re-augmenting {split} features (FAST - no VGG19 extraction needed)...")

    # Load OLD base features (un-augmented originals)
    old_cache = Path(config['old_cache_dir'])

    # Try to find base features without augmentation
    # The original extraction saves one copy before augmentation
    base_features_file = old_cache / f'{split}_features_base.npy'
    base_labels_file = old_cache / f'{split}_labels_base.npy'

    if not base_features_file.exists():
        # Fallback: load the augmented features and extract base samples
        # (Every 10th sample is a base feature with 10x aug)
        print(f"     Loading from augmented cache...")
        old_features = np.load(old_cache / f'{split}_features.npy', mmap_mode='r')
        old_labels = np.load(old_cache / f'{split}_labels.npy')

        # Extract base features (every 10th sample with 10x augmentation)
        # This is an approximation - better to re-extract if this fails
        base_features = old_features[::10]  # Every 10th sample
        base_labels = old_labels[::10]

        print(f"     Extracted {len(base_features)} base samples from augmented cache")
    else:
        base_features = np.load(base_features_file, mmap_mode='r')
        base_labels = np.load(base_labels_file)
        print(f"     Loaded {len(base_features)} base samples")

    # Apply NEW moderate augmentation (3x)
    print(f"     Applying 3x augmentation...")

    all_features = []
    all_labels = []

    for features, label in zip(base_features, base_labels):
        # Original
        all_features.append(features)
        all_labels.append(label)

        if split == 'train':
            # Aug 1: Slight brightness
            brightness_factor = 1.0 + np.random.uniform(-config['brightness_range'], config['brightness_range'])
            aug1 = features * brightness_factor
            aug1 = np.clip(aug1, 0, 1)
            all_features.append(aug1)
            all_labels.append(label)

            # Aug 2: Small noise
            noise = np.random.normal(0, config['noise_std'], features.shape)
            aug2 = features + noise
            aug2 = np.clip(aug2, 0, 1)
            all_features.append(aug2)
            all_labels.append(label)

    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)

    # Save new cache
    Path(config['new_cache_dir']).mkdir(parents=True, exist_ok=True)
    np.save(new_cache_file, features_array)
    np.save(new_labels_file, labels_array)

    print(f"     ✅ Saved: {features_array.shape}")

    return features_array, labels_array

# ============================================================================
# Model Architecture
# ============================================================================

def build_balanced_bilstm(input_shape, config):
    """BiLSTM with moderate regularization"""
    inputs = layers.Input(shape=input_shape, name='input')

    # BiLSTM layers with moderate dropout
    x = layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'],
            return_sequences=True,
            dropout=config['dropout_rate'],
            recurrent_dropout=config['recurrent_dropout'],
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        ),
        name='bilstm_1'
    )(inputs)
    x = layers.BatchNormalization()(x)

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

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(config['l2_reg']))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(config['l2_reg']))(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(2, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='BalancedViolenceDetector')
    return model

# ============================================================================
# Main Training
# ============================================================================

def main():
    print("\n" + "="*80)
    print("LOADING & RE-AUGMENTING DATA (FAST)")
    print("="*80)

    # Load and re-augment (FAST - reuses VGG19 features)
    X_train, y_train = load_base_features_and_reaugment('train', CONFIG)
    X_val, y_val = load_base_features_and_reaugment('val', CONFIG)

    print(f"\n📊 Dataset Statistics:")
    print(f"  Train: {X_train.shape} | Violent: {y_train.sum()} | Non-violent: {len(y_train) - y_train.sum()}")
    print(f"  Val:   {X_val.shape} | Violent: {y_val.sum()} | Non-violent: {len(y_val) - y_val.sum()}")

    # Convert labels
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
        loss=FocalLoss(gamma=3.0),
        metrics=['accuracy']
    )

    print("\n" + "="*80)
    print("TRAINING WITH PER-CLASS MONITORING")
    print("="*80)

    # Callbacks
    checkpoint_path = Path(CONFIG['checkpoint_dir'])
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        PerClassAccuracyCallback((X_val, y_val_cat)),

        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),

        callbacks.ModelCheckpoint(
            str(checkpoint_path / 'best_balanced_{epoch:03d}_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    start_time = time.time()
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks_list,
        verbose=1
    )

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"⏱️  Time: {elapsed/3600:.1f} hours")
    print(f"📊 Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"💾 Checkpoints: {checkpoint_path}")

    # Show final per-class accuracy
    per_class_callback = [cb for cb in callbacks_list if isinstance(cb, PerClassAccuracyCallback)][0]
    final_violent = per_class_callback.history['violent_accuracy'][-1]
    final_nonviolent = per_class_callback.history['nonviolent_accuracy'][-1]

    print(f"\n🎯 Final Per-Class Performance:")
    print(f"   Violent:     {final_violent*100:.2f}%")
    print(f"   Non-violent: {final_nonviolent*100:.2f}%")
    print(f"   Gap:         {abs(final_violent - final_nonviolent)*100:.2f}%")

    if final_violent > 0.85 and final_nonviolent > 0.85:
        print(f"\n✅ SUCCESS! Both classes performing well!")
    else:
        print(f"\n⚠️  Performance needs improvement")

    return model, history

if __name__ == "__main__":
    model, history = main()
