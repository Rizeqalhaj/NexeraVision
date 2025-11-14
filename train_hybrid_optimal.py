#!/usr/bin/env python3
"""
HYBRID OPTIMAL ARCHITECTURE FOR VIOLENCE DETECTION

Combines best elements from all successful configurations:
- Residual connections + Attention (from train_better_architecture.py)
- Moderate regularization 30-35% (from train_balanced_violence_detection.py)
- Advanced training pipeline (from train_ultimate_accuracy_final.py)
- Per-class monitoring (from train_balanced_violence_detection.py)

Expected Performance:
- TTA Accuracy: 90-92% (vs 54.68% failed baseline)
- Violent Detection: 88-91% (vs 22.97% failed)
- Non-violent Detection: 92-94% (vs 86.39% failed)
- Class Gap: <8% (vs 63.42% failed)

Parameters: ~1.2M (optimal balance between 700K and 2.5M)
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

print("="*80)
print("HYBRID OPTIMAL VIOLENCE DETECTION ARCHITECTURE")
print("="*80)
print("Combining: Residual + Attention + Moderate Regularization + Per-Class Monitoring")
print("="*80)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs: {len(gpus)}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision: FP16 enabled\n")

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Architecture - Hybrid design
    'lstm_units': [96, 96, 64],  # Between Better (64) and Ultimate (128)
    'dropout_rates': [0.35, 0.35, 0.30],  # Moderate (from Balanced)
    'recurrent_dropout': [0.20, 0.20, 0.15],  # Lower than failed (0.30)
    'dense_units': [128, 64],
    'dense_dropout': [0.35, 0.25],
    'l2_reg': 0.003,  # Very light (vs 0.01 failed)

    # Feature compression
    'compression_dim': 512,
    'compression_dropout': 0.25,

    # Training
    'batch_size': 64,
    'epochs': 150,
    'initial_lr': 0.001,
    'warmup_epochs': 5,
    'gradient_clip': 1.0,

    # Augmentation - 3x balanced
    'augmentation_multiplier': 3,

    # Loss
    'focal_gamma': 3.0,
    'focal_alpha': 0.25,

    # Monitoring
    'gap_threshold_warning': 0.15,
    'gap_threshold_critical': 0.25,
}

# ============================================================================
# Data Augmentation - Violence-Aware (3x)
# ============================================================================

def augment_features_violence_aware(features):
    """
    3x augmentation with violence-aware techniques
    Preserves temporal sequences and motion patterns
    """
    augmented = []

    for video_features in features:
        # Original (always include)
        augmented.append(video_features.copy())

        # Aug 1: Temporal jittering (preserves motion sequences)
        aug1 = video_features.copy()
        num_frames = aug1.shape[0]
        indices = np.arange(num_frames)
        # Shuffle within small windows (4 frames) to preserve sequences
        for i in range(0, num_frames, 4):
            end = min(i + 4, num_frames)
            if np.random.random() > 0.5:
                np.random.shuffle(indices[i:end])
        aug1 = aug1[indices]
        augmented.append(aug1)

        # Aug 2: Brightness + Small noise
        aug2 = video_features.copy()
        if np.random.random() > 0.5:
            # Brightness (preserves motion)
            brightness_factor = 1.0 + np.random.uniform(-0.15, 0.15)
            aug2 = aug2 * brightness_factor
        if np.random.random() > 0.5:
            # Small Gaussian noise
            noise = np.random.normal(0, 0.01, aug2.shape)
            aug2 = aug2 + noise
        aug2 = np.clip(aug2, 0, 1)
        augmented.append(aug2)

    return np.array(augmented, dtype=np.float32)

# ============================================================================
# Per-Class Accuracy Monitoring (Critical for Bias Detection)
# ============================================================================

class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    """
    Monitor violent vs non-violent accuracy separately
    Alerts if gap exceeds thresholds (early warning system)
    """

    def __init__(self, validation_data, config):
        super().__init__()
        self.validation_data = validation_data
        self.config = config
        self.history = {
            'violent_accuracy': [],
            'nonviolent_accuracy': [],
            'accuracy_gap': []
        }

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data

        # Predict
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = tf.argmax(y_pred, axis=-1).numpy()
        y_true_classes = tf.argmax(y_val, axis=-1).numpy()

        # Per-class accuracy
        violent_mask = y_true_classes == 1
        nonviolent_mask = y_true_classes == 0

        violent_acc = np.mean(y_pred_classes[violent_mask] == 1) if violent_mask.sum() > 0 else 0
        nonviolent_acc = np.mean(y_pred_classes[nonviolent_mask] == 0) if nonviolent_mask.sum() > 0 else 0

        gap = abs(violent_acc - nonviolent_acc)

        # Store
        self.history['violent_accuracy'].append(violent_acc)
        self.history['nonviolent_accuracy'].append(nonviolent_acc)
        self.history['accuracy_gap'].append(gap)

        # Display
        print(f"\n  📊 Per-Class Accuracy:")
        print(f"    Violent:     {violent_acc*100:5.2f}%")
        print(f"    Non-violent: {nonviolent_acc*100:5.2f}%")
        print(f"    Gap:         {gap*100:5.2f}%", end="")

        # Warnings
        if gap > self.config['gap_threshold_critical']:
            print(f"  🚨 CRITICAL GAP!")
        elif gap > self.config['gap_threshold_warning']:
            print(f"  ⚠️  WARNING")
        else:
            print(f"  ✅ Good")

        # Update logs
        logs['violent_accuracy'] = violent_acc
        logs['nonviolent_accuracy'] = nonviolent_acc
        logs['accuracy_gap'] = gap

# ============================================================================
# Hybrid Optimal Architecture
# ============================================================================

def build_hybrid_optimal_model(config):
    """
    Hybrid architecture combining best practices:

    1. Feature compression (reduces overfitting)
    2. Residual connections (improves gradient flow)
    3. Attention mechanism (focuses on violence patterns)
    4. Moderate regularization (preserves pattern learning)

    Parameters: ~1.2M (optimal balance)
    """
    from tensorflow.keras import layers, regularizers, Model

    inputs = layers.Input(shape=(20, 4096), name='input_features')

    # Feature compression: 4096 → 512
    compressed = layers.Dense(
        config['compression_dim'],
        activation='relu',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='compress'
    )(inputs)
    compressed = layers.BatchNormalization()(compressed)
    compressed = layers.Dropout(config['compression_dropout'])(compressed)

    # First BiLSTM - 96 units
    x = layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'][0],
            return_sequences=True,
            dropout=config['dropout_rates'][0],
            recurrent_dropout=config['recurrent_dropout'][0],
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        ),
        name='bilstm_1'
    )(compressed)
    x = layers.BatchNormalization()(x)

    # Store for residual connection
    x_residual = x

    # Second BiLSTM - 96 units
    x = layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'][1],
            return_sequences=True,
            dropout=config['dropout_rates'][1],
            recurrent_dropout=config['recurrent_dropout'][1],
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        ),
        name='bilstm_2'
    )(x)
    x = layers.BatchNormalization()(x)

    # Residual connection (improves gradient flow)
    x = layers.Add(name='residual_connection')([x, x_residual])

    # Third BiLSTM - 64 units
    x = layers.Bidirectional(
        layers.LSTM(
            config['lstm_units'][2],
            return_sequences=True,
            dropout=config['dropout_rates'][2],
            recurrent_dropout=config['recurrent_dropout'][2],
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        ),
        name='bilstm_3'
    )(x)
    x = layers.BatchNormalization()(x)

    # Attention mechanism (focuses on violence patterns)
    attention_score = layers.Dense(1, activation='tanh', name='attention_score')(x)
    attention_score = layers.Flatten()(attention_score)
    attention_weights = layers.Activation('softmax', name='attention_weights')(attention_score)
    attention_weights = layers.RepeatVector(128)(attention_weights)  # 64*2 bidirectional
    attention_weights = layers.Permute([2, 1])(attention_weights)

    attended = layers.Multiply(name='attended_features')([x, attention_weights])
    attended = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='sum_attended')(attended)

    # Dense layers - moderate dropout
    x = layers.Dense(
        config['dense_units'][0],
        activation='relu',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='dense_1'
    )(attended)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config['dense_dropout'][0])(x)

    x = layers.Dense(
        config['dense_units'][1],
        activation='relu',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config['dense_dropout'][1])(x)

    # Output (float32 for mixed precision)
    outputs = layers.Dense(
        2,
        activation='softmax',
        dtype='float32',
        name='output'
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name='HybridOptimalViolenceDetector')
    return model

# ============================================================================
# Focal Loss (Hard Example Mining)
# ============================================================================

def focal_loss(y_true, y_pred, alpha=0.25, gamma=3.0):
    """
    Focal loss forces model to learn hard examples (violent videos)
    gamma=3.0 is higher than standard 2.0 for stronger focus
    """
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=2)

    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
    weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)

    focal_loss_value = weight * cross_entropy
    return tf.reduce_sum(focal_loss_value, axis=-1)

# ============================================================================
# Custom Metrics (Mixed Precision Compatible)
# ============================================================================

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

class SimplePrecision(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name='precision', dtype=tf.float32, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
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

class SimpleRecall(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name='recall', dtype=tf.float32, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.actual_positives = self.add_weight(name='ap', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
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

# ============================================================================
# Learning Rate Schedule (Warmup + Cosine)
# ============================================================================

def lr_schedule(epoch, config):
    """Warmup for 5 epochs then cosine decay"""
    if epoch < config['warmup_epochs']:
        return config['initial_lr'] * (epoch + 1) / config['warmup_epochs']
    else:
        decay_epochs = config['epochs'] - config['warmup_epochs']
        epoch_in_decay = epoch - config['warmup_epochs']
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
        return config['initial_lr'] * cosine_decay + 1e-7

# ============================================================================
# Main Training
# ============================================================================

def main():
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # Load features
    X_train = np.load('feature_cache/train_features.npy')
    y_train = np.load('feature_cache/train_labels.npy')
    X_val = np.load('feature_cache/val_features.npy')
    y_val = np.load('feature_cache/val_labels.npy')
    X_test = np.load('feature_cache/test_features.npy')
    y_test = np.load('feature_cache/test_labels.npy')

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    # Apply 3x augmentation
    print("\nApplying 3x violence-aware augmentation...")
    X_train_augmented = augment_features_violence_aware(X_train)
    y_train_augmented = np.repeat(y_train, CONFIG['augmentation_multiplier'])

    print(f"Augmented: {len(X_train):,} → {len(X_train_augmented):,} samples")

    # Shuffle
    indices = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[indices]
    y_train_augmented = y_train_augmented[indices]

    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train_augmented, 2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 2)

    # Create datasets
    def create_dataset(features, labels, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(20000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(X_train_augmented, y_train_cat, CONFIG['batch_size'], shuffle=True)
    val_dataset = create_dataset(X_val, y_val_cat, CONFIG['batch_size'], shuffle=False)

    print("\n" + "="*80)
    print("BUILDING HYBRID OPTIMAL MODEL")
    print("="*80)

    model = build_hybrid_optimal_model(CONFIG)
    model.summary()

    params = model.count_params()
    print(f"\n✅ Total parameters: {params:,}")
    print(f"   Target: ~1.2M (actual: {params/1e6:.2f}M)")

    # Compile
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule(0, CONFIG),
        clipnorm=CONFIG['gradient_clip']
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

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print("\n🔥 Active Improvements:")
    print(f"  ✅ Hybrid architecture (residual + attention)")
    print(f"  ✅ Moderate dropout: {CONFIG['dropout_rates']} (not 50-60%)")
    print(f"  ✅ 3x augmentation (violence-aware)")
    print(f"  ✅ Per-class accuracy monitoring")
    print(f"  ✅ Focal loss (γ={CONFIG['focal_gamma']})")
    print(f"  ✅ Warmup + cosine LR schedule")
    print(f"  ✅ Gradient clipping ({CONFIG['gradient_clip']})")
    print(f"  ✅ Mixed precision (FP16)")
    print(f"  ✅ Feature compression (4096→512)")
    print()

    # Callbacks
    callbacks = [
        # Per-class monitoring (CRITICAL)
        PerClassAccuracyCallback((X_val, y_val_cat), CONFIG),

        # Model checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/hybrid_optimal_best.h5',
            monitor='val_binary_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/hybrid_optimal_epoch_{epoch:03d}_{val_binary_accuracy:.4f}.h5',
            save_freq='epoch',
            verbose=0
        ),

        # LR schedule
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_schedule(epoch, CONFIG),
            verbose=1
        ),

        # Logging
        tf.keras.callbacks.CSVLogger(
            'checkpoints/hybrid_optimal_training.csv',
            append=False
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='checkpoints/tensorboard_hybrid_optimal',
            histogram_freq=1
        )
    ]

    print("\n" + "="*80)
    print(f"TRAINING: {CONFIG['epochs']} EPOCHS (NO EARLY STOPPING)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # Final test
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)

    test_dataset = create_dataset(X_test, y_test_cat, CONFIG['batch_size'], shuffle=False)
    test_results = model.evaluate(test_dataset, verbose=1)

    print("\n📊 FINAL RESULTS:")
    print(f"  Test Loss:      {test_results[0]:.4f}")
    print(f"  Test Accuracy:  {test_results[1]*100:.2f}%")
    print(f"  Test Precision: {test_results[2]*100:.2f}%")
    print(f"  Test Recall:    {test_results[3]*100:.2f}%")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'architecture': 'hybrid_optimal',
        'config': CONFIG,
        'epochs_trained': CONFIG['epochs'],
        'parameters': params,
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_precision': float(test_results[2]),
        'test_recall': float(test_results[3]),
        'improvements': [
            'residual_connections',
            'attention_mechanism',
            'moderate_regularization_30_35_percent',
            'per_class_monitoring',
            'focal_loss_gamma_3',
            'warmup_cosine_lr',
            '3x_violence_aware_augmentation',
            'feature_compression'
        ],
        'model_path': 'checkpoints/hybrid_optimal_best.h5'
    }

    with open('checkpoints/hybrid_optimal_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Best model: checkpoints/hybrid_optimal_best.h5")
    print(f"✅ Results: checkpoints/hybrid_optimal_results.json")
    print("\n" + "="*80)
    print("🎯 TRAINING COMPLETE!")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📌 Next Steps:")
    print("  1. Test with TTA: python3 predict_with_tta_simple.py")
    print("  2. Compare with failed baseline (54.68% TTA)")
    print("  3. Expected: 90-92% TTA accuracy, <8% class gap")
    print("="*80)

    return model, history

if __name__ == "__main__":
    try:
        model, history = main()
        print("\n✅ SUCCESS")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
