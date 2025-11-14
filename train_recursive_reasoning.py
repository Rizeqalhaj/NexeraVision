#!/usr/bin/env python3
"""
Violence Detection with Recursive Reasoning
Inspired by "Less is More: Recursive Reasoning with Tiny Networks" (arXiv 2510.04871)

Key improvements:
1. Multi-scale temporal processing (fast + slow paths)
2. Recursive refinement iterations
3. Hierarchical reasoning (motion → violence)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import numpy as np
from pathlib import Path

# ============================================================================
# RECURSIVE REASONING MODEL ARCHITECTURE
# ============================================================================

def build_recursive_reasoning_model():
    """
    Build violence detection model with recursive reasoning
    Inspired by HRM (Hierarchical Reasoning Model) approach
    """

    inputs = layers.Input(shape=(20, 4096), name='input_features')

    # ========================================================================
    # MULTI-SCALE TEMPORAL PROCESSING
    # ========================================================================

    # Fast Path: Frame-level reasoning (high frequency, detailed)
    fast_features = layers.Dense(256, activation='relu', name='fast_compression')(inputs)
    fast_features = layers.BatchNormalization()(fast_features)

    fast_lstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3),
        name='fast_bilstm')(fast_features)
    fast_lstm = layers.BatchNormalization()(fast_lstm)

    # Slow Path: Segment-level reasoning (low frequency, contextual)
    # Reshape 20 frames → 5 segments of 4 frames each
    slow_features = layers.Dense(256, activation='relu', name='slow_compression')(inputs)
    slow_features = layers.BatchNormalization()(slow_features)

    # Pool every 4 frames to create 5 segments
    slow_pooled = layers.AveragePooling1D(pool_size=4, strides=4, name='segment_pooling')(slow_features)

    slow_lstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3),
        name='slow_bilstm')(slow_pooled)
    slow_lstm = layers.BatchNormalization()(slow_lstm)

    # Upsample slow path back to 20 frames to match fast path
    slow_upsampled = layers.UpSampling1D(size=4, name='slow_upsample')(slow_lstm)

    # Combine fast and slow paths
    combined = layers.Concatenate(name='multi_scale_combine')([fast_lstm, slow_upsampled])

    # ========================================================================
    # RECURSIVE REFINEMENT (3 iterations)
    # ========================================================================

    # Initial processing
    recursive_state = layers.Dense(192, activation='relu', name='recursive_init')(combined)
    recursive_state = layers.BatchNormalization()(recursive_state)

    # Iteration 1
    refined_1 = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True, dropout=0.3),
        name='recursive_lstm_1')(recursive_state)
    refined_1 = layers.BatchNormalization()(refined_1)
    recursive_state = layers.Add(name='residual_1')([recursive_state, refined_1])

    # Iteration 2
    refined_2 = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True, dropout=0.3),
        name='recursive_lstm_2')(recursive_state)
    refined_2 = layers.BatchNormalization()(refined_2)
    recursive_state = layers.Add(name='residual_2')([recursive_state, refined_2])

    # Iteration 3
    refined_3 = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True, dropout=0.3),
        name='recursive_lstm_3')(recursive_state)
    refined_3 = layers.BatchNormalization()(refined_3)
    recursive_state = layers.Add(name='residual_3')([recursive_state, refined_3])

    # ========================================================================
    # HIERARCHICAL ATTENTION
    # ========================================================================

    # Attention mechanism on refined features
    attention_score = layers.Dense(1, activation='tanh', name='attention_score')(recursive_state)
    attention_score = layers.Flatten()(attention_score)
    attention_weights = layers.Activation('softmax', name='attention_weights')(attention_score)
    attention_weights_expanded = layers.RepeatVector(96)(attention_weights)
    attention_weights_expanded = layers.Permute([2, 1])(attention_weights_expanded)
    attended = layers.Multiply(name='attended_features')([recursive_state, attention_weights_expanded])
    attended = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1),
                            output_shape=(96,), name='attention_pooling')(attended)

    # ========================================================================
    # HIERARCHICAL REASONING: Motion Detection → Violence Detection
    # ========================================================================

    # Level 1: Motion/Activity Detection
    motion_branch = layers.Dense(64, activation='relu',
                                kernel_regularizer=regularizers.l2(0.003),
                                name='motion_detector')(attended)
    motion_branch = layers.BatchNormalization()(motion_branch)
    motion_branch = layers.Dropout(0.3)(motion_branch)

    # Level 2: Violence Detection (conditioned on motion)
    violence_branch = layers.Dense(128, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.003),
                                  name='violence_detector')(attended)
    violence_branch = layers.BatchNormalization()(violence_branch)
    violence_branch = layers.Dropout(0.3)(violence_branch)

    # Combine hierarchical reasoning
    hierarchical = layers.Concatenate(name='hierarchical_combine')([motion_branch, violence_branch])

    # ========================================================================
    # FINAL CLASSIFICATION
    # ========================================================================

    x = layers.Dense(96, activation='relu',
                    kernel_regularizer=regularizers.l2(0.003),
                    name='final_dense')(hierarchical)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(2, activation='softmax', dtype='float32', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs,
                        name='RecursiveReasoningViolenceDetector')

    return model

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'cache_dir': '/workspace/violence_detection_mvp/cache',
    'model_save_dir': '/workspace/violence_detection_mvp/models',
    'checkpoint_dir': '/workspace/violence_detection_mvp/checkpoints',

    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,

    # Callbacks
    'patience': 15,
    'reduce_lr_patience': 5,
}

# ============================================================================
# TRAINING
# ============================================================================

if __name__ == '__main__':

    print("=" * 80)
    print("🧠 RECURSIVE REASONING VIOLENCE DETECTION MODEL")
    print("=" * 80)
    print("\nInspired by: 'Less is More: Recursive Reasoning with Tiny Networks'")
    print("arXiv 2510.04871\n")

    print("Key Improvements:")
    print("  1. Multi-scale temporal processing (fast + slow paths)")
    print("  2. Recursive refinement (3 iterations)")
    print("  3. Hierarchical reasoning (motion → violence)")
    print()

    # Create directories
    for dir_path in [CONFIG['model_save_dir'], CONFIG['checkpoint_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # LOAD CACHED FEATURES
    # ========================================================================

    print("=" * 80)
    print("📥 LOADING CACHED FEATURES")
    print("=" * 80)

    cache_dir = Path(CONFIG['cache_dir'])

    X_train = np.load(cache_dir / 'X_train.npy')
    y_train = np.load(cache_dir / 'y_train.npy')
    X_val = np.load(cache_dir / 'X_val.npy')
    y_val = np.load(cache_dir / 'y_val.npy')

    print(f"✓ Training: {X_train.shape[0]} samples")
    print(f"✓ Validation: {X_val.shape[0]} samples")
    print()

    # ========================================================================
    # BUILD MODEL
    # ========================================================================

    print("=" * 80)
    print("🏗️  BUILDING MODEL")
    print("=" * 80)

    model = build_recursive_reasoning_model()

    print(f"\n✓ Model built")
    print(f"  Parameters: {model.count_params():,}")
    print()

    # Show model summary
    model.summary()
    print()

    # ========================================================================
    # COMPILE MODEL
    # ========================================================================

    print("=" * 80)
    print("⚙️  COMPILING MODEL")
    print("=" * 80)

    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✓ Model compiled")
    print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']})")
    print(f"  Loss: sparse_categorical_crossentropy")
    print()

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    model_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=str(Path(CONFIG['checkpoint_dir']) / 'recursive_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=CONFIG['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.CSVLogger(
            str(Path(CONFIG['checkpoint_dir']) / 'recursive_training.log')
        ),
    ]

    # ========================================================================
    # TRAIN MODEL
    # ========================================================================

    print("=" * 80)
    print("🚀 TRAINING MODEL")
    print("=" * 80)
    print()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        callbacks=model_callbacks,
        verbose=1
    )

    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================

    print("\n" + "=" * 80)
    print("💾 SAVING MODEL")
    print("=" * 80)

    final_model_path = Path(CONFIG['model_save_dir']) / 'recursive_reasoning_model.h5'
    model.save_weights(str(final_model_path))

    print(f"✓ Model saved to: {final_model_path}")
    print()

    # ========================================================================
    # TRAINING SUMMARY
    # ========================================================================

    print("=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)

    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

    print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Achieved at Epoch: {best_epoch}")
    print()

    print("Expected Improvements over baseline:")
    print("  Multi-scale processing: +2-3%")
    print("  Recursive refinement: +1-2%")
    print("  Hierarchical reasoning: +1-2%")
    print("  Total expected: +4-7% over baseline 87.84%")
    print("  Target accuracy: 92-95%")
    print()

    print("=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
