#!/usr/bin/env python3
"""
WORKING BOOST - Reconstruct exact architecture from checkpoint inspection
Architecture matches the saved weights perfectly
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path

print("="*80)
print("ACCURACY BOOST: 91.35% → 93%+ (Exact Architecture Match)")
print("="*80)

# Custom AttentionLayer that matches the checkpoint
class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer matching the saved checkpoint structure"""

    def __init__(self, name='attention', **kwargs):
        super(AttentionLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        # Create attention dense layer
        self.attention_dense = tf.keras.layers.Dense(1, use_bias=False, name='dense')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Calculate attention scores
        attention_scores = self.attention_dense(inputs)
        # Apply softmax
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        # Calculate weighted sum
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        return config

# Load test data
print("\nLoading test data...")
X_test = np.load('feature_cache/test_features.npy')
y_test = np.load('feature_cache/test_labels.npy')

print(f"Test set: {X_test.shape}")

# Build the EXACT model architecture from checkpoint inspection
def build_exact_model():
    """
    Build exact architecture from checkpoint inspection:
    - 3 LSTM layers (128 units each, NOT Bidirectional)
    - Custom AttentionLayer
    - 3 Dense layers (256, 128, 64)
    - 5 BatchNorm layers
    - 6 Dropout layers
    - Output layer (2 classes)
    """

    inputs = tf.keras.Input(shape=(20, 4096), name='video_input')

    # LSTM 1
    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)

    # LSTM 2
    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_2')(x)

    # LSTM 3
    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_3')(x)

    # Attention
    attention_output = AttentionLayer(name='attention')(x)

    # Dense 1
    x = tf.keras.layers.Dense(256, name='dense_1')(attention_output)
    x = tf.keras.layers.BatchNormalization(name='bn_4')(x)
    x = tf.keras.layers.Activation('relu', name='relu_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_4')(x)

    # Dense 2
    x = tf.keras.layers.Dense(128, name='dense_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_5')(x)
    x = tf.keras.layers.Activation('relu', name='relu_2')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_5')(x)

    # Dense 3
    x = tf.keras.layers.Dense(64, name='dense_3')(x)
    x = tf.keras.layers.Activation('relu', name='relu_3')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_6')(x)

    # Output
    outputs = tf.keras.layers.Dense(2, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='violence_detection_model')
    return model

# Build model
print("\nBuilding exact model architecture...")
model = build_exact_model()

print(f"Model layers: {len(model.layers)}")
print(f"Model parameters: {model.count_params():,}")

# Load weights
print("\nLoading weights from checkpoint...")
model_path = 'checkpoints/ultimate_best_model.h5'

try:
    model.load_weights(model_path)
    print(f"✅ Successfully loaded weights from: {model_path}")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("\nModel summary:")
    model.summary()
    exit(1)

# Augmentation function
def augment_features(features, strength='normal'):
    """Apply augmentation"""
    augmented = []

    for video_features in features:
        video = video_features.copy()

        if strength == 'light':
            prob, noise_std, dropout = 0.3, 0.02, 0.01
        elif strength == 'normal':
            prob, noise_std, dropout = 0.5, 0.05, 0.03
        else:  # heavy
            prob, noise_std, dropout = 0.7, 0.08, 0.05

        if np.random.random() > (1 - prob):
            num_frames = video.shape[0]
            indices = np.arange(num_frames)
            for i in range(0, num_frames, 4):
                end = min(i + 4, num_frames)
                np.random.shuffle(indices[i:end])
            video = video[indices]

        if np.random.random() > (1 - prob):
            noise = np.random.normal(0, noise_std, video.shape)
            video = video + noise

        if np.random.random() > (1 - prob):
            mask = np.random.random(video.shape) > dropout
            video = video * mask

        augmented.append(video)

    return np.array(augmented, dtype=np.float32)

# ============================================================================
# BASELINE
# ============================================================================
print("\n" + "="*80)
print("BASELINE PREDICTION")
print("="*80)

baseline_pred = model.predict(X_test, batch_size=64, verbose=0)
baseline_classes = np.argmax(baseline_pred, axis=1)
baseline_accuracy = np.mean(baseline_classes == y_test)

print(f"\nBaseline Accuracy: {baseline_accuracy*100:.2f}%")

# ============================================================================
# TEST-TIME AUGMENTATION
# ============================================================================
print("\n" + "="*80)
print("TEST-TIME AUGMENTATION (TTA)")
print("="*80)

print("\nGenerating augmented predictions...")
tta_predictions = [baseline_pred]

aug_configs = [
    ('light', 5),
    ('normal', 3),
    ('heavy', 2)
]

for strength, n_iter in aug_configs:
    for i in range(n_iter):
        X_aug = augment_features(X_test, strength=strength)
        pred = model.predict(X_aug, batch_size=64, verbose=0)
        tta_predictions.append(pred)
        acc = np.mean(np.argmax(pred, axis=1) == y_test)
        print(f"  {strength:6s} aug {i+1}: {acc*100:.2f}%")

tta_avg = np.mean(tta_predictions, axis=0)
tta_classes = np.argmax(tta_avg, axis=1)
tta_accuracy = np.mean(tta_classes == y_test)

print(f"\n✅ TTA Accuracy ({len(tta_predictions)} predictions): {tta_accuracy*100:.2f}%")
print(f"   Improvement: {(tta_accuracy - baseline_accuracy)*100:+.2f}%")

# ============================================================================
# MONTE CARLO DROPOUT
# ============================================================================
print("\n" + "="*80)
print("MONTE CARLO DROPOUT")
print("="*80)

print("\nRunning MC Dropout (10 forward passes)...")
mc_predictions = []

for i in range(10):
    pred = model(X_test, training=True).numpy()
    mc_predictions.append(pred)
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    if i < 5 or i == 9:
        print(f"  Pass {i+1:2d}: {acc*100:.2f}%")
    elif i == 5:
        print("  ...")

mc_avg = np.mean(mc_predictions, axis=0)
mc_classes = np.argmax(mc_avg, axis=1)
mc_accuracy = np.mean(mc_classes == y_test)

print(f"\n✅ MC Dropout Accuracy: {mc_accuracy*100:.2f}%")
print(f"   Improvement: {(mc_accuracy - baseline_accuracy)*100:+.2f}%")

# ============================================================================
# FINAL ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("FINAL WEIGHTED ENSEMBLE")
print("="*80)

all_techniques = {
    'baseline': (baseline_pred, 1.0),
    'tta': (tta_avg, 2.0),
    'mc_dropout': (mc_avg, 1.5)
}

weighted_sum = np.zeros_like(baseline_pred)
total_weight = 0

print("\nCombining predictions:")
for name, (pred, weight) in all_techniques.items():
    weighted_sum += pred * weight
    total_weight += weight
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    print(f"  {name:15s}: {acc*100:.2f}% (weight: {weight:.1f})")

final_pred = weighted_sum / total_weight
final_classes = np.argmax(final_pred, axis=1)
final_accuracy = np.mean(final_classes == y_test)

print(f"\n✅ Final Ensemble: {final_accuracy*100:.2f}%")
print(f"   Total Improvement: {(final_accuracy - baseline_accuracy)*100:+.2f}%")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

non_violent_mask = y_test == 0
violent_mask = y_test == 1

print(f"\n📊 Per-Class Performance:")
print(f"{'Class':<15s} {'Baseline':>10s} {'Final':>10s} {'Change':>10s}")
print("-" * 50)

baseline_nv = np.mean(baseline_classes[non_violent_mask] == y_test[non_violent_mask])
final_nv = np.mean(final_classes[non_violent_mask] == y_test[non_violent_mask])
print(f"{'Non-violent':<15s} {baseline_nv*100:>9.2f}% {final_nv*100:>9.2f}% {(final_nv-baseline_nv)*100:>+9.2f}%")

baseline_v = np.mean(baseline_classes[violent_mask] == y_test[violent_mask])
final_v = np.mean(final_classes[violent_mask] == y_test[violent_mask])
print(f"{'Violent':<15s} {baseline_v*100:>9.2f}% {final_v*100:>9.2f}% {(final_v-baseline_v)*100:>+9.2f}%")

print(f"{'Overall':<15s} {baseline_accuracy*100:>9.2f}% {final_accuracy*100:>9.2f}% {(final_accuracy-baseline_accuracy)*100:>+9.2f}%")

from sklearn.metrics import confusion_matrix

print("\n📊 Confusion Matrices:")
print("\nBaseline:")
cm_baseline = confusion_matrix(y_test, baseline_classes)
print(f"  [[TN={cm_baseline[0,0]:4d}  FP={cm_baseline[0,1]:4d}]")
print(f"   [FN={cm_baseline[1,0]:4d}  TP={cm_baseline[1,1]:4d}]]")

print("\nFinal Ensemble:")
cm_final = confusion_matrix(y_test, final_classes)
print(f"  [[TN={cm_final[0,0]:4d}  FP={cm_final[0,1]:4d}]")
print(f"   [FN={cm_final[1,0]:4d}  TP={cm_final[1,1]:4d}]]")

print("\nChanges:")
fp_change = cm_final[0,1] - cm_baseline[0,1]
fn_change = cm_final[1,0] - cm_baseline[1,0]
print(f"  False Positives: {cm_baseline[0,1]} → {cm_final[0,1]} ({fp_change:+d})")
print(f"  False Negatives: {cm_baseline[1,0]} → {cm_final[1,0]} ({fn_change:+d})")

fixes = np.logical_and(baseline_classes != y_test, final_classes == y_test)
breaks = np.logical_and(baseline_classes == y_test, final_classes != y_test)

print(f"\n📈 Prediction Changes:")
print(f"  Fixed:  {np.sum(fixes):3d} incorrect → correct")
print(f"  Broke:  {np.sum(breaks):3d} correct → incorrect")
print(f"  Net:    {np.sum(fixes) - np.sum(breaks):+4d} samples")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎯 FINAL SUMMARY")
print("="*80)

print(f"\n📈 Accuracy Journey:")
print(f"  Baseline:     {baseline_accuracy*100:.2f}%")
print(f"  + TTA:        {tta_accuracy*100:.2f}% ({(tta_accuracy-baseline_accuracy)*100:+.2f}%)")
print(f"  + MC Dropout: {mc_accuracy*100:.2f}% ({(mc_accuracy-baseline_accuracy)*100:+.2f}%)")
print(f"  Final:        {final_accuracy*100:.2f}% ({(final_accuracy-baseline_accuracy)*100:+.2f}%)")

if final_accuracy * 100 >= 93.0:
    print(f"\n🎉 SUCCESS! Reached {final_accuracy*100:.2f}% - TARGET ACHIEVED!")
elif final_accuracy * 100 >= 92.5:
    print(f"\n✅ Very close! {final_accuracy*100:.2f}% (only {93.0 - final_accuracy*100:.2f}% from target)")
elif final_accuracy * 100 >= 92.0:
    print(f"\n👍 Good progress! {final_accuracy*100:.2f}%")
else:
    print(f"\n⚠️  At {final_accuracy*100:.2f}%")

print("\n" + "="*80)
