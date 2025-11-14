#!/usr/bin/env python3
"""
MEMORY-SAFE BOOST - Process in smaller batches to avoid OOM
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path

print("="*80)
print("ACCURACY BOOST: 90.78% → 93%+ (Memory Safe)")
print("="*80)

# Custom AttentionLayer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, name='attention', **kwargs):
        super(AttentionLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(1, use_bias=False, name='dense')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def get_config(self):
        return super().get_config()

# Load test data
print("\nLoading test data...")
X_test = np.load('feature_cache/test_features.npy')
y_test = np.load('feature_cache/test_labels.npy')

print(f"Test set: {X_test.shape}")

# Build model
def build_exact_model():
    inputs = tf.keras.Input(shape=(20, 4096), name='video_input')

    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)

    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_2')(x)

    x = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_3')(x)

    attention_output = AttentionLayer(name='attention')(x)

    x = tf.keras.layers.Dense(256, name='dense_1')(attention_output)
    x = tf.keras.layers.BatchNormalization(name='bn_4')(x)
    x = tf.keras.layers.Activation('relu', name='relu_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_4')(x)

    x = tf.keras.layers.Dense(128, name='dense_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_5')(x)
    x = tf.keras.layers.Activation('relu', name='relu_2')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_5')(x)

    x = tf.keras.layers.Dense(64, name='dense_3')(x)
    x = tf.keras.layers.Activation('relu', name='relu_3')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_6')(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax', name='output')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='violence_detection_model')

print("\nBuilding model...")
model = build_exact_model()
model.load_weights('checkpoints/ultimate_best_model.h5')
print(f"✅ Loaded: {model.count_params():,} parameters")

# Augmentation
def augment_features(features, strength='normal'):
    augmented = []
    for video_features in features:
        video = video_features.copy()

        if strength == 'light':
            prob, noise_std, dropout = 0.3, 0.02, 0.01
        elif strength == 'normal':
            prob, noise_std, dropout = 0.5, 0.05, 0.03
        else:
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

# BASELINE
print("\n" + "="*80)
print("BASELINE")
print("="*80)

baseline_pred = model.predict(X_test, batch_size=64, verbose=0)
baseline_classes = np.argmax(baseline_pred, axis=1)
baseline_accuracy = np.mean(baseline_classes == y_test)

print(f"Baseline: {baseline_accuracy*100:.2f}%")

# TEST-TIME AUGMENTATION
print("\n" + "="*80)
print("TEST-TIME AUGMENTATION (TTA)")
print("="*80)

print("\nGenerating predictions...")
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
        print(f"  {strength:6s} {i+1}: {acc*100:.2f}%")

tta_avg = np.mean(tta_predictions, axis=0)
tta_classes = np.argmax(tta_avg, axis=1)
tta_accuracy = np.mean(tta_classes == y_test)

print(f"\n✅ TTA ({len(tta_predictions)} aug): {tta_accuracy*100:.2f}% (+{(tta_accuracy-baseline_accuracy)*100:.2f}%)")

# MC DROPOUT (Memory Safe - Process in batches)
print("\n" + "="*80)
print("MONTE CARLO DROPOUT (Batch Processing)")
print("="*80)

print("\nRunning 10 passes with dropout enabled...")

# Split into smaller batches to avoid OOM
BATCH_SIZE = 128  # Smaller batches
n_samples = len(X_test)
n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE

mc_predictions = []

for pass_num in range(10):
    batch_preds = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, n_samples)
        X_batch = X_test[start_idx:end_idx]

        # Predict with dropout enabled
        batch_pred = model(X_batch, training=True).numpy()
        batch_preds.append(batch_pred)

    # Concatenate all batch predictions
    full_pred = np.concatenate(batch_preds, axis=0)
    mc_predictions.append(full_pred)

    acc = np.mean(np.argmax(full_pred, axis=1) == y_test)
    if pass_num < 5 or pass_num == 9:
        print(f"  Pass {pass_num+1:2d}: {acc*100:.2f}%")
    elif pass_num == 5:
        print("  ...")

mc_avg = np.mean(mc_predictions, axis=0)
mc_classes = np.argmax(mc_avg, axis=1)
mc_accuracy = np.mean(mc_classes == y_test)

print(f"\n✅ MC Dropout: {mc_accuracy*100:.2f}% (+{(mc_accuracy-baseline_accuracy)*100:.2f}%)")

# FINAL ENSEMBLE
print("\n" + "="*80)
print("FINAL ENSEMBLE")
print("="*80)

all_techniques = {
    'baseline': (baseline_pred, 1.0),
    'tta': (tta_avg, 2.0),
    'mc_dropout': (mc_avg, 1.5)
}

weighted_sum = np.zeros_like(baseline_pred)
total_weight = 0

print("\nCombining:")
for name, (pred, weight) in all_techniques.items():
    weighted_sum += pred * weight
    total_weight += weight
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    print(f"  {name:12s}: {acc*100:.2f}% (weight {weight:.1f})")

final_pred = weighted_sum / total_weight
final_classes = np.argmax(final_pred, axis=1)
final_accuracy = np.mean(final_classes == y_test)

print(f"\n✅ Final: {final_accuracy*100:.2f}% (+{(final_accuracy-baseline_accuracy)*100:.2f}%)")

# ANALYSIS
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

non_violent_mask = y_test == 0
violent_mask = y_test == 1

print(f"\nPer-Class:")
print(f"{'':15s} {'Baseline':>10s} {'Final':>10s} {'Change':>10s}")
print("-" * 50)

baseline_nv = np.mean(baseline_classes[non_violent_mask] == y_test[non_violent_mask])
final_nv = np.mean(final_classes[non_violent_mask] == y_test[non_violent_mask])
print(f"{'Non-violent':<15s} {baseline_nv*100:>9.2f}% {final_nv*100:>9.2f}% {(final_nv-baseline_nv)*100:>+9.2f}%")

baseline_v = np.mean(baseline_classes[violent_mask] == y_test[violent_mask])
final_v = np.mean(final_classes[violent_mask] == y_test[violent_mask])
print(f"{'Violent':<15s} {baseline_v*100:>9.2f}% {final_v*100:>9.2f}% {(final_v-baseline_v)*100:>+9.2f}%")

print(f"{'Overall':<15s} {baseline_accuracy*100:>9.2f}% {final_accuracy*100:>9.2f}% {(final_accuracy-baseline_accuracy)*100:>+9.2f}%")

from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:")
print("\nBaseline:")
cm_base = confusion_matrix(y_test, baseline_classes)
print(f"  [[TN={cm_base[0,0]:4d}  FP={cm_base[0,1]:4d}]")
print(f"   [FN={cm_base[1,0]:4d}  TP={cm_base[1,1]:4d}]]")

print("\nFinal:")
cm_final = confusion_matrix(y_test, final_classes)
print(f"  [[TN={cm_final[0,0]:4d}  FP={cm_final[0,1]:4d}]")
print(f"   [FN={cm_final[1,0]:4d}  TP={cm_final[1,1]:4d}]]")

print(f"\nChanges:")
print(f"  FP: {cm_base[0,1]} → {cm_final[0,1]} ({cm_final[0,1]-cm_base[0,1]:+d})")
print(f"  FN: {cm_base[1,0]} → {cm_final[1,0]} ({cm_final[1,0]-cm_base[1,0]:+d})")

fixes = np.sum(np.logical_and(baseline_classes != y_test, final_classes == y_test))
breaks = np.sum(np.logical_and(baseline_classes == y_test, final_classes != y_test))

print(f"\n  Fixed:  {fixes:3d}")
print(f"  Broke:  {breaks:3d}")
print(f"  Net:    {fixes - breaks:+4d}")

# SUMMARY
print("\n" + "="*80)
print("🎯 SUMMARY")
print("="*80)

print(f"\n📈 Journey:")
print(f"  Start:      {baseline_accuracy*100:.2f}%")
print(f"  + TTA:      {tta_accuracy*100:.2f}% ({(tta_accuracy-baseline_accuracy)*100:+.2f}%)")
print(f"  + MC Drop:  {mc_accuracy*100:.2f}% ({(mc_accuracy-baseline_accuracy)*100:+.2f}%)")
print(f"  Final:      {final_accuracy*100:.2f}% ({(final_accuracy-baseline_accuracy)*100:+.2f}%)")

if final_accuracy * 100 >= 93.0:
    print(f"\n🎉 SUCCESS! {final_accuracy*100:.2f}% - TARGET HIT!")
elif final_accuracy * 100 >= 92.5:
    print(f"\n✅ Close! {final_accuracy*100:.2f}% (need +{93.0-final_accuracy*100:.2f}%)")
    print("   → Train 3 models with different seeds → ensemble to 93%+")
elif final_accuracy * 100 >= 92.0:
    print(f"\n👍 Progress! {final_accuracy*100:.2f}%")
    print("   → Train 5-model ensemble for +1.5-2.5% boost")
else:
    print(f"\n📊 At {final_accuracy*100:.2f}%")
    print("   → Need multi-model ensemble for 93%")

print("\n" + "="*80)
