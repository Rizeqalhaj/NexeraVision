#!/usr/bin/env python3
"""
3-MODEL ENSEMBLE PREDICTION
Combine 3 models + TTA for maximum accuracy
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path

print("="*80)
print("3-MODEL ENSEMBLE + TTA PREDICTION")
print("="*80)

# Custom AttentionLayer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, name='attention', **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(1, use_bias=False, name='dense')
        super().build(input_shape)

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        return tf.reduce_sum(inputs * attention_weights, axis=1)

# Load test data
X_test = np.load('feature_cache/test_features.npy')
y_test = np.load('feature_cache/test_labels.npy')

print(f"\nTest set: {X_test.shape}")

# Build model architecture
def build_model():
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
    x = AttentionLayer(name='attention')(x)
    x = tf.keras.layers.Dense(256, name='dense_1')(x)
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
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32', name='output')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Load all 3 models
print("\nLoading ensemble models...")
models = []

for i in range(1, 4):
    model_path = f'checkpoints/ensemble_m{i}_best.h5'
    if Path(model_path).exists():
        model = build_model()
        model.load_weights(model_path)
        models.append((i, model))
        print(f"  ✅ Model {i}: {model_path}")
    else:
        print(f"  ❌ Model {i} not found: {model_path}")

if len(models) == 0:
    print("\n❌ No ensemble models found!")
    print("Train models first: python train_3_model_ensemble.py")
    exit(1)

print(f"\n✅ Loaded {len(models)} models")

# Get predictions from each model
print("\n" + "="*80)
print("INDIVIDUAL MODEL PREDICTIONS")
print("="*80)

all_predictions = []

for model_id, model in models:
    pred = model.predict(X_test, batch_size=64, verbose=0)
    all_predictions.append(pred)
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    print(f"  Model {model_id}: {acc*100:.2f}%")

# Simple ensemble average
ensemble_pred = np.mean(all_predictions, axis=0)
ensemble_classes = np.argmax(ensemble_pred, axis=1)
ensemble_acc = np.mean(ensemble_classes == y_test)

print(f"\n✅ {len(models)}-Model Ensemble: {ensemble_acc*100:.2f}%")
print(f"   Improvement: +{(ensemble_acc - np.mean([np.mean(np.argmax(p, axis=1) == y_test) for p in all_predictions]))*100:.2f}%")

# TTA on ensemble
print("\n" + "="*80)
print("TEST-TIME AUGMENTATION ON ENSEMBLE")
print("="*80)

def augment(features):
    augmented = []
    for video in features:
        if np.random.random() > 0.5:
            num_frames = video.shape[0]
            indices = np.arange(num_frames)
            for i in range(0, num_frames, 4):
                end = min(i + 4, num_frames)
                np.random.shuffle(indices[i:end])
            video = video[indices]
        if np.random.random() > 0.5:
            video = video + np.random.normal(0, 0.05, video.shape)
        if np.random.random() > 0.5:
            mask = np.random.random(video.shape) > 0.03
            video = video * mask
        augmented.append(video)
    return np.array(augmented, dtype=np.float32)

print("\nGenerating TTA predictions (5 augmentations)...")

tta_ensemble_preds = [ensemble_pred]  # Start with original ensemble

for aug_num in range(5):
    X_aug = augment(X_test)

    # Get predictions from all models on augmented data
    aug_preds = []
    for model_id, model in models:
        pred = model.predict(X_aug, batch_size=64, verbose=0)
        aug_preds.append(pred)

    # Average across models
    aug_ensemble = np.mean(aug_preds, axis=0)
    tta_ensemble_preds.append(aug_ensemble)

    acc = np.mean(np.argmax(aug_ensemble, axis=1) == y_test)
    print(f"  TTA {aug_num+1}: {acc*100:.2f}%")

# Average all TTA predictions
final_pred = np.mean(tta_ensemble_preds, axis=0)
final_classes = np.argmax(final_pred, axis=1)
final_acc = np.mean(final_classes == y_test)

print(f"\n✅ Final (Ensemble + TTA): {final_acc*100:.2f}%")
print(f"   Total improvement: +{(final_acc - ensemble_acc)*100:.2f}%")

# Analysis
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

from sklearn.metrics import confusion_matrix, classification_report

non_violent_mask = y_test == 0
violent_mask = y_test == 1

print(f"\nPer-Class Accuracy:")
nv_acc = np.mean(final_classes[non_violent_mask] == y_test[non_violent_mask])
v_acc = np.mean(final_classes[violent_mask] == y_test[violent_mask])
print(f"  Non-violent: {nv_acc*100:.2f}%")
print(f"  Violent:     {v_acc*100:.2f}%")
print(f"  Overall:     {final_acc*100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, final_classes)
print(f"  [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
print(f"   [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")

print("\nClassification Report:")
print(classification_report(y_test, final_classes, target_names=['Non-violent', 'Violent']))

# Summary
print("\n" + "="*80)
print("🎯 FINAL SUMMARY")
print("="*80)

individual_avg = np.mean([np.mean(np.argmax(p, axis=1) == y_test) for p in all_predictions])

print(f"\n📈 Accuracy Journey:")
print(f"  Individual models avg: {individual_avg*100:.2f}%")
print(f"  {len(models)}-model ensemble:    {ensemble_acc*100:.2f}% (+{(ensemble_acc-individual_avg)*100:.2f}%)")
print(f"  + TTA (5 aug):         {final_acc*100:.2f}% (+{(final_acc-ensemble_acc)*100:.2f}%)")
print(f"  Total improvement:     +{(final_acc-individual_avg)*100:.2f}%")

if final_acc * 100 >= 93.0:
    print(f"\n🎉 SUCCESS! {final_acc*100:.2f}% - TARGET ACHIEVED!")
elif final_acc * 100 >= 92.5:
    print(f"\n✅ Very close! {final_acc*100:.2f}% (only {93.0-final_acc*100:.2f}% from 93%)")
else:
    print(f"\n📊 At {final_acc*100:.2f}% (need +{93.0-final_acc*100:.2f}% for 93%)")

print("\n" + "="*80)
