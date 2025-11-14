#!/usr/bin/env python3
"""
TEST-TIME AUGMENTATION - Boost accuracy by 0.5-1% instantly
Apply augmentation during prediction and average results
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.model_architecture import ViolenceDetectionModel

print("="*80)
print("TEST-TIME AUGMENTATION (TTA)")
print("="*80)

# Load test data
X_test = np.load('feature_cache/test_features.npy')
y_test = np.load('feature_cache/test_labels.npy')

print(f"\nTest set: {X_test.shape}")

# Augmentation function (same as training)
def augment_features(features):
    """Apply augmentation to features"""
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

# Load best model
print("\nLoading best model...")
model_path = 'checkpoints/ultimate_best_model.h5'
model = ViolenceDetectionModel(config=Config).build_model()
model.load_weights(model_path)
print(f"Loaded: {model_path}")

# Baseline prediction (no TTA)
print("\n" + "="*80)
print("BASELINE (No TTA)")
print("="*80)

baseline_pred = model.predict(X_test, batch_size=64, verbose=0)
baseline_classes = np.argmax(baseline_pred, axis=1)
baseline_accuracy = np.mean(baseline_classes == y_test)

print(f"Baseline Accuracy: {baseline_accuracy*100:.2f}%")

# Test-Time Augmentation
print("\n" + "="*80)
print("TEST-TIME AUGMENTATION")
print("="*80)

n_augmentations = 10  # Number of augmented versions to try

print(f"\nGenerating {n_augmentations} augmented predictions...")

all_predictions = [baseline_pred]  # Start with original

for i in range(n_augmentations - 1):
    print(f"  Augmentation {i+1}/{n_augmentations-1}...", end='')

    # Augment test data
    X_test_aug = augment_features(X_test.copy())

    # Predict
    aug_pred = model.predict(X_test_aug, batch_size=64, verbose=0)
    all_predictions.append(aug_pred)

    # Quick accuracy
    aug_classes = np.argmax(aug_pred, axis=1)
    aug_acc = np.mean(aug_classes == y_test)
    print(f" {aug_acc*100:.2f}%")

# Average all predictions
print("\nAveraging predictions...")
tta_predictions = np.mean(all_predictions, axis=0)
tta_classes = np.argmax(tta_predictions, axis=1)
tta_accuracy = np.mean(tta_classes == y_test)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nBaseline (no TTA):     {baseline_accuracy*100:.2f}%")
print(f"With TTA ({n_augmentations} augmentations): {tta_accuracy*100:.2f}%")

improvement = (tta_accuracy - baseline_accuracy) * 100
print(f"\nImprovement: {improvement:+.2f}%")

# Detailed analysis
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

# Per-class metrics
non_violent_mask = y_test == 0
violent_mask = y_test == 1

print("\nBaseline Per-Class:")
print(f"  Non-violent: {np.mean(baseline_classes[non_violent_mask] == y_test[non_violent_mask])*100:.2f}%")
print(f"  Violent:     {np.mean(baseline_classes[violent_mask] == y_test[violent_mask])*100:.2f}%")

print("\nTTA Per-Class:")
print(f"  Non-violent: {np.mean(tta_classes[non_violent_mask] == y_test[non_violent_mask])*100:.2f}%")
print(f"  Violent:     {np.mean(tta_classes[violent_mask] == y_test[violent_mask])*100:.2f}%")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

print("\nBaseline Confusion Matrix:")
cm_baseline = confusion_matrix(y_test, baseline_classes)
print(f"  [[TN={cm_baseline[0,0]:4d}  FP={cm_baseline[0,1]:4d}]")
print(f"   [FN={cm_baseline[1,0]:4d}  TP={cm_baseline[1,1]:4d}]]")

print("\nTTA Confusion Matrix:")
cm_tta = confusion_matrix(y_test, tta_classes)
print(f"  [[TN={cm_tta[0,0]:4d}  FP={cm_tta[0,1]:4d}]")
print(f"   [FN={cm_tta[1,0]:4d}  TP={cm_tta[1,1]:4d}]]")

# Fixes analysis
fixes = np.logical_and(baseline_classes != y_test, tta_classes == y_test)
breaks = np.logical_and(baseline_classes == y_test, tta_classes != y_test)

print(f"\n📊 TTA Impact:")
print(f"  Fixed predictions:  {np.sum(fixes)} samples")
print(f"  Broke predictions:  {np.sum(breaks)} samples")
print(f"  Net improvement:    {np.sum(fixes) - np.sum(breaks)} samples")

print("\n" + "="*80)
print(f"🎯 FINAL TTA ACCURACY: {tta_accuracy*100:.2f}%")
print("="*80)

# Recommendation
if tta_accuracy * 100 >= 93.0:
    print("\n🎉 SUCCESS! Hit 93%+ target with TTA!")
elif improvement > 0.3:
    print(f"\n✅ TTA provides {improvement:.2f}% boost - worth using!")
else:
    print("\n⚠️  TTA provides minimal benefit - try snapshot ensemble instead")
