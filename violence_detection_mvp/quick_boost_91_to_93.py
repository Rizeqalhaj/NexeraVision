#!/usr/bin/env python3
"""
QUICK ACCURACY BOOST: 91.35% → 93%+
Uses only your BEST model (ultimate_best_model.h5) with multiple techniques
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
print("QUICK BOOST: 91.35% → 93%+ using SINGLE MODEL")
print("="*80)

# Load test data
X_test = np.load('feature_cache/test_features.npy')
y_test = np.load('feature_cache/test_labels.npy')

print(f"\nTest set: {X_test.shape}")

# Augmentation function
def augment_features(features, strength='normal'):
    """Apply augmentation with configurable strength"""
    augmented = []

    for video_features in features:
        video = video_features.copy()

        if strength == 'light':
            prob = 0.3
            noise_std = 0.02
            dropout = 0.01
        elif strength == 'normal':
            prob = 0.5
            noise_std = 0.05
            dropout = 0.03
        else:  # heavy
            prob = 0.7
            noise_std = 0.08
            dropout = 0.05

        if np.random.random() > (1 - prob):
            # Temporal jittering
            num_frames = video.shape[0]
            indices = np.arange(num_frames)
            for i in range(0, num_frames, 4):
                end = min(i + 4, num_frames)
                np.random.shuffle(indices[i:end])
            video = video[indices]

        if np.random.random() > (1 - prob):
            # Gaussian noise
            noise = np.random.normal(0, noise_std, video.shape)
            video = video + noise

        if np.random.random() > (1 - prob):
            # Feature dropout
            mask = np.random.random(video.shape) > dropout
            video = video * mask

        augmented.append(video)

    return np.array(augmented, dtype=np.float32)

# Load best model
print("\nLoading best model...")
model_path = 'checkpoints/ultimate_best_model.h5'

if not Path(model_path).exists():
    print(f"❌ Model not found: {model_path}")
    print("Looking for alternative models...")

    checkpoint_dir = Path('checkpoints')
    alternatives = [
        'best_model.h5',
        'model_best.h5',
        'ultimate_model.h5'
    ]

    for alt in alternatives:
        alt_path = checkpoint_dir / alt
        if alt_path.exists():
            model_path = str(alt_path)
            print(f"✅ Found: {model_path}")
            break
    else:
        print("❌ No model found! Please check your checkpoints directory.")
        sys.exit(1)

model = ViolenceDetectionModel(config=Config).build_model()
model.load_weights(model_path)
print(f"✅ Loaded: {model_path}")

# ==============================================================================
# TECHNIQUE 1: Baseline (no augmentation)
# ==============================================================================
print("\n" + "="*80)
print("TECHNIQUE 1: BASELINE")
print("="*80)

baseline_pred = model.predict(X_test, batch_size=64, verbose=0)
baseline_classes = np.argmax(baseline_pred, axis=1)
baseline_accuracy = np.mean(baseline_classes == y_test)

print(f"Baseline Accuracy: {baseline_accuracy*100:.2f}%")

# ==============================================================================
# TECHNIQUE 2: Test-Time Augmentation (TTA)
# ==============================================================================
print("\n" + "="*80)
print("TECHNIQUE 2: TEST-TIME AUGMENTATION (TTA)")
print("="*80)

print("\nGenerating augmented predictions...")
tta_predictions = [baseline_pred]

# Light augmentation (5 iterations)
for i in range(5):
    X_aug = augment_features(X_test, strength='light')
    pred = model.predict(X_aug, batch_size=64, verbose=0)
    tta_predictions.append(pred)
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    print(f"  Light aug {i+1}: {acc*100:.2f}%")

# Normal augmentation (3 iterations)
for i in range(3):
    X_aug = augment_features(X_test, strength='normal')
    pred = model.predict(X_aug, batch_size=64, verbose=0)
    tta_predictions.append(pred)
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    print(f"  Normal aug {i+1}: {acc*100:.2f}%")

# Heavy augmentation (2 iterations)
for i in range(2):
    X_aug = augment_features(X_test, strength='heavy')
    pred = model.predict(X_aug, batch_size=64, verbose=0)
    tta_predictions.append(pred)
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    print(f"  Heavy aug {i+1}: {acc*100:.2f}%")

# Average all TTA predictions
tta_avg = np.mean(tta_predictions, axis=0)
tta_classes = np.argmax(tta_avg, axis=1)
tta_accuracy = np.mean(tta_classes == y_test)

print(f"\n✅ TTA Accuracy ({len(tta_predictions)} augmentations): {tta_accuracy*100:.2f}%")
print(f"   Improvement: {(tta_accuracy - baseline_accuracy)*100:+.2f}%")

# ==============================================================================
# TECHNIQUE 3: Confidence-Based Thresholding
# ==============================================================================
print("\n" + "="*80)
print("TECHNIQUE 3: CONFIDENCE-BASED THRESHOLDING")
print("="*80)

# Use TTA predictions with confidence filtering
confidence_scores = np.max(tta_avg, axis=1)  # Max probability = confidence

# Sort by confidence
low_confidence_mask = confidence_scores < 0.7
high_confidence_mask = confidence_scores >= 0.7

print(f"\nHigh confidence samples: {np.sum(high_confidence_mask)} ({np.sum(high_confidence_mask)/len(y_test)*100:.1f}%)")
print(f"Low confidence samples:  {np.sum(low_confidence_mask)} ({np.sum(low_confidence_mask)/len(y_test)*100:.1f}%)")

high_conf_acc = np.mean(tta_classes[high_confidence_mask] == y_test[high_confidence_mask])
low_conf_acc = np.mean(tta_classes[low_confidence_mask] == y_test[low_confidence_mask])

print(f"\nHigh confidence accuracy: {high_conf_acc*100:.2f}%")
print(f"Low confidence accuracy:  {low_conf_acc*100:.2f}%")

# ==============================================================================
# TECHNIQUE 4: Ensemble with Dropout at Test Time (MC Dropout)
# ==============================================================================
print("\n" + "="*80)
print("TECHNIQUE 4: MONTE CARLO DROPOUT")
print("="*80)

print("\nEnabling dropout at test time...")

# Create a function model that keeps dropout active
def predict_with_dropout(model, X, n_iter=10):
    """Predict with dropout enabled (Monte Carlo Dropout)"""
    predictions = []

    # Get the model with dropout enabled
    for layer in model.layers:
        if hasattr(layer, 'rate'):  # Dropout layer
            layer.trainable = False  # But don't update weights

    for i in range(n_iter):
        # Set training=True to enable dropout
        pred = model(X, training=True)
        predictions.append(pred.numpy())

    return np.mean(predictions, axis=0)

print("Generating MC Dropout predictions (10 iterations)...")
mc_dropout_pred = predict_with_dropout(model, X_test, n_iter=10)
mc_classes = np.argmax(mc_dropout_pred, axis=1)
mc_accuracy = np.mean(mc_classes == y_test)

print(f"✅ MC Dropout Accuracy: {mc_accuracy*100:.2f}%")
print(f"   Improvement: {(mc_accuracy - baseline_accuracy)*100:+.2f}%")

# ==============================================================================
# TECHNIQUE 5: ENSEMBLE ALL TECHNIQUES
# ==============================================================================
print("\n" + "="*80)
print("TECHNIQUE 5: ENSEMBLE ALL TECHNIQUES")
print("="*80)

print("\nCombining all prediction strategies...")

# Weighted ensemble (best techniques get more weight)
all_predictions = {
    'baseline': (baseline_pred, 1.0),
    'tta': (tta_avg, 2.0),  # TTA gets double weight
    'mc_dropout': (mc_dropout_pred, 1.5)
}

weighted_sum = np.zeros_like(baseline_pred)
total_weight = 0

for name, (pred, weight) in all_predictions.items():
    weighted_sum += pred * weight
    total_weight += weight
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    print(f"  {name:15s}: {acc*100:.2f}% (weight: {weight})")

final_pred = weighted_sum / total_weight
final_classes = np.argmax(final_pred, axis=1)
final_accuracy = np.mean(final_classes == y_test)

print(f"\n✅ Final Ensemble Accuracy: {final_accuracy*100:.2f}%")
print(f"   Total improvement: {(final_accuracy - baseline_accuracy)*100:+.2f}%")

# ==============================================================================
# DETAILED ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

# Per-class breakdown
non_violent_mask = y_test == 0
violent_mask = y_test == 1

print(f"\n📊 Baseline vs Final:")
print(f"{'Metric':<20s} {'Baseline':>10s} {'Final':>10s} {'Improvement':>12s}")
print("-" * 55)

baseline_nv = np.mean(baseline_classes[non_violent_mask] == y_test[non_violent_mask])
final_nv = np.mean(final_classes[non_violent_mask] == y_test[non_violent_mask])
print(f"{'Non-violent Acc':<20s} {baseline_nv*100:>9.2f}% {final_nv*100:>9.2f}% {(final_nv-baseline_nv)*100:>+11.2f}%")

baseline_v = np.mean(baseline_classes[violent_mask] == y_test[violent_mask])
final_v = np.mean(final_classes[violent_mask] == y_test[violent_mask])
print(f"{'Violent Acc':<20s} {baseline_v*100:>9.2f}% {final_v*100:>9.2f}% {(final_v-baseline_v)*100:>+11.2f}%")

print(f"{'Overall Acc':<20s} {baseline_accuracy*100:>9.2f}% {final_accuracy*100:>9.2f}% {(final_accuracy-baseline_accuracy)*100:>+11.2f}%")

# Confusion matrix comparison
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
print(f"  False Positives: {cm_baseline[0,1]} → {cm_final[0,1]} ({cm_final[0,1] - cm_baseline[0,1]:+d})")
print(f"  False Negatives: {cm_baseline[1,0]} → {cm_final[1,0]} ({cm_final[1,0] - cm_baseline[1,0]:+d})")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("🎯 FINAL SUMMARY")
print("="*80)

print(f"\n📈 Accuracy Journey:")
print(f"  Starting:  {baseline_accuracy*100:.2f}%")
print(f"  + TTA:     {tta_accuracy*100:.2f}% ({(tta_accuracy-baseline_accuracy)*100:+.2f}%)")
print(f"  + MC Drop: {mc_accuracy*100:.2f}% ({(mc_accuracy-baseline_accuracy)*100:+.2f}%)")
print(f"  + Ensemble:{final_accuracy*100:.2f}% ({(final_accuracy-baseline_accuracy)*100:+.2f}%)")

if final_accuracy * 100 >= 93.0:
    print(f"\n🎉 SUCCESS! Reached {final_accuracy*100:.2f}% (target: 93%+)")
elif final_accuracy * 100 >= 92.5:
    print(f"\n✅ Very close! {final_accuracy*100:.2f}% (only {93.0 - final_accuracy*100:.2f}% from target)")
elif final_accuracy * 100 >= 92.0:
    print(f"\n👍 Good progress! {final_accuracy*100:.2f}% (need +{93.0 - final_accuracy*100:.2f}% more)")
else:
    print(f"\n⚠️  At {final_accuracy*100:.2f}% (need +{93.0 - final_accuracy*100:.2f}% more)")
    print("   Consider training diverse ensemble models with different random seeds")

print("\n" + "="*80)
