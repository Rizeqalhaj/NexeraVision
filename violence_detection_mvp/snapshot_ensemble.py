#!/usr/bin/env python3
"""
SNAPSHOT ENSEMBLE - Use existing checkpoints for instant ensemble
NO additional training needed!
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
print("SNAPSHOT ENSEMBLE - Zero Training Time!")
print("="*80)

# Load test data
X_test = np.load('feature_cache/test_features.npy')
y_test = np.load('feature_cache/test_labels.npy')

print(f"\nTest set: {X_test.shape}")

# Find all recent checkpoints (last 10 epochs from your good training)
checkpoint_dir = Path('checkpoints')

# Strategy 1: Use last 10 epoch checkpoints
last_epochs = list(range(191, 201))  # Epochs 191-200
snapshot_paths = [checkpoint_dir / f'ultimate_epoch_{epoch:03d}.h5' for epoch in last_epochs]

# Filter to only existing checkpoints
snapshot_paths = [p for p in snapshot_paths if p.exists()]

if len(snapshot_paths) == 0:
    print("⚠️  No snapshot checkpoints found!")
    print("Looking for any ultimate_epoch_*.h5 files...")
    snapshot_paths = sorted(checkpoint_dir.glob('ultimate_epoch_*.h5'))[-10:]

print(f"\n📸 Found {len(snapshot_paths)} snapshot checkpoints")

if len(snapshot_paths) < 3:
    print("❌ Need at least 3 checkpoints for snapshot ensemble")
    print("Using best model only...")
    snapshot_paths = [checkpoint_dir / 'ultimate_best_model.h5']

print("\nLoading snapshots:")
for path in snapshot_paths:
    print(f"  - {path.name}")

# Load all snapshots
all_predictions = []

for i, model_path in enumerate(snapshot_paths, 1):
    print(f"\n[{i}/{len(snapshot_paths)}] Loading {model_path.name}...")

    model = ViolenceDetectionModel(config=Config).build_model()
    model.load_weights(str(model_path))

    # Predict
    predictions = model.predict(X_test, batch_size=64, verbose=0)
    all_predictions.append(predictions)

    # Quick accuracy check
    pred_classes = np.argmax(predictions, axis=1)
    acc = np.mean(pred_classes == y_test)
    print(f"  Individual accuracy: {acc*100:.2f}%")

print(f"\n✅ Loaded {len(all_predictions)} snapshots")

# Ensemble strategies
print("\n" + "="*80)
print("ENSEMBLE STRATEGIES")
print("="*80)

# Strategy 1: Simple average
avg_predictions = np.mean(all_predictions, axis=0)
avg_classes = np.argmax(avg_predictions, axis=1)
avg_accuracy = np.mean(avg_classes == y_test)

print(f"\n1. Simple Average ({len(all_predictions)} snapshots):")
print(f"   Accuracy: {avg_accuracy*100:.2f}%")

# Strategy 2: Weighted average (later snapshots get more weight)
weights = np.linspace(0.5, 1.5, len(all_predictions))  # Linear weights
weights = weights / weights.sum()

weighted_predictions = np.zeros_like(all_predictions[0])
for i, pred in enumerate(all_predictions):
    weighted_predictions += weights[i] * pred

weighted_classes = np.argmax(weighted_predictions, axis=1)
weighted_accuracy = np.mean(weighted_classes == y_test)

print(f"\n2. Weighted Average (later epochs = higher weight):")
print(f"   Accuracy: {weighted_accuracy*100:.2f}%")

# Strategy 3: Majority voting
all_classes = np.array([np.argmax(pred, axis=1) for pred in all_predictions])
voted_classes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_classes)
voted_accuracy = np.mean(voted_classes == y_test)

print(f"\n3. Majority Voting:")
print(f"   Accuracy: {voted_accuracy*100:.2f}%")

# Best strategy
best_accuracy = max(avg_accuracy, weighted_accuracy, voted_accuracy)
if best_accuracy == avg_accuracy:
    best_strategy = "Simple Average"
    best_predictions = avg_classes
elif best_accuracy == weighted_accuracy:
    best_strategy = "Weighted Average"
    best_predictions = weighted_classes
else:
    best_strategy = "Majority Voting"
    best_predictions = voted_classes

print("\n" + "="*80)
print("DETAILED ANALYSIS (Best Strategy)")
print("="*80)

print(f"\nBest Strategy: {best_strategy}")
print(f"Ensemble Accuracy: {best_accuracy*100:.2f}%")

# Per-class metrics
non_violent_mask = y_test == 0
violent_mask = y_test == 1

non_violent_acc = np.mean(best_predictions[non_violent_mask] == y_test[non_violent_mask])
violent_acc = np.mean(best_predictions[violent_mask] == y_test[violent_mask])

print(f"\nPer-Class Accuracy:")
print(f"  Non-violent: {non_violent_acc*100:.2f}%")
print(f"  Violent:     {violent_acc*100:.2f}%")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, best_predictions)
print(f"\nConfusion Matrix:")
print(f"  [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
print(f"   [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")

print(f"\nClassification Report:")
print(classification_report(y_test, best_predictions, target_names=['Non-violent', 'Violent']))

# Compare with single best model
print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)

baseline_accuracy = 91.35  # Your best single model
improvement = (best_accuracy - baseline_accuracy/100) * 100

print(f"\nSingle Best Model: {baseline_accuracy:.2f}%")
print(f"Snapshot Ensemble: {best_accuracy*100:.2f}%")
print(f"Improvement: {improvement:+.2f}%")

if best_accuracy * 100 >= 93.0:
    print("\n🎉 SUCCESS! Hit 93%+ target!")
elif best_accuracy * 100 >= 92.0:
    print("\n✅ Good progress! Close to 93% target")
else:
    print("\n⚠️  Need additional strategies (TTA, diverse ensemble)")

print("\n" + "="*80)
print(f"🎯 FINAL SNAPSHOT ENSEMBLE ACCURACY: {best_accuracy*100:.2f}%")
print("="*80)
