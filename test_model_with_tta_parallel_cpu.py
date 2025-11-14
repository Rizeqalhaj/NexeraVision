#!/usr/bin/env python3
"""
Test Violence Detection Model with TTA (Test-Time Augmentation)
Parallel CPU version for 192-core processing
"""

import os

# CRITICAL: Set ALL environment variables BEFORE any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Main process uses GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

# Limit BLAS threads globally to prevent thread explosion
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress OpenCV warnings
cv2.setLogLevel(0)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': '/workspace/violence_detection_mvp/models/best_model.h5',
    'test_data_path': '/workspace/Training/test',
    'results_dir': '/workspace/violence_detection_mvp/test_results',

    'num_frames': 20,
    'frame_size': (224, 224),
    'num_workers': 192,  # Full CPU power

    # TTA Configuration
    'tta_augmentations': 10,
    'tta_brightness_range': 0.15,
    'tta_noise_std': 0.01,
}

# ============================================================================
# PARALLEL CPU FEATURE EXTRACTION (WORKER FUNCTION)
# ============================================================================

# Global variable for VGG19 feature extractor in worker processes
_worker_feature_extractor = None

def worker_init():
    """Initialize worker process with CPU-only mode"""
    global _worker_feature_extractor

    # CRITICAL: Set CPU-only and limit threads BEFORE importing tensorflow
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    import tensorflow as tf

    # Load VGG19 once per worker
    base_model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    _worker_feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )

def extract_video_features_worker(video_path):
    """
    Worker function for parallel CPU-only VGG19 feature extraction
    Uses pre-loaded VGG19 from worker_init()
    """
    global _worker_feature_extractor

    import cv2
    import numpy as np
    import tensorflow as tf

    try:
        import sys
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        cap = cv2.VideoCapture(str(video_path))

        sys.stderr.close()
        sys.stderr = original_stderr

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 20:  # CONFIG['num_frames']
            cap.release()
            return None

        # Extract frames
        indices = np.linspace(0, total_frames - 1, 20, dtype=int)
        frames = []

        for idx in indices:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32)
                    frames.append(frame)
            except:
                continue

        cap.release()

        if len(frames) < 16:  # 80% of 20 frames
            return None

        # Pad if needed
        while len(frames) < 20:
            frames.append(frames[-1])

        frames_array = np.array(frames[:20])

        # Use pre-loaded VGG19 from worker_init()
        frames_preprocessed = tf.keras.applications.vgg19.preprocess_input(frames_array)
        features = _worker_feature_extractor.predict(frames_preprocessed, verbose=0, batch_size=20)
        features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

        return features

    except Exception as e:
        return None

# ============================================================================
# MODEL BUILDING FUNCTION
# ============================================================================

from tensorflow.keras import layers, models, regularizers

def build_model():
    """Rebuild the exact model architecture from training"""
    input_shape = (20, 4096)
    inputs = layers.Input(shape=input_shape, name='input_features')

    # Feature compression
    x = layers.Dense(512, activation='relu', name='feature_compression')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.32 * 0.5)(x)

    # BiLSTM with residual
    x = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True, dropout=0.32, recurrent_dropout=0.18,
                   kernel_regularizer=regularizers.l2(0.003)), name='bilstm_1')(x)
    x = layers.BatchNormalization()(x)
    x_residual = x

    x = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True, dropout=0.32, recurrent_dropout=0.18,
                   kernel_regularizer=regularizers.l2(0.003)), name='bilstm_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add(name='residual_add')([x, x_residual])

    x = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True, dropout=0.32, recurrent_dropout=0.18,
                   kernel_regularizer=regularizers.l2(0.003)), name='bilstm_3')(x)
    x = layers.BatchNormalization()(x)

    # Attention mechanism
    attention_score = layers.Dense(1, activation='tanh', name='attention_score')(x)
    attention_score = layers.Flatten()(attention_score)
    attention_weights = layers.Activation('softmax', name='attention_weights')(attention_score)
    attention_weights_expanded = layers.RepeatVector(96)(attention_weights)
    attention_weights_expanded = layers.Permute([2, 1])(attention_weights_expanded)
    attended = layers.Multiply(name='attended_features')([x, attention_weights_expanded])
    attended = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1),
                            output_shape=(96,),
                            name='attention_pooling')(attended)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.003), name='dense_1')(attended)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.32)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.003), name='dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.32 * 0.8)(x)

    outputs = layers.Dense(2, activation='softmax', dtype='float32', name='output')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='HybridOptimalViolenceDetector')

# ============================================================================
# TTA FUNCTIONS
# ============================================================================

def augment_features(features, aug_idx):
    """Apply augmentation to features for TTA"""
    augmented = features.copy()

    # Brightness variation
    brightness = 1.0 + np.random.uniform(-CONFIG['tta_brightness_range'], CONFIG['tta_brightness_range'])
    augmented = augmented * brightness
    augmented = np.clip(augmented, features.min(), features.max())

    # Add noise
    noise = np.random.normal(0, CONFIG['tta_noise_std'], augmented.shape)
    augmented = augmented + noise

    # Temporal jitter
    if aug_idx % 2 == 0:
        for i in range(0, CONFIG['num_frames'], 4):
            end = min(i + 4, CONFIG['num_frames'])
            if end - i > 1:
                indices = np.arange(i, end)
                np.random.shuffle(indices)
                augmented[i:end] = augmented[indices]

    return augmented

def predict_with_tta(features, model, num_augmentations):
    """Predict with Test-Time Augmentation"""
    predictions = []

    # Original prediction
    pred = model.predict(np.expand_dims(features, axis=0), verbose=0)
    predictions.append(pred[0])

    # Augmented predictions
    for aug_idx in range(num_augmentations - 1):
        augmented = augment_features(features, aug_idx)
        pred = model.predict(np.expand_dims(augmented, axis=0), verbose=0)
        predictions.append(pred[0])

    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Force spawn method to avoid CUDA context inheritance
    multiprocessing.set_start_method('spawn', force=True)

    # Create results directory
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🧪 VIOLENCE DETECTION MODEL TESTING WITH TTA")
    print("=" * 80)
    print()

    print("Configuration:")
    print(f"  Model: {CONFIG['model_path']}")
    print(f"  Test data: {CONFIG['test_data_path']}")
    print(f"  TTA augmentations: {CONFIG['tta_augmentations']}")
    print(f"  Parallel workers: {CONFIG['num_workers']} CPU cores")
    print()

    # ============================================================================
    # LOAD MODEL (MAIN PROCESS - GPU)
    # ============================================================================

    print("=" * 80)
    print("📥 LOADING MODEL (GPU)")
    print("=" * 80)
    print()

    if not Path(CONFIG['model_path']).exists():
        print(f"❌ Model not found: {CONFIG['model_path']}")
        exit(1)

    print("Loading model...")

    model = build_model()
    print("✓ Model architecture built")

    model.load_weights(CONFIG['model_path'])
    print("✓ Weights loaded successfully")
    print(f"  Parameters: {model.count_params():,}")
    print()

    # ============================================================================
    # COLLECT TEST DATA
    # ============================================================================

    print("=" * 80)
    print("📁 COLLECTING TEST DATA")
    print("=" * 80)
    print()

    test_path = Path(CONFIG['test_data_path'])
    
    violent_videos = list((test_path / 'Violent').glob('*.mp4')) + list((test_path / 'Violent').glob('*.avi'))
    nonviolent_videos = list((test_path / 'NonViolent').glob('*.mp4')) + list((test_path / 'NonViolent').glob('*.avi'))
    
    print(f"Found:")
    print(f"  Violent: {len(violent_videos)} videos")
    print(f"  Non-Violent: {len(nonviolent_videos)} videos")
    print(f"  Total: {len(violent_videos) + len(nonviolent_videos)} videos")
    print()
    
    all_videos = [(v, 1) for v in violent_videos] + [(v, 0) for v in nonviolent_videos]
    
    # ============================================================================
    # EXTRACT FEATURES (192-CORE PARALLEL CPU)
    # ============================================================================
    
    print("=" * 80)
    print(f"🎬 EXTRACTING FEATURES ({CONFIG['num_workers']}-CORE PARALLEL CPU)")
    print("=" * 80)
    print()
    
    test_features = []
    test_labels = []
    failed = 0
    
    # Parallel CPU processing with initializer
    with ProcessPoolExecutor(max_workers=CONFIG['num_workers'], initializer=worker_init) as executor:
        futures = {executor.submit(extract_video_features_worker, video): (video, label) for video, label in all_videos}
    
        for future in tqdm(as_completed(futures), total=len(all_videos), desc="Extracting"):
            video, label = futures[future]
            try:
                features = future.result()
                if features is not None:
                    test_features.append(features)
                    test_labels.append(label)
                else:
                    failed += 1
            except Exception:
                failed += 1
    
    print(f"\n✓ Extracted: {len(test_features)} videos")
    if failed > 0:
        print(f"⚠️  Failed: {failed} videos")
    print()
    
    if len(test_features) == 0:
        print("❌ No valid test videos found!")
        exit(1)

    # ============================================================================
    # TESTING WITHOUT TTA (BASELINE)
    # ============================================================================
    
    print("=" * 80)
    print("🧪 TESTING WITHOUT TTA (BASELINE)")
    print("=" * 80)
    print()
    
    print("Running predictions...")
    baseline_predictions = []
    
    for features in tqdm(test_features, desc="Baseline"):
        pred = model.predict(np.expand_dims(features, axis=0), verbose=0)
        baseline_predictions.append(pred[0])
    
    baseline_predictions = np.array(baseline_predictions)
    baseline_classes = np.argmax(baseline_predictions, axis=1)
    true_labels = np.array(test_labels)
    
    baseline_accuracy = np.mean(baseline_classes == true_labels)
    
    print(f"\n✓ Baseline Accuracy: {baseline_accuracy*100:.2f}%")
    print()
    
    # Per-class accuracy
    violent_mask = true_labels == 1
    nonviolent_mask = true_labels == 0
    
    baseline_violent_acc = np.mean(baseline_classes[violent_mask] == true_labels[violent_mask])
    baseline_nonviolent_acc = np.mean(baseline_classes[nonviolent_mask] == true_labels[nonviolent_mask])
    
    print(f"Per-class:")
    print(f"  Violent:     {baseline_violent_acc*100:.2f}%")
    print(f"  Non-Violent: {baseline_nonviolent_acc*100:.2f}%")
    print(f"  Gap:         {abs(baseline_violent_acc - baseline_nonviolent_acc)*100:.2f}%")
    print()
    
    # ============================================================================
    # TESTING WITH TTA
    # ============================================================================
    
    print("=" * 80)
    print("🎯 TESTING WITH TTA (ROBUST PREDICTIONS)")
    print("=" * 80)
    print(f"Augmentations per video: {CONFIG['tta_augmentations']}")
    print()
    
    print("Running TTA predictions...")
    tta_predictions = []
    
    for features in tqdm(test_features, desc="TTA"):
        pred = predict_with_tta(features, model, CONFIG['tta_augmentations'])
        tta_predictions.append(pred)
    
    tta_predictions = np.array(tta_predictions)
    tta_classes = np.argmax(tta_predictions, axis=1)
    
    tta_accuracy = np.mean(tta_classes == true_labels)
    
    print(f"\n✓ TTA Accuracy: {tta_accuracy*100:.2f}%")
    print()
    
    # Per-class accuracy
    tta_violent_acc = np.mean(tta_classes[violent_mask] == true_labels[violent_mask])
    tta_nonviolent_acc = np.mean(tta_classes[nonviolent_mask] == true_labels[nonviolent_mask])
    
    print(f"Per-class:")
    print(f"  Violent:     {tta_violent_acc*100:.2f}%")
    print(f"  Non-Violent: {tta_nonviolent_acc*100:.2f}%")
    print(f"  Gap:         {abs(tta_violent_acc - tta_nonviolent_acc)*100:.2f}%")
    print()
    
    # ============================================================================
    # CONFUSION MATRIX & METRICS
    # ============================================================================
    
    print("=" * 80)
    print("📊 DETAILED METRICS (TTA)")
    print("=" * 80)
    print()
    
    cm = confusion_matrix(true_labels, tta_classes)
    
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Non-V  Violent")
    print(f"Actual Non-V  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Violent {cm[1,0]:5d}  {cm[1,1]:5d}")
    print()
    
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Metrics:")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print()
    
    print("Classification Report:")
    print(classification_report(true_labels, tta_classes, target_names=['Non-Violent', 'Violent']))
    print()
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    
    print("=" * 80)
    print("💾 SAVING RESULTS")
    print("=" * 80)
    print()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': CONFIG['model_path'],
        'test_videos': {
            'total': len(test_features),
            'violent': int(violent_mask.sum()),
            'nonviolent': int(nonviolent_mask.sum()),
            'failed': failed
        },
        'baseline': {
            'accuracy': float(baseline_accuracy),
            'violent_accuracy': float(baseline_violent_acc),
            'nonviolent_accuracy': float(baseline_nonviolent_acc),
            'gap': float(abs(baseline_violent_acc - baseline_nonviolent_acc))
        },
        'tta': {
            'accuracy': float(tta_accuracy),
            'violent_accuracy': float(tta_violent_acc),
            'nonviolent_accuracy': float(tta_nonviolent_acc),
            'gap': float(abs(tta_violent_acc - tta_nonviolent_acc)),
            'augmentations': CONFIG['tta_augmentations']
        },
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
    }
    
    results_file = Path(CONFIG['results_dir']) / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    print()
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("=" * 80)
    print("🎉 TESTING COMPLETE")
    print("=" * 80)
    print()
    
    improvement = (tta_accuracy - baseline_accuracy) * 100
    
    print("FINAL RESULTS:")
    print(f"  Baseline Accuracy: {baseline_accuracy*100:.2f}%")
    print(f"  TTA Accuracy:      {tta_accuracy*100:.2f}%")
    print(f"  Improvement:       +{improvement:.2f}%")
    print()
    
    if tta_accuracy >= 0.90:
        print("✅ EXCELLENT! Accuracy ≥ 90%")
    elif tta_accuracy >= 0.88:
        print("✅ GOOD! Accuracy ≥ 88%")
    elif tta_accuracy >= 0.85:
        print("✅ ACCEPTABLE! Accuracy ≥ 85%")
    else:
        print("⚠️  Below target accuracy")
    
    print()
    print("Model ready for deployment!" if tta_accuracy >= 0.88 else "Consider additional training or data collection")
    print()
    print("=" * 80)
