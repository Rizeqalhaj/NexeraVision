#!/usr/bin/env python3
"""
Violence Detection Model Testing with Test-Time Augmentation (TTA)
Clean implementation with parallel CPU feature extraction
"""

import os
# Set environment variables BEFORE any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU for main process
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

# Limit TensorFlow threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

cv2.setLogLevel(0)

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'model_path': '/workspace/violence_detection_mvp/models/best_model.h5',
    'test_data_path': '/workspace/Training/test',
    'results_dir': '/workspace/violence_detection_mvp/test_results',
    'num_frames': 20,
    'num_workers': 48,  # Conservative for stability
    'tta_augmentations': 10,
}

# ============================================================================
# WORKER FUNCTIONS (CPU-ONLY)
# ============================================================================

_worker_vgg = None

def init_worker():
    """Initialize worker with CPU-only VGG19"""
    global _worker_vgg

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    base = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    _worker_vgg = tf.keras.Model(inputs=base.input, outputs=base.get_layer('fc2').output)

def extract_features(video_path):
    """Extract VGG19 features from video"""
    global _worker_vgg

    try:
        import sys
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        cap = cv2.VideoCapture(str(video_path))
        sys.stderr.close()
        sys.stderr = stderr

        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 20:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, 20, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                frames.append(frame)

        cap.release()

        if len(frames) < 16:
            return None

        while len(frames) < 20:
            frames.append(frames[-1])

        frames_array = np.array(frames[:20])
        frames_preprocessed = tf.keras.applications.vgg19.preprocess_input(frames_array)
        features = _worker_vgg.predict(frames_preprocessed, verbose=0, batch_size=20)
        features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

        return features

    except:
        return None

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_model():
    """Build exact model architecture from training"""
    from tensorflow.keras import layers, models, regularizers

    inputs = layers.Input(shape=(20, 4096), name='input_features')

    # Feature compression
    x = layers.Dense(512, activation='relu', name='feature_compression')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.16)(x)

    # BiLSTM layers
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

    # Attention
    attention_score = layers.Dense(1, activation='tanh', name='attention_score')(x)
    attention_score = layers.Flatten()(attention_score)
    attention_weights = layers.Activation('softmax', name='attention_weights')(attention_score)
    attention_weights_expanded = layers.RepeatVector(96)(attention_weights)
    attention_weights_expanded = layers.Permute([2, 1])(attention_weights_expanded)
    attended = layers.Multiply(name='attended_features')([x, attention_weights_expanded])
    attended = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), output_shape=(96,),
                            name='attention_pooling')(attended)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.003), name='dense_1')(attended)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.32)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.003), name='dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.256)(x)

    outputs = layers.Dense(2, activation='softmax', dtype='float32', name='output')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='HybridOptimalViolenceDetector')

# ============================================================================
# TTA FUNCTIONS
# ============================================================================

def augment_features(features, idx):
    """Apply augmentation to features"""
    aug = features.copy()

    # Brightness variation
    brightness = 1.0 + np.random.uniform(-0.15, 0.15)
    aug = np.clip(aug * brightness, features.min(), features.max())

    # Noise
    noise = np.random.normal(0, 0.01, aug.shape)
    aug = aug + noise

    # Temporal jitter
    if idx % 2 == 0:
        for i in range(0, 20, 4):
            end = min(i + 4, 20)
            if end - i > 1:
                indices = np.arange(i, end)
                np.random.shuffle(indices)
                aug[i:end] = aug[indices]

    return aug

def predict_with_tta(features, model, n_aug):
    """Predict with Test-Time Augmentation"""
    preds = []

    # Original
    pred = model.predict(np.expand_dims(features, axis=0), verbose=0)
    preds.append(pred[0])

    # Augmented
    for i in range(n_aug - 1):
        aug = augment_features(features, i)
        pred = model.predict(np.expand_dims(aug, axis=0), verbose=0)
        preds.append(pred[0])

    return np.mean(preds, axis=0)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🧪 VIOLENCE DETECTION - TTA TESTING")
    print("=" * 80)
    print(f"\nModel: {CONFIG['model_path']}")
    print(f"Test data: {CONFIG['test_data_path']}")
    print(f"Workers: {CONFIG['num_workers']} CPU cores")
    print(f"TTA augmentations: {CONFIG['tta_augmentations']}\n")

    # ========================================================================
    # LOAD MODEL
    # ========================================================================

    print("=" * 80)
    print("📥 LOADING MODEL")
    print("=" * 80)

    model = build_model()
    model.load_weights(CONFIG['model_path'])
    print(f"✓ Model loaded ({model.count_params():,} parameters)\n")

    # ========================================================================
    # COLLECT VIDEOS
    # ========================================================================

    print("=" * 80)
    print("📁 COLLECTING TEST VIDEOS")
    print("=" * 80)

    test_path = Path(CONFIG['test_data_path'])
    violent = list((test_path / 'Violent').glob('*.mp4')) + list((test_path / 'Violent').glob('*.avi'))
    nonviolent = list((test_path / 'NonViolent').glob('*.mp4')) + list((test_path / 'NonViolent').glob('*.avi'))

    print(f"Violent: {len(violent)}")
    print(f"Non-Violent: {len(nonviolent)}")
    print(f"Total: {len(violent) + len(nonviolent)}\n")

    all_videos = [(v, 1) for v in violent] + [(v, 0) for v in nonviolent]

    # ========================================================================
    # EXTRACT FEATURES (PARALLEL CPU)
    # ========================================================================

    print("=" * 80)
    print(f"🎬 EXTRACTING FEATURES ({CONFIG['num_workers']} workers)")
    print("=" * 80)

    features_list = []
    labels_list = []
    failed = 0

    with ProcessPoolExecutor(max_workers=CONFIG['num_workers'], initializer=init_worker) as executor:
        futures = {executor.submit(extract_features, v): (v, l) for v, l in all_videos}

        for future in tqdm(as_completed(futures), total=len(all_videos), desc="Extracting"):
            video, label = futures[future]
            try:
                feat = future.result()
                if feat is not None:
                    features_list.append(feat)
                    labels_list.append(label)
                else:
                    failed += 1
            except:
                failed += 1

    print(f"\n✓ Extracted: {len(features_list)}")
    if failed > 0:
        print(f"⚠️  Failed: {failed}")
    print()

    if len(features_list) == 0:
        print("❌ No valid videos found!")
        exit(1)

    labels = np.array(labels_list)

    # ========================================================================
    # BASELINE TEST (NO TTA)
    # ========================================================================

    print("=" * 80)
    print("🧪 BASELINE TEST (No TTA)")
    print("=" * 80)

    baseline_preds = []
    for feat in tqdm(features_list, desc="Baseline"):
        pred = model.predict(np.expand_dims(feat, axis=0), verbose=0)
        baseline_preds.append(pred[0])

    baseline_preds = np.array(baseline_preds)
    baseline_classes = np.argmax(baseline_preds, axis=1)
    baseline_acc = np.mean(baseline_classes == labels)

    v_mask = labels == 1
    nv_mask = labels == 0

    v_acc = np.mean(baseline_classes[v_mask] == labels[v_mask])
    nv_acc = np.mean(baseline_classes[nv_mask] == labels[nv_mask])

    print(f"\n✓ Baseline Accuracy: {baseline_acc*100:.2f}%")
    print(f"  Violent: {v_acc*100:.2f}%")
    print(f"  Non-Violent: {nv_acc*100:.2f}%\n")

    # ========================================================================
    # TTA TEST
    # ========================================================================

    print("=" * 80)
    print("🎯 TTA TEST")
    print("=" * 80)

    tta_preds = []
    for feat in tqdm(features_list, desc="TTA"):
        pred = predict_with_tta(feat, model, CONFIG['tta_augmentations'])
        tta_preds.append(pred)

    tta_preds = np.array(tta_preds)
    tta_classes = np.argmax(tta_preds, axis=1)
    tta_acc = np.mean(tta_classes == labels)

    tta_v_acc = np.mean(tta_classes[v_mask] == labels[v_mask])
    tta_nv_acc = np.mean(tta_classes[nv_mask] == labels[nv_mask])

    print(f"\n✓ TTA Accuracy: {tta_acc*100:.2f}%")
    print(f"  Violent: {tta_v_acc*100:.2f}%")
    print(f"  Non-Violent: {tta_nv_acc*100:.2f}%\n")

    # ========================================================================
    # METRICS
    # ========================================================================

    print("=" * 80)
    print("📊 DETAILED METRICS")
    print("=" * 80)

    cm = confusion_matrix(labels, tta_classes)

    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            Non-V  Violent")
    print(f"Actual Non-V {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Violent {cm[1,0]:5d}  {cm[1,1]:5d}\n")

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%\n")

    print(classification_report(labels, tta_classes, target_names=['Non-Violent', 'Violent']))

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    results = {
        'timestamp': datetime.now().isoformat(),
        'test_videos': {
            'total': len(features_list),
            'violent': int(v_mask.sum()),
            'nonviolent': int(nv_mask.sum()),
            'failed': failed
        },
        'baseline': {
            'accuracy': float(baseline_acc),
            'violent_accuracy': float(v_acc),
            'nonviolent_accuracy': float(nv_acc),
        },
        'tta': {
            'accuracy': float(tta_acc),
            'violent_accuracy': float(tta_v_acc),
            'nonviolent_accuracy': float(tta_nv_acc),
            'augmentations': CONFIG['tta_augmentations']
        },
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
    }

    results_file = Path(CONFIG['results_dir']) / f"tta_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}\n")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("=" * 80)
    print("🎉 TESTING COMPLETE")
    print("=" * 80)

    improvement = (tta_acc - baseline_acc) * 100

    print(f"\nBaseline: {baseline_acc*100:.2f}%")
    print(f"TTA:      {tta_acc*100:.2f}%")
    print(f"Improvement: +{improvement:.2f}%\n")

    if tta_acc >= 0.90:
        print("✅ EXCELLENT! Accuracy ≥ 90%")
    elif tta_acc >= 0.88:
        print("✅ GOOD! Accuracy ≥ 88%")
    else:
        print("⚠️  Below target")

    print("\n" + "=" * 80 + "\n")
