#!/usr/bin/env python3
"""
Violence Detection Model Testing with Dual-Level TTA
- Level 1: VGG19 TTA (frame augmentation before feature extraction)
- Level 2: Model TTA (feature augmentation before prediction)
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
    'num_workers': 48,

    # VGG19-level TTA (frame augmentation)
    'vgg_tta_augmentations': 5,  # Extract features 5x per video with different augmentations

    # Model-level TTA (feature augmentation)
    'model_tta_augmentations': 3,  # 3x per feature set

    # Total predictions per video: 5 × 3 = 15 augmentations
}

# ============================================================================
# FRAME AUGMENTATION FUNCTIONS (VGG19-level TTA)
# ============================================================================

def augment_frame(frame, aug_type):
    """Apply visual augmentation to a single frame for VGG19-level TTA"""
    aug = frame.copy()

    if aug_type == 0:
        # Original (no augmentation)
        return aug

    elif aug_type == 1:
        # Horizontal flip
        aug = cv2.flip(aug, 1)

    elif aug_type == 2:
        # Brightness adjustment
        brightness = np.random.uniform(0.85, 1.15)
        aug = np.clip(aug * brightness, 0, 255).astype(np.uint8)

    elif aug_type == 3:
        # Contrast adjustment
        alpha = np.random.uniform(0.9, 1.1)  # Contrast
        beta = np.random.uniform(-10, 10)    # Brightness
        aug = np.clip(alpha * aug + beta, 0, 255).astype(np.uint8)

    elif aug_type == 4:
        # Slight rotation (-5 to +5 degrees)
        angle = np.random.uniform(-5, 5)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    elif aug_type == 5:
        # Random crop and resize
        h, w = aug.shape[:2]
        crop_size = int(min(h, w) * np.random.uniform(0.9, 1.0))
        y = np.random.randint(0, h - crop_size + 1)
        x = np.random.randint(0, w - crop_size + 1)
        aug = aug[y:y+crop_size, x:x+crop_size]
        aug = cv2.resize(aug, (224, 224))

    return aug

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

def extract_features_with_vgg_tta(video_path_and_aug):
    """
    Extract VGG19 features with frame-level augmentation
    Returns: features array (20, 4096)
    """
    global _worker_vgg

    video_path, aug_type = video_path_and_aug

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
                # Resize first
                frame = cv2.resize(frame, (224, 224))

                # Apply augmentation (VGG19-level TTA)
                frame = augment_frame(frame, aug_type)

                # Convert to RGB
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
# MODEL-LEVEL TTA FUNCTIONS
# ============================================================================

def augment_features(features, idx):
    """Apply augmentation to features (model-level TTA)"""
    aug = features.copy()

    # Brightness variation
    brightness = 1.0 + np.random.uniform(-0.1, 0.1)
    aug = np.clip(aug * brightness, features.min(), features.max())

    # Noise
    noise = np.random.normal(0, 0.005, aug.shape)
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

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🧪 VIOLENCE DETECTION - DUAL-LEVEL TTA TESTING")
    print("=" * 80)
    print(f"\nModel: {CONFIG['model_path']}")
    print(f"Test data: {CONFIG['test_data_path']}")
    print(f"Workers: {CONFIG['num_workers']} CPU cores")
    print(f"\nTTA Configuration:")
    print(f"  VGG19-level augmentations: {CONFIG['vgg_tta_augmentations']}")
    print(f"  Model-level augmentations: {CONFIG['model_tta_augmentations']}")
    print(f"  Total predictions/video: {CONFIG['vgg_tta_augmentations'] * CONFIG['model_tta_augmentations']}")
    print()

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
    # EXTRACT FEATURES WITH VGG19-LEVEL TTA (PARALLEL CPU)
    # ========================================================================

    print("=" * 80)
    print(f"🎬 EXTRACTING FEATURES WITH VGG19-LEVEL TTA")
    print("=" * 80)
    print(f"Each video processed {CONFIG['vgg_tta_augmentations']}x with different augmentations\n")

    # Create tasks: each video × number of VGG TTA augmentations
    extraction_tasks = []
    for video, label in all_videos:
        for aug_type in range(CONFIG['vgg_tta_augmentations']):
            extraction_tasks.append(((video, aug_type), label))

    print(f"Total extraction tasks: {len(extraction_tasks)} ({len(all_videos)} videos × {CONFIG['vgg_tta_augmentations']} augmentations)")

    video_features = {}  # {video_path: [features_aug0, features_aug1, ...]}
    video_labels = {}
    failed = 0

    with ProcessPoolExecutor(max_workers=CONFIG['num_workers'], initializer=init_worker) as executor:
        futures = {executor.submit(extract_features_with_vgg_tta, task): (task, label)
                   for task, label in extraction_tasks}

        for future in tqdm(as_completed(futures), total=len(extraction_tasks), desc="Extracting"):
            (video_path, aug_type), label = futures[future]
            try:
                feat = future.result()
                if feat is not None:
                    video_path_str = str(video_path)
                    if video_path_str not in video_features:
                        video_features[video_path_str] = []
                        video_labels[video_path_str] = label
                    video_features[video_path_str].append(feat)
                else:
                    failed += 1
            except:
                failed += 1

    # Filter videos that have all augmentations extracted
    valid_videos = {k: v for k, v in video_features.items()
                    if len(v) == CONFIG['vgg_tta_augmentations']}

    print(f"\n✓ Videos with complete feature sets: {len(valid_videos)}")
    print(f"⚠️  Failed extractions: {failed}")
    print()

    if len(valid_videos) == 0:
        print("❌ No valid videos found!")
        exit(1)

    # ========================================================================
    # BASELINE TEST (NO TTA - use first feature set only)
    # ========================================================================

    print("=" * 80)
    print("🧪 BASELINE TEST (No TTA)")
    print("=" * 80)

    baseline_preds = []
    labels = []

    for video_path in tqdm(valid_videos.keys(), desc="Baseline"):
        # Use first feature set (original, no augmentation)
        feat = valid_videos[video_path][0]
        pred = model.predict(np.expand_dims(feat, axis=0), verbose=0)
        baseline_preds.append(pred[0])
        labels.append(video_labels[video_path])

    baseline_preds = np.array(baseline_preds)
    baseline_classes = np.argmax(baseline_preds, axis=1)
    labels = np.array(labels)
    baseline_acc = np.mean(baseline_classes == labels)

    v_mask = labels == 1
    nv_mask = labels == 0

    v_acc = np.mean(baseline_classes[v_mask] == labels[v_mask])
    nv_acc = np.mean(baseline_classes[nv_mask] == labels[nv_mask])

    print(f"\n✓ Baseline Accuracy: {baseline_acc*100:.2f}%")
    print(f"  Violent: {v_acc*100:.2f}%")
    print(f"  Non-Violent: {nv_acc*100:.2f}%\n")

    # ========================================================================
    # DUAL-LEVEL TTA TEST
    # ========================================================================

    print("=" * 80)
    print("🎯 DUAL-LEVEL TTA TEST")
    print("=" * 80)
    print(f"VGG19 augmentations: {CONFIG['vgg_tta_augmentations']}")
    print(f"Model augmentations: {CONFIG['model_tta_augmentations']}")
    print(f"Total predictions per video: {CONFIG['vgg_tta_augmentations'] * CONFIG['model_tta_augmentations']}\n")

    tta_preds = []

    for video_path in tqdm(valid_videos.keys(), desc="Dual-TTA"):
        all_preds = []

        # For each VGG19-augmented feature set
        for vgg_feat in valid_videos[video_path]:

            # Apply model-level TTA
            for model_aug_idx in range(CONFIG['model_tta_augmentations']):
                if model_aug_idx == 0:
                    # Original features
                    feat = vgg_feat
                else:
                    # Augmented features
                    feat = augment_features(vgg_feat, model_aug_idx)

                pred = model.predict(np.expand_dims(feat, axis=0), verbose=0)
                all_preds.append(pred[0])

        # Average all predictions for this video
        avg_pred = np.mean(all_preds, axis=0)
        tta_preds.append(avg_pred)

    tta_preds = np.array(tta_preds)
    tta_classes = np.argmax(tta_preds, axis=1)
    tta_acc = np.mean(tta_classes == labels)

    tta_v_acc = np.mean(tta_classes[v_mask] == labels[v_mask])
    tta_nv_acc = np.mean(tta_classes[nv_mask] == labels[nv_mask])

    print(f"\n✓ Dual-TTA Accuracy: {tta_acc*100:.2f}%")
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
        'tta_config': {
            'vgg_augmentations': CONFIG['vgg_tta_augmentations'],
            'model_augmentations': CONFIG['model_tta_augmentations'],
            'total_predictions_per_video': CONFIG['vgg_tta_augmentations'] * CONFIG['model_tta_augmentations']
        },
        'test_videos': {
            'total': len(valid_videos),
            'violent': int(v_mask.sum()),
            'nonviolent': int(nv_mask.sum()),
            'failed': failed
        },
        'baseline': {
            'accuracy': float(baseline_acc),
            'violent_accuracy': float(v_acc),
            'nonviolent_accuracy': float(nv_acc),
        },
        'dual_tta': {
            'accuracy': float(tta_acc),
            'violent_accuracy': float(tta_v_acc),
            'nonviolent_accuracy': float(tta_nv_acc),
        },
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
    }

    results_file = Path(CONFIG['results_dir']) / f"dual_tta_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}\n")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("=" * 80)
    print("🎉 DUAL-LEVEL TTA TESTING COMPLETE")
    print("=" * 80)

    improvement = (tta_acc - baseline_acc) * 100

    print(f"\nBaseline:  {baseline_acc*100:.2f}%")
    print(f"Dual-TTA:  {tta_acc*100:.2f}%")
    print(f"Improvement: +{improvement:.2f}%\n")

    if tta_acc >= 0.91:
        print("✅ OUTSTANDING! Accuracy ≥ 91%")
    elif tta_acc >= 0.90:
        print("✅ EXCELLENT! Accuracy ≥ 90%")
    elif tta_acc >= 0.88:
        print("✅ GOOD! Accuracy ≥ 88%")
    else:
        print("⚠️  Below target")

    print("\n" + "=" * 80 + "\n")
