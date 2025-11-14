#!/usr/bin/env python3
"""
Test All Violence Detection Models
Tests all 5 models on both violent and non-violent videos
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU for stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
cv2.setLogLevel(0)

# ============================================================================
# CUSTOM LAYERS (must match training architecture)
# ============================================================================

class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer for focusing on important frames."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(1, use_bias=False)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'models_dir': '/home/admin/Desktop/NexaraVision/downloaded_models',
    'models': [
        'best_model.h5',
        'ultimate_best_model.h5',
        'ensemble_m1_best.h5',
        'ensemble_m2_best.h5',
        'ensemble_m3_best.h5'
    ],
    'test_videos': {
        'violent': [
            '/home/admin/Downloads/WF/archive/RWF-2000/train/Fight/V726w0IKCp4_2.avi',
            '/home/admin/Downloads/WF/archive/RWF-2000/train/Fight/Ile3EVQA_1.avi',
            '/home/admin/Downloads/WF/archive/RWF-2000/train/Fight/oowA6LToeTE_0.avi'
        ],
        'non_violent': [
            '/home/admin/Downloads/WF/archive/RWF-2000/train/NonFight/1A3zEkCHBl8_0.avi',
            '/home/admin/Downloads/WF/archive/RWF-2000/train/NonFight/LbYQs9oA_0.avi',
            '/home/admin/Downloads/WF/archive/RWF-2000/train/NonFight/HW48LUZKOL4_1.avi'
        ]
    },
    'num_frames': 20,
    'frame_size': (224, 224)
}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def extract_frames(video_path):
    """Extract frames uniformly from video"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"    ❌ Cannot open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if total_frames < CONFIG['num_frames']:
        print(f"    ⚠️  Too short ({total_frames} frames)")
        cap.release()
        return None

    # Extract frames uniformly
    indices = np.linspace(0, total_frames - 1, CONFIG['num_frames'], dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, CONFIG['frame_size'])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            frames.append(frame)

    cap.release()

    if len(frames) < CONFIG['num_frames']:
        print(f"    ❌ Only extracted {len(frames)} frames")
        return None

    return np.array(frames)

def extract_vgg19_features(frames):
    """Extract VGG19 features from frames"""
    # Load VGG19 once
    if not hasattr(extract_vgg19_features, 'extractor'):
        print("    🔍 Loading VGG19 (one-time)...")
        base = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        extract_vgg19_features.extractor = tf.keras.Model(
            base.input,
            base.get_layer('fc2').output
        )

    # Extract features
    preprocessed = tf.keras.applications.vgg19.preprocess_input(frames)
    features = extract_vgg19_features.extractor.predict(
        preprocessed,
        verbose=0,
        batch_size=20
    )

    # Normalize
    features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
    return features

# ============================================================================
# MODEL TESTING
# ============================================================================

def load_model(model_path):
    """Load model with custom layers - supports Keras 3.x"""
    # Try strategy 1: Standard loading
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer},
            compile=False
        )
        return model
    except Exception:
        pass

    # Try strategy 2: safe_mode=False for Keras 3.x
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer},
            compile=False,
            safe_mode=False
        )
        return model
    except Exception as e:
        print(f"    ❌ Failed to load: {str(e)[:60]}")
        return None

def test_model_on_video(model, features):
    """Run prediction on video features"""
    pred = model.predict(np.expand_dims(features, 0), verbose=0)
    prob_nv = pred[0][0] * 100
    prob_v = pred[0][1] * 100
    return prob_nv, prob_v

# ============================================================================
# MAIN TEST FLOW
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("🧪 TESTING ALL VIOLENCE DETECTION MODELS")
    print("=" * 80 + "\n")

    # Load all models
    print("📥 LOADING MODELS...")
    print("-" * 80)
    models = {}
    for model_name in CONFIG['models']:
        model_path = Path(CONFIG['models_dir']) / model_name
        if not model_path.exists():
            print(f"  ⚠️  {model_name}: NOT FOUND")
            continue

        print(f"  📦 {model_name}...", end=" ")
        model = load_model(str(model_path))
        if model:
            models[model_name] = model
            print(f"✅ ({model.count_params():,} params)")
        else:
            print()

    if not models:
        print("\n❌ No models loaded successfully")
        return

    print(f"\n✅ Loaded {len(models)} models\n")

    # Test on videos
    results = {}

    for category in ['violent', 'non_violent']:
        print("=" * 80)
        print(f"🎬 TESTING {category.upper().replace('_', '-')} VIDEOS")
        print("=" * 80 + "\n")

        results[category] = {}

        for video_path in CONFIG['test_videos'][category]:
            video_path = Path(video_path)
            if not video_path.exists():
                print(f"  ⚠️  {video_path.name}: NOT FOUND")
                continue

            print(f"📹 {video_path.name}")
            print("-" * 80)

            # Extract frames
            print("  🎬 Extracting frames...")
            frames = extract_frames(video_path)
            if frames is None:
                print()
                continue

            # Extract features
            print("  🎯 Extracting VGG19 features...")
            features = extract_vgg19_features(frames)
            print(f"  ✅ Features ready: {features.shape}")

            # Test all models
            print("\n  🔮 PREDICTIONS:")
            print("  " + "─" * 76)

            video_results = {}
            for model_name, model in models.items():
                prob_nv, prob_v = test_model_on_video(model, features)
                video_results[model_name] = (prob_nv, prob_v)

                # Format model name
                display_name = model_name.replace('_best.h5', '').replace('.h5', '')

                # Display result
                if prob_v > prob_nv:
                    result = f"🚨 VIOLENT"
                    conf = prob_v
                else:
                    result = f"✅ NON-VIOLENT"
                    conf = prob_nv

                bar = '█' * int(conf / 3)
                print(f"  {display_name:20s} → {result:15s} ({conf:5.1f}%) {bar}")

            results[category][video_path.name] = video_results
            print("  " + "─" * 76 + "\n")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY - MODEL ACCURACY")
    print("=" * 80 + "\n")

    for model_name in models.keys():
        print(f"🔹 {model_name}")

        # Calculate accuracy
        correct = 0
        total = 0

        for category in ['violent', 'non_violent']:
            for video_name, video_results in results[category].items():
                if model_name in video_results:
                    prob_nv, prob_v = video_results[model_name]
                    predicted_violent = prob_v > prob_nv
                    actual_violent = category == 'violent'

                    if predicted_violent == actual_violent:
                        correct += 1
                    total += 1

        if total > 0:
            accuracy = (correct / total) * 100
            print(f"   Accuracy: {correct}/{total} = {accuracy:.1f}%")
        else:
            print(f"   No results")
        print()

    print("=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
