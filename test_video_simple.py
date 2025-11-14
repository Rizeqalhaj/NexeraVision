#!/usr/bin/env python3
"""
Simple Violence Detection Test - Works with any model format
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
cv2.setLogLevel(0)

# Make tf globally available
import builtins
builtins.tf = tf

def extract_frames(video_path, num_frames=20):
    """Extract frames uniformly from video"""
    print(f"  📹 Opening: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"  ℹ️  Frames: {total}, FPS: {fps}, Duration: {total/fps:.2f}s")

    if total < num_frames:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        return None

    print(f"  ✅ Extracted {len(frames)} frames")
    return np.array(frames)

def try_load_model(model_path):
    """Try different loading strategies"""
    print(f"\n📥 Loading model: {model_path}")

    # Strategy 1: Try with safe_mode=False
    try:
        print("  Trying: safe_mode=False...")
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        print(f"  ✅ Loaded ({model.count_params():,} params)")
        return model
    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:80]}")

    # Strategy 2: Try loading just weights
    try:
        print("  Trying: weights-only loading...")
        # This would need architecture definition
        print("  ⚠️  Skipped (needs architecture)")
    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:80]}")

    return None

def main():
    print("=" * 80)
    print("🧪 VIOLENCE DETECTION TEST (Simple)")
    print("=" * 80)

    # Find video
    tests_dir = Path('/workspace/violence_detection_mvp/tests')
    videos = list(tests_dir.glob('*.mp4')) + list(tests_dir.glob('*.avi'))

    if not videos:
        print("❌ No videos in tests directory")
        return

    video_path = videos[0]
    print(f"\n📁 Video: {video_path.name}")

    # Find model - try different model files
    models_dir = Path('/workspace/violence_detection_mvp/models')
    model_files = [
        'best_model.h5',
        'violence_detection_model.h5',
        'model.h5',
        'final_model.h5'
    ]

    model = None
    for model_name in model_files:
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"\n🔍 Found: {model_name}")
            model = try_load_model(model_path)
            if model:
                break

    if not model:
        print("\n❌ Could not load any model")
        print("\n📋 Available models:")
        for f in models_dir.glob('*.h5'):
            print(f"   - {f.name}")
        return

    # Extract frames
    print("\n" + "=" * 80)
    print("🎬 PROCESSING VIDEO")
    print("=" * 80)

    frames = extract_frames(video_path)
    if frames is None:
        print("❌ Failed to extract frames")
        return

    # Extract VGG19 features
    print("  🔍 Loading VGG19...")
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    extractor = tf.keras.Model(vgg.input, vgg.get_layer('fc2').output)

    print("  🎯 Extracting features...")
    preprocessed = tf.keras.applications.vgg19.preprocess_input(frames)
    features = extractor.predict(preprocessed, verbose=0, batch_size=20)
    features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
    print(f"  ✅ Features: {features.shape}")

    # Predict
    print("\n" + "=" * 80)
    print("🔮 PREDICTION")
    print("=" * 80)

    pred = model.predict(np.expand_dims(features, 0), verbose=0)

    prob_nv = pred[0][0] * 100
    prob_v = pred[0][1] * 100

    print(f"\n  📊 Results:")
    print(f"     {'─' * 50}")
    print(f"     Non-Violent: {prob_nv:6.2f}%  {'█' * int(prob_nv/2)}")
    print(f"     Violent:     {prob_v:6.2f}%  {'█' * int(prob_v/2)}")
    print(f"     {'─' * 50}")

    if prob_v > prob_nv:
        print(f"\n  🚨 PREDICTION: VIOLENT ({prob_v:.2f}%)")
    else:
        print(f"\n  ✅ PREDICTION: NON-VIOLENT ({prob_nv:.2f}%)")

    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
