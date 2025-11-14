#!/usr/bin/env python3
"""
Auto-detect and load violence detection model with multiple strategies
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

# Make tf globally available for Lambda layers
import builtins
builtins.tf = tf

# AttentionLayer definition
class AttentionLayer(tf.keras.layers.Layer):
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

def try_load_model(model_path):
    """Try different loading strategies"""
    strategies = [
        {
            'name': 'Strategy 1: safe_mode=False only',
            'kwargs': {'compile': False, 'safe_mode': False}
        },
        {
            'name': 'Strategy 2: safe_mode=False + AttentionLayer',
            'kwargs': {'compile': False, 'safe_mode': False, 'custom_objects': {'AttentionLayer': AttentionLayer}}
        },
        {
            'name': 'Strategy 3: AttentionLayer only',
            'kwargs': {'compile': False, 'custom_objects': {'AttentionLayer': AttentionLayer}}
        },
        {
            'name': 'Strategy 4: All custom objects',
            'kwargs': {'compile': False, 'safe_mode': False, 'custom_objects': {'AttentionLayer': AttentionLayer, 'tf': tf}}
        },
    ]

    for strategy in strategies:
        try:
            print(f"\n  Trying: {strategy['name']}")
            model = tf.keras.models.load_model(model_path, **strategy['kwargs'])
            print(f"  ✅ SUCCESS! Model loaded ({model.count_params():,} parameters)")
            return model, strategy['name']
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  ❌ Failed: {error_msg}")

    return None, None

def extract_frames(video_path):
    """Extract 20 frames uniformly from video"""
    cap = cv2.VideoCapture(str(video_path))
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
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            frames.append(frame)

    cap.release()
    return np.array(frames) if len(frames) == 20 else None

def main():
    print("=" * 80)
    print("🧪 AUTO-LOAD VIOLENCE DETECTION TEST")
    print("=" * 80)

    # Find video
    tests_dir = Path('/workspace/violence_detection_mvp/tests')
    videos = list(tests_dir.glob('*.mp4')) + list(tests_dir.glob('*.avi'))

    if not videos:
        print("\n❌ No videos in tests directory")
        return

    video_path = videos[0]
    print(f"\n📁 Video: {video_path.name}")

    # Try to load model
    model_path = '/workspace/violence_detection_mvp/models/best_model.h5'
    print(f"\n📥 Loading model: {model_path}")
    print("-" * 80)

    model, strategy = try_load_model(model_path)

    if not model:
        print("\n❌ All loading strategies failed")
        return

    print(f"\n✅ Model loaded successfully using: {strategy}")

    # Extract frames
    print("\n" + "=" * 80)
    print("🎬 PROCESSING VIDEO")
    print("=" * 80)

    frames = extract_frames(video_path)
    if frames is None:
        print("\n❌ Failed to extract frames")
        return

    print(f"  ✅ Extracted {len(frames)} frames")

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
