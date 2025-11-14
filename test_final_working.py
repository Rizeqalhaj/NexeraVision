#!/usr/bin/env python3
"""
Violence Detection Test - Based on inspected model architecture
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

# ============================================================================
# CUSTOM LAYERS - Must match exact training architecture
# ============================================================================

class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer - matches saved model weights exactly"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W = None

    def build(self, input_shape):
        # Create weight directly (not using Dense sublayer)
        self.W = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores: inputs @ W
        attention_scores = tf.matmul(inputs, self.W)  # (batch, time, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return super(AttentionLayer, self).get_config()

CONFIG = {
    'model_path': '/workspace/violence_detection_mvp/models/best_model.h5',
    'num_frames': 20,
    'frame_size': (224, 224),
}

def extract_frames(video_path):
    """Extract 20 frames uniformly from video"""
    print(f"  📹 Opening: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("  ❌ Cannot open video")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"  ℹ️  Frames: {total}, FPS: {fps}, Duration: {total/fps:.1f}s")

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

    if len(frames) < 20:
        return None

    print(f"  ✅ Extracted {len(frames)} frames")
    return np.array(frames)

def main():
    print("=" * 80)
    print("🧪 VIOLENCE DETECTION TEST")
    print("=" * 80 + "\n")

    # Find video
    tests_dir = Path('/workspace/violence_detection_mvp/tests')
    videos = list(tests_dir.glob('*.mp4')) + list(tests_dir.glob('*.avi'))

    if not videos:
        print("❌ No videos found")
        sys.exit(1)

    video_path = videos[0]
    print(f"📁 Video: {video_path.name}\n")

    # Load model
    print("=" * 80)
    print("📥 LOADING MODEL")
    print("=" * 80)
    print(f"  📂 Path: {CONFIG['model_path']}")

    try:
        model = tf.keras.models.load_model(
            CONFIG['model_path'],
            custom_objects={'AttentionLayer': AttentionLayer},
            compile=False
        )
        print(f"  ✅ Loaded successfully ({model.count_params():,} parameters)")
        print(f"  ℹ️  Layers: {len(model.layers)}")
        print(f"  ℹ️  Input: {model.input_shape}, Output: {model.output_shape}\n")
    except Exception as e:
        print(f"  ❌ Loading failed: {e}")
        sys.exit(1)

    # Process video
    print("=" * 80)
    print("🎬 PROCESSING VIDEO")
    print("=" * 80)

    frames = extract_frames(video_path)
    if frames is None:
        print("❌ Failed to extract frames")
        sys.exit(1)

    # Extract VGG19 features
    print("  🔍 Loading VGG19...")
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    extractor = tf.keras.Model(vgg.input, vgg.get_layer('fc2').output)

    print("  🎯 Extracting features...")
    preprocessed = tf.keras.applications.vgg19.preprocess_input(frames)
    features = extractor.predict(preprocessed, verbose=0, batch_size=20)
    features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
    print(f"  ✅ Features: {features.shape}\n")

    # Predict
    print("=" * 80)
    print("🔮 PREDICTION")
    print("=" * 80)

    pred = model.predict(np.expand_dims(features, 0), verbose=0)

    prob_nv = pred[0][0] * 100
    prob_v = pred[0][1] * 100

    print(f"\n  📊 Results:")
    print(f"     {'─' * 50}")
    print(f"     Non-Violent: {prob_nv:6.2f}%  {'█' * int(prob_nv/2)}")
    print(f"     Violent:     {prob_v:6.2f}%  {'█' * int(prob_v/2)}")
    print(f"     {'─' * 50}\n")

    if prob_v > prob_nv:
        print(f"  🚨 PREDICTION: VIOLENT")
        print(f"  📈 Confidence: {prob_v:.2f}%\n")
    else:
        print(f"  ✅ PREDICTION: NON-VIOLENT")
        print(f"  📈 Confidence: {prob_nv:.2f}%\n")

    print("=" * 80)
    print("✅ COMPLETE")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
