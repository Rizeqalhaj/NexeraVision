#!/usr/bin/env python3
"""
Test trained violence detection model on a single video
Usage: python3 test_single_video.py /path/to/video.mp4
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from tensorflow.keras import layers, models, regularizers

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
cv2.setLogLevel(0)

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'model_path': '/workspace/violence_detection_mvp/models/best_model.h5',
    'num_frames': 20,
    'frame_size': (224, 224),
}

# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

def build_model():
    """Build exact model architecture from training"""
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
# VIDEO PROCESSING
# ============================================================================

def extract_frames(video_path):
    """Extract frames from video"""
    print(f"  📹 Opening video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  ❌ ERROR: Cannot open video file")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0

    print(f"  ℹ️  Total frames: {total_frames}")
    print(f"  ℹ️  FPS: {fps}")
    print(f"  ℹ️  Duration: {duration:.2f} seconds")

    if total_frames < CONFIG['num_frames']:
        print(f"  ⚠️  WARNING: Video has only {total_frames} frames (need {CONFIG['num_frames']})")
        cap.release()
        return None

    # Extract frames
    print(f"  🎬 Extracting {CONFIG['num_frames']} frames...")
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
        print(f"  ❌ ERROR: Could only extract {len(frames)} frames")
        return None

    print(f"  ✅ Extracted {len(frames)} frames")
    return np.array(frames)

def extract_vgg19_features(frames):
    """Extract VGG19 features from frames"""
    print(f"  🔍 Loading VGG19...")

    base_model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )

    print(f"  🎯 Extracting VGG19 features...")
    frames_preprocessed = tf.keras.applications.vgg19.preprocess_input(frames)
    features = feature_extractor.predict(frames_preprocessed, verbose=0, batch_size=20)

    # Normalize
    features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

    print(f"  ✅ Features extracted: {features.shape}")
    return features

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("🧪 SINGLE VIDEO VIOLENCE DETECTION TEST")
    print("=" * 80)
    print()

    # Get video path
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        # Default: look in tests directory
        tests_dir = Path('/workspace/violence_detection_mvp/tests')
        videos = list(tests_dir.glob('*.mp4')) + list(tests_dir.glob('*.avi'))

        if not videos:
            print("❌ No videos found in /workspace/violence_detection_mvp/tests")
            print("\nUsage: python3 test_single_video.py /path/to/video.mp4")
            sys.exit(1)

        video_path = videos[0]
        print(f"📁 Auto-detected video: {video_path.name}")
        print()

    if not video_path.exists():
        print(f"❌ ERROR: Video not found: {video_path}")
        sys.exit(1)

    # ========================================================================
    # LOAD MODEL
    # ========================================================================

    print("=" * 80)
    print("📥 LOADING MODEL")
    print("=" * 80)

    model_path = Path(CONFIG['model_path'])
    if not model_path.exists():
        print(f"❌ ERROR: Model not found: {model_path}")
        sys.exit(1)

    print(f"  📂 Model: {model_path}")

    # Build architecture and load weights
    model = build_model()
    model.load_weights(str(model_path))

    print(f"  ✅ Model loaded ({model.count_params():,} parameters)")
    print()

    # ========================================================================
    # PROCESS VIDEO
    # ========================================================================

    print("=" * 80)
    print("🎬 PROCESSING VIDEO")
    print("=" * 80)

    # Extract frames
    frames = extract_frames(video_path)
    if frames is None:
        print("\n❌ Failed to extract frames")
        sys.exit(1)

    # Extract features
    features = extract_vgg19_features(frames)

    print()

    # ========================================================================
    # PREDICT
    # ========================================================================

    print("=" * 80)
    print("🔮 PREDICTION")
    print("=" * 80)

    # Run prediction
    features_input = np.expand_dims(features, axis=0)  # Add batch dimension
    prediction = model.predict(features_input, verbose=0)

    # Get probabilities
    prob_nonviolent = prediction[0][0] * 100
    prob_violent = prediction[0][1] * 100

    # Determine class
    predicted_class = "VIOLENT" if prob_violent > prob_nonviolent else "NON-VIOLENT"
    confidence = max(prob_violent, prob_nonviolent)

    # Display results
    print()
    print(f"  📊 Prediction Results:")
    print(f"     {'─' * 50}")
    print(f"     Video: {video_path.name}")
    print(f"     {'─' * 50}")
    print(f"     Non-Violent: {prob_nonviolent:6.2f}%  {'█' * int(prob_nonviolent/2)}")
    print(f"     Violent:     {prob_violent:6.2f}%  {'█' * int(prob_violent/2)}")
    print(f"     {'─' * 50}")
    print()

    # Final verdict
    if predicted_class == "VIOLENT":
        icon = "🚨"
    else:
        icon = "✅"

    print(f"  {icon} PREDICTION: {predicted_class}")
    print(f"  📈 Confidence: {confidence:.2f}%")
    print()

    # Confidence interpretation
    if confidence >= 90:
        print(f"  💪 Very high confidence")
    elif confidence >= 75:
        print(f"  👍 High confidence")
    elif confidence >= 60:
        print(f"  🤔 Moderate confidence")
    else:
        print(f"  ⚠️  Low confidence - uncertain prediction")

    print()
    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    print()

if __name__ == '__main__':
    main()
