#!/usr/bin/env python3
"""
Separate process for CPU-only feature extraction
This must be run as a separate script to avoid CUDA context issues
"""

import os
# CRITICAL: Set CPU-only BEFORE any imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import tensorflow as tf
import cv2

def extract_features(video_path):
    """Extract VGG19 features from video - CPU only"""
    try:
        # Suppress stderr
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        cap = cv2.VideoCapture(str(video_path))

        sys.stderr.close()
        sys.stderr = original_stderr

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 20:
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

        if len(frames) < 16:
            return None

        while len(frames) < 20:
            frames.append(frames[-1])

        frames_array = np.array(frames[:20])

        # Load VGG19 (will be CPU-only due to env var)
        base_model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('fc2').output
        )

        # Extract features
        frames_preprocessed = tf.keras.applications.vgg19.preprocess_input(frames_array)
        features = feature_extractor.predict(frames_preprocessed, verbose=0, batch_size=20)
        features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

        return features

    except Exception as e:
        return None

if __name__ == '__main__':
    # This script is called by the main process with video path as argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        features = extract_features(video_path)

        if features is not None:
            # Save to temp file
            output_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/features.npy'
            np.save(output_path, features)
            print("SUCCESS")
        else:
            print("FAILED")
