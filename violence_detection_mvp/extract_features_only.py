#!/usr/bin/env python3
"""
Extract and cache VGG19 features BEFORE training
This runs once and saves features to disk for fast training
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import cv2
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH


def load_videos_from_folder_structure(dataset_path: Path):
    """Load video paths from organized dataset."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']

    splits = {}

    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split

        if not split_dir.exists():
            print(f"❌ ERROR: {split_dir} not found")
            sys.exit(1)

        videos = []
        labels = []

        # Load violent videos (label = 1)
        violent_dir = split_dir / 'violent'
        if violent_dir.exists():
            for ext in video_extensions:
                for video in violent_dir.glob(f'*{ext}'):
                    videos.append(str(video))
                    labels.append(1)

        # Load non-violent videos (label = 0)
        nonviolent_dir = split_dir / 'nonviolent'
        if nonviolent_dir.exists():
            for ext in video_extensions:
                for video in nonviolent_dir.glob(f'*{ext}'):
                    videos.append(str(video))
                    labels.append(0)

        splits[split] = (videos, np.array(labels))

        violent_count = np.sum(splits[split][1] == 1)
        nonviolent_count = np.sum(splits[split][1] == 0)

        print(f"✅ {split.upper():5s}: {len(videos):6,} videos "
              f"({violent_count:,} violent, {nonviolent_count:,} non-violent)")

    return splits


def extract_vgg19_features(
    video_paths: list,
    labels: np.ndarray,
    cache_dir: Path,
    split_name: str,
    batch_size: int = 8
):
    """Extract VGG19 features from videos."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path = cache_dir / f"{split_name}_features.npy"
    labels_path = cache_dir / f"{split_name}_labels.npy"

    # Check cache
    if features_path.exists() and labels_path.exists():
        print(f"   ✅ Features already cached at {features_path}")
        print(f"   Loading from cache...")
        features = np.load(features_path)
        cached_labels = np.load(labels_path)
        print(f"   ✅ Loaded: {features.shape}")
        return features, cached_labels

    print(f"   🔍 Extracting VGG19 features for {len(video_paths):,} videos...")

    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input

    # Load VGG19 (fc2 layer = 4096-dim features)
    print(f"   Loading VGG19 model...")
    base_model = VGG19(weights='imagenet', include_top=True)
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )

    features = []

    for video_path in tqdm(video_paths, desc=f"   Extracting {split_name}"):
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample SEQUENCE_LENGTH frames uniformly
            if total_frames > SEQUENCE_LENGTH:
                indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
            else:
                indices = list(range(total_frames))
                while len(indices) < SEQUENCE_LENGTH:
                    indices.append(indices[-1])

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                    frames.append(frame)
                else:
                    if len(frames) > 0:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))

            cap.release()

            # Extract features for all frames
            frames = np.array(frames)
            frames = preprocess_input(frames)

            frame_features = feature_extractor.predict(frames, batch_size=batch_size, verbose=0)
            features.append(frame_features)

        except Exception as e:
            print(f"\n   ⚠️  Error processing {Path(video_path).name}: {e}")
            features.append(np.zeros((SEQUENCE_LENGTH, 4096)))

    features = np.array(features)

    # Cache features
    print(f"\n   💾 Saving features to {features_path}")
    np.save(features_path, features)
    np.save(labels_path, labels)

    print(f"   ✅ Saved: {features.shape}")

    return features, labels


def main():
    parser = argparse.ArgumentParser(description='Extract VGG19 features for training')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to organized dataset folder')
    parser.add_argument('--cache-dir', type=str, default='./feature_cache',
                       help='Directory for caching features')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for VGG19 inference')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("VGG19 FEATURE EXTRACTION")
    print("="*80)
    print()

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  No GPU found, using CPU (will be slow)")

    # Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    print()

    dataset_path = Path(args.dataset_path)
    splits = load_videos_from_folder_structure(dataset_path)

    train_videos, train_labels = splits['train']
    val_videos, val_labels = splits['val']
    test_videos, test_labels = splits['test']

    # Extract features
    print("\n" + "="*80)
    print("EXTRACTING VGG19 FEATURES")
    print("="*80)
    print()

    cache_dir = Path(args.cache_dir)

    print("📊 Training set:")
    train_features, train_labels = extract_vgg19_features(
        train_videos, train_labels, cache_dir, 'train', batch_size=args.batch_size
    )

    print("\n📊 Validation set:")
    val_features, val_labels = extract_vgg19_features(
        val_videos, val_labels, cache_dir, 'val', batch_size=args.batch_size
    )

    print("\n📊 Test set:")
    test_features, test_labels = extract_vgg19_features(
        test_videos, test_labels, cache_dir, 'test', batch_size=args.batch_size
    )

    print("\n" + "="*80)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"\n📁 Features saved to: {cache_dir}")
    print(f"   Train: {train_features.shape}")
    print(f"   Val:   {val_features.shape}")
    print(f"   Test:  {test_features.shape}")
    print()
    print("🎯 NEXT STEP:")
    print("   Start training:")
    print(f"   python3 train_rtx5000_dual_optimized.py --dataset-path {args.dataset_path} --cache-dir {args.cache_dir} --epochs 100 --batch-size 64")
    print()


if __name__ == "__main__":
    main()
