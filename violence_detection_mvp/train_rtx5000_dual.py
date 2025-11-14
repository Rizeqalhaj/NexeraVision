"""
Training script for 2× RTX 5000 Ada Generation
Adapted from runpod_train_l40s.py for dual GPU setup

Usage:
    python train_rtx5000_dual.py --dataset-path /workspace/organized_dataset
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List
import argparse
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH
from src.model_architecture import ViolenceDetectionModel


def check_gpus():
    """Check GPU availability and configure for 2× RTX 5000 Ada."""
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("❌ ERROR: No GPU detected!")
        return False, 0

    print(f"✅ GPUs Available: {len(gpus)} GPU(s)")

    # Enable memory growth for all GPUs
    for i, gpu in enumerate(gpus):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   GPU {i}: Memory growth enabled")
        except RuntimeError as e:
            print(f"   GPU {i}: Memory growth error: {e}")

    # Get GPU info
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                               '--format=csv,noheader'],
                              capture_output=True, text=True)

        gpu_lines = result.stdout.strip().split('\n')
        total_memory = 0

        for i, line in enumerate(gpu_lines):
            print(f"   GPU {i}: {line.strip()}")
            memory_str = line.split(',')[1].strip()
            memory_gb = int(memory_str.split()[0]) / 1024
            total_memory += memory_gb

        print(f"\n   Total VRAM: {total_memory:.1f} GB")

        if 'RTX 5000 Ada' in result.stdout or 'RTX 5000' in result.stdout:
            print("   ✅ RTX 5000 Ada Detected - Optimizing for 64 GB total VRAM")
    except Exception as e:
        print(f"   Could not get GPU info: {e}")
        total_memory = 32

    return True, total_memory


def load_videos_from_folder_structure(dataset_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Load videos from physical folder structure:
    dataset_path/
        train/violent/
        train/nonviolent/
        val/violent/
        val/nonviolent/
        test/violent/
        test/nonviolent/
    """
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract VGG19 features from videos."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path = cache_dir / f"{split_name}_features.npy"
    labels_path = cache_dir / f"{split_name}_labels.npy"

    # Check cache
    if features_path.exists() and labels_path.exists():
        print(f"   📦 Loading cached features from {features_path}")
        features = np.load(features_path)
        cached_labels = np.load(labels_path)
        return features, cached_labels

    print(f"   🔍 Extracting VGG19 features for {len(video_paths):,} videos...")

    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
    import cv2
    from tqdm import tqdm

    # Load VGG19 (fc2 layer = 4096-dim features)
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
                    # Use last valid frame if read fails
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
            # Add zero features for failed videos
            features.append(np.zeros((SEQUENCE_LENGTH, 4096)))

    features = np.array(features)

    # Cache features
    print(f"   💾 Caching features to {features_path}")
    np.save(features_path, features)
    np.save(labels_path, labels)

    return features, labels


def create_mirrored_strategy():
    """Create TensorFlow MirroredStrategy for multi-GPU training."""
    strategy = tf.distribute.MirroredStrategy()
    print(f"\n🚀 Multi-GPU Strategy Created")
    print(f"   Number of devices: {strategy.num_replicas_in_sync}")
    return strategy


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train violence detection model on 2× RTX 5000 Ada')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to organized dataset folder')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (will be split across GPUs)')
    parser.add_argument('--cache-dir', type=str, default='/workspace/feature_cache',
                       help='Directory for caching features')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("VIOLENCE DETECTION TRAINING - 2× RTX 5000 Ada Generation")
    print("="*80)
    print()

    # Check GPUs
    gpu_available, total_memory = check_gpus()
    if not gpu_available:
        sys.exit(1)

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

    print(f"\n✅ Feature extraction complete!")
    print(f"   Train features shape: {train_features.shape}")
    print(f"   Val features shape: {val_features.shape}")
    print(f"   Test features shape: {test_features.shape}")

    # Create multi-GPU strategy
    strategy = create_mirrored_strategy()

    # Build model within strategy scope
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    print()

    with strategy.scope():
        model_builder = ViolenceDetectionModel()
        model = model_builder.build_model()

        print("✅ Model built successfully")
        print(f"   Total parameters: {model.count_params():,}")

    # Train model
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print()

    # Create checkpoint directory
    checkpoint_dir = Path('/workspace/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(checkpoint_dir / 'training_log.csv')
        )
    ]

    # Train
    history = model.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    print()

    test_loss, test_accuracy = model.evaluate(test_features, test_labels, verbose=1)

    print(f"\n✅ Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy*100:.2f}%")

    # Save final results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'total_epochs': len(history.history['loss']),
        'dataset_path': str(args.dataset_path),
        'timestamp': datetime.now().isoformat()
    }

    results_file = checkpoint_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best model saved: {checkpoint_dir / 'best_model.h5'}")
    print(f"   Results saved: {results_file}")
    print()


if __name__ == "__main__":
    main()
