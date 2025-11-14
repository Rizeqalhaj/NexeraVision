#!/usr/bin/env python3
"""
Extract features with checkpointing, then train
Solves the problem of restarting from scratch on crashes
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from src.config import Config

class CheckpointedFeatureExtractor:
    """Extract features with checkpointing to resume on failures"""

    def __init__(self, dataset_path, cache_path, checkpoint_path):
        self.dataset_path = Path(dataset_path)
        self.cache_path = Path(cache_path)
        self.checkpoint_path = Path(checkpoint_path)

        # Create directories
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Load VGG19
        print("Loading VGG19...")
        vgg19 = VGG19(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
        transfer_layer = vgg19.get_layer('fc2')
        self.model = Model(inputs=vgg19.input, outputs=transfer_layer.output)
        print("VGG19 loaded!")

    def extract_frames(self, video_path, n_frames=20):
        """Extract frames from video with timeout protection"""
        try:
            cap = cv2.VideoCapture(str(video_path))

            # Set timeouts
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None

            # Calculate frame indices
            if total_frames < n_frames:
                indices = list(range(total_frames))
                indices += [total_frames - 1] * (n_frames - total_frames)
            else:
                step = (total_frames - 1) / (n_frames - 1)
                indices = [int(round(i * step)) for i in range(n_frames)]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    # Convert BGR to RGB and resize
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (224, 224))
                    frames.append(resized)
                elif frames:
                    frames.append(frames[-1])  # Repeat last frame
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

            cap.release()
            return np.array(frames, dtype=np.float32)

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return None

    def extract_features_from_frames(self, frames):
        """Extract VGG19 features from frames"""
        if frames is None:
            return None

        # Preprocess for VGG19
        processed = preprocess_input(frames)

        # Extract features
        features = self.model.predict(processed, batch_size=4, verbose=0)
        return features

    def process_split(self, split_name):
        """Process train/val/test split with checkpointing"""
        print(f"\n{'='*70}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*70}\n")

        split_dir = self.dataset_path / split_name
        cache_file = self.cache_path / f"{split_name}_features.h5"
        checkpoint_file = self.checkpoint_path / f"{split_name}_checkpoint.txt"

        # Load checkpoint if exists
        processed_videos = set()
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                processed_videos = set(line.strip() for line in f)
            print(f"📂 Resuming: {len(processed_videos)} videos already processed")

        # Collect all videos
        all_videos = []
        for class_name in ['violent', 'nonviolent']:
            class_dir = split_dir / class_name
            if class_dir.exists():
                videos = list(class_dir.glob('*.mp4'))
                all_videos.extend([(v, class_name) for v in videos])

        # Filter out already processed
        remaining_videos = [(v, c) for v, c in all_videos if v.name not in processed_videos]

        print(f"Total videos: {len(all_videos)}")
        print(f"Already processed: {len(processed_videos)}")
        print(f"Remaining: {len(remaining_videos)}\n")

        if not remaining_videos:
            print(f"✅ {split_name} already complete!")
            return

        # Process remaining videos
        with h5py.File(cache_file, 'a') as hf:
            for video_path, class_name in tqdm(remaining_videos, desc=f"Extracting {split_name}"):
                try:
                    # Extract frames
                    frames = self.extract_frames(video_path)
                    if frames is None:
                        print(f"⚠️  Skipping corrupted: {video_path.name}")
                        # Mark as processed to skip next time
                        with open(checkpoint_file, 'a') as cf:
                            cf.write(f"{video_path.name}\n")
                        continue

                    # Extract features
                    features = self.extract_features_from_frames(frames)
                    if features is None:
                        continue

                    # Save to HDF5
                    video_id = f"{class_name}/{video_path.stem}"
                    if video_id in hf:
                        del hf[video_id]

                    hf.create_dataset(video_id, data=features, compression='gzip')

                    # Checkpoint
                    with open(checkpoint_file, 'a') as cf:
                        cf.write(f"{video_path.name}\n")

                except Exception as e:
                    print(f"✗ Error processing {video_path.name}: {e}")
                    continue

        print(f"\n✅ {split_name.upper()} extraction complete!")

    def extract_all(self):
        """Extract features for all splits"""
        for split in ['train', 'val', 'test']:
            self.process_split(split)

        print(f"\n{'='*70}")
        print("✅ ALL FEATURE EXTRACTION COMPLETE!")
        print(f"{'='*70}\n")

def main():
    dataset_path = "/workspace/organized_dataset"
    cache_path = "/workspace/features_checkpointed"
    checkpoint_path = "/workspace/feature_checkpoints"

    print("="*70)
    print("Checkpointed Feature Extraction")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Cache: {cache_path}")
    print(f"Checkpoints: {checkpoint_path}")
    print("="*70)

    extractor = CheckpointedFeatureExtractor(dataset_path, cache_path, checkpoint_path)
    extractor.extract_all()

    print("\n🚀 Now run training with pre-extracted features!")
    print("python3 train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset --cache-dir /workspace/features_checkpointed")

if __name__ == "__main__":
    main()
