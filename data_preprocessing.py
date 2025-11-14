#!/usr/bin/env python3
"""
NexaraVision Data Preprocessing Pipeline
Handles video loading, frame extraction, and data augmentation
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import random

class VideoDataPreprocessor:
    """Preprocesses video data for violence detection"""

    def __init__(self,
                 dataset_dir="/workspace/datasets/tier1",
                 frames_per_video=20,
                 img_size=(224, 224),
                 train_split=0.8,
                 val_split=0.1,
                 test_split=0.1):
        """
        Initialize preprocessor

        Args:
            dataset_dir: Path to dataset directory
            frames_per_video: Number of frames to extract per video
            img_size: Target image size (height, width)
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
        """
        self.dataset_dir = Path(dataset_dir)
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # Verify splits sum to 1
        assert abs(train_split + val_split + test_split - 1.0) < 0.001, \
            "Splits must sum to 1.0"

        self.video_paths = []
        self.labels = []
        self.dataset_names = []

    def scan_datasets(self):
        """Scan all datasets and collect video paths with labels"""

        print("=" * 80)
        print("Scanning Datasets")
        print("=" * 80)

        datasets = {
            'RWF2000': self._scan_rwf2000,
            'UCF_Crime': self._scan_ucf_crime,
            'SCVD': self._scan_scvd,
            'RealLife': self._scan_reallife
        }

        for dataset_name, scan_func in datasets.items():
            dataset_path = self.dataset_dir / dataset_name

            if dataset_path.exists():
                print(f"\n📂 Scanning {dataset_name}...")
                videos, labels = scan_func(dataset_path)

                self.video_paths.extend(videos)
                self.labels.extend(labels)
                self.dataset_names.extend([dataset_name] * len(videos))

                violence_count = sum(labels)
                non_violence_count = len(labels) - violence_count

                print(f"   Violence: {violence_count:,}")
                print(f"   Non-Violence: {non_violence_count:,}")
                print(f"   Total: {len(videos):,}")
            else:
                print(f"\n⚠️  {dataset_name} not found, skipping")

        print("\n" + "=" * 80)
        print(f"Total Videos: {len(self.video_paths):,}")
        print(f"Violence: {sum(self.labels):,}")
        print(f"Non-Violence: {len(self.labels) - sum(self.labels):,}")
        print("=" * 80)

        return len(self.video_paths)

    def _scan_rwf2000(self, dataset_path):
        """Scan RWF2000 dataset structure"""
        videos = []
        labels = []

        # RWF2000 structure: train/Fight, train/NonFight, val/Fight, val/NonFight
        for split in ['train', 'val']:
            # Violence (Fight) videos
            fight_dir = dataset_path / split / 'Fight'
            if fight_dir.exists():
                for video in fight_dir.glob('*.avi'):
                    videos.append(str(video))
                    labels.append(1)  # Violence

            # Non-violence (NonFight) videos
            nonfight_dir = dataset_path / split / 'NonFight'
            if nonfight_dir.exists():
                for video in nonfight_dir.glob('*.avi'):
                    videos.append(str(video))
                    labels.append(0)  # Non-violence

        return videos, labels

    def _scan_ucf_crime(self, dataset_path):
        """Scan UCF_Crime dataset structure"""
        videos = []
        labels = []

        # UCF_Crime structure: Test/, Train/ with category folders
        violence_keywords = ['abuse', 'arrest', 'arson', 'assault', 'burglary',
                            'explosion', 'fighting', 'robbery', 'shooting',
                            'shoplifting', 'stealing', 'vandalism', 'violence']

        for split in ['Train', 'Test']:
            split_dir = dataset_path / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        # Check if category indicates violence
                        is_violence = any(kw in category_dir.name.lower()
                                        for kw in violence_keywords)
                        label = 1 if is_violence else 0

                        # Add all videos in category
                        for video in category_dir.rglob('*.mp4'):
                            videos.append(str(video))
                            labels.append(label)

        return videos, labels

    def _scan_scvd(self, dataset_path):
        """Scan SCVD (SmartCity) dataset structure"""
        videos = []
        labels = []

        # SCVD typically has Violence/NonViolence folders
        for video in dataset_path.rglob('*.mp4'):
            path_str = str(video).lower()

            # Determine label from path
            if 'violence' in path_str or 'violent' in path_str:
                if 'non' not in path_str:
                    label = 1  # Violence
                else:
                    label = 0  # Non-violence
            elif 'nonviolence' in path_str or 'non_violence' in path_str:
                label = 0  # Non-violence
            else:
                # Default: check parent directory name
                if 'fight' in path_str or 'assault' in path_str:
                    label = 1
                else:
                    label = 0

            videos.append(str(video))
            labels.append(label)

        return videos, labels

    def _scan_reallife(self, dataset_path):
        """Scan Real-Life Violence dataset structure"""
        videos = []
        labels = []

        # Real-Life typically has Violence/NonViolence folders
        for video in dataset_path.rglob('*.mp4'):
            path_str = str(video).lower()

            # Determine label from path
            if 'violence' in path_str and 'non' not in path_str:
                label = 1  # Violence
            elif 'nonviolence' in path_str or 'non_violence' in path_str:
                label = 0  # Non-violence
            else:
                # Check parent directory
                parent_name = video.parent.name.lower()
                label = 1 if 'violence' in parent_name else 0

            videos.append(str(video))
            labels.append(label)

        return videos, labels

    def extract_frames(self, video_path, num_frames=None):
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: self.frames_per_video)

        Returns:
            numpy array of shape (num_frames, height, width, 3)
        """
        if num_frames is None:
            num_frames = self.frames_per_video

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            # Return black frames if video can't be opened
            return np.zeros((num_frames, *self.img_size, 3), dtype=np.float32)

        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            # If video has fewer frames, repeat frames
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # Uniformly sample frames
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Resize and normalize
                frame = cv2.resize(frame, self.img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
                frames.append(frame)
            else:
                # Add black frame if read fails
                frames.append(np.zeros((*self.img_size, 3), dtype=np.float32))

        cap.release()

        return np.array(frames)

    def create_splits(self, random_seed=42):
        """Create train/val/test splits"""

        print("\n" + "=" * 80)
        print("Creating Train/Val/Test Splits")
        print("=" * 80)

        # Convert to numpy arrays
        X = np.array(self.video_paths)
        y = np.array(self.labels)
        datasets = np.array(self.dataset_names)

        # Set random seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        # First split: separate test set
        X_temp, X_test, y_temp, y_test, ds_temp, ds_test = train_test_split(
            X, y, datasets,
            test_size=self.test_split,
            stratify=y,
            random_state=random_seed
        )

        # Second split: separate train and validation
        val_size_adjusted = self.val_split / (self.train_split + self.val_split)
        X_train, X_val, y_train, y_val, ds_train, ds_val = train_test_split(
            X_temp, y_temp, ds_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=random_seed
        )

        # Print split statistics
        print(f"\nTrain Set: {len(X_train):,} videos")
        print(f"  Violence: {sum(y_train):,} ({sum(y_train)/len(y_train)*100:.1f}%)")
        print(f"  Non-Violence: {len(y_train) - sum(y_train):,} ({(len(y_train)-sum(y_train))/len(y_train)*100:.1f}%)")

        print(f"\nValidation Set: {len(X_val):,} videos")
        print(f"  Violence: {sum(y_val):,} ({sum(y_val)/len(y_val)*100:.1f}%)")
        print(f"  Non-Violence: {len(y_val) - sum(y_val):,} ({(len(y_val)-sum(y_val))/len(y_val)*100:.1f}%)")

        print(f"\nTest Set: {len(X_test):,} videos")
        print(f"  Violence: {sum(y_test):,} ({sum(y_test)/len(y_test)*100:.1f}%)")
        print(f"  Non-Violence: {len(y_test) - sum(y_test):,} ({(len(y_test)-sum(y_test))/len(y_test)*100:.1f}%)")

        print("=" * 80)

        return {
            'train': (X_train, y_train, ds_train),
            'val': (X_val, y_val, ds_val),
            'test': (X_test, y_test, ds_test)
        }

    def save_splits(self, splits, output_path="/workspace/processed/splits.json"):
        """Save split information to file"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        splits_data = {}

        for split_name, (videos, labels, datasets) in splits.items():
            splits_data[split_name] = {
                'videos': videos.tolist(),
                'labels': labels.tolist(),
                'datasets': datasets.tolist(),
                'count': len(videos),
                'violence_count': int(sum(labels)),
                'non_violence_count': int(len(labels) - sum(labels))
            }

        with open(output_path, 'w') as f:
            json.dump(splits_data, f, indent=2)

        print(f"\n✅ Splits saved to: {output_path}")

def main():
    """Test preprocessing pipeline"""

    print("=" * 80)
    print("NexaraVision Data Preprocessing")
    print("=" * 80)

    # Initialize preprocessor
    preprocessor = VideoDataPreprocessor()

    # Scan datasets
    total_videos = preprocessor.scan_datasets()

    if total_videos == 0:
        print("\n❌ No videos found!")
        return

    # Create splits
    splits = preprocessor.create_splits()

    # Save splits
    preprocessor.save_splits(splits)

    # Test frame extraction on one video
    print("\n" + "=" * 80)
    print("Testing Frame Extraction")
    print("=" * 80)

    test_video = preprocessor.video_paths[0]
    print(f"\nTest video: {test_video}")
    print(f"Extracting {preprocessor.frames_per_video} frames...")

    frames = preprocessor.extract_frames(test_video)

    print(f"\n✅ Extracted frames shape: {frames.shape}")
    print(f"   Frame range: [{frames.min():.3f}, {frames.max():.3f}]")
    print(f"   Expected: (20, 224, 224, 3) in range [0, 1]")

    print("\n" + "=" * 80)
    print("✅ Preprocessing Pipeline Ready!")
    print("=" * 80)

if __name__ == "__main__":
    main()
