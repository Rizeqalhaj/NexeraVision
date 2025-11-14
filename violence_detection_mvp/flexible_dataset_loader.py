#!/usr/bin/env python3
"""
Flexible dataset loader that works with any class names
Supports: Fight/NonFight, Fight/Normal, violent/nonviolent, etc.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_dataset_flexible(dataset_path: str) -> Dict:
    """
    Load dataset with automatic class detection

    Supports various naming conventions:
    - Fight / NonFight
    - Fight / Normal
    - violent / nonviolent
    - violence / normal

    Returns:
        Dict with train/val/test splits containing paths and labels
    """
    dataset_path = Path(dataset_path)

    # Auto-detect class names from train directory
    train_dir = dataset_path / "train"
    if not train_dir.exists():
        raise ValueError(f"Training directory not found: {train_dir}")

    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

    if len(class_dirs) != 2:
        raise ValueError(f"Expected 2 class directories, found {len(class_dirs)}: {[d.name for d in class_dirs]}")

    # Determine which is violent (class 1) and which is non-violent (class 0)
    violent_keywords = ['fight', 'violent', 'violence', 'aggression']
    normal_keywords = ['normal', 'nonviolent', 'nonfight', 'non-violent', 'non-fight']

    class_0_dir = None
    class_1_dir = None

    for class_dir in class_dirs:
        class_name_lower = class_dir.name.lower()

        if any(keyword in class_name_lower for keyword in violent_keywords):
            class_1_dir = class_dir.name  # Violent class
        elif any(keyword in class_name_lower for keyword in normal_keywords):
            class_0_dir = class_dir.name  # Non-violent class

    # Fallback: alphabetical order if keywords don't match
    if class_0_dir is None or class_1_dir is None:
        sorted_dirs = sorted([d.name for d in class_dirs])
        class_0_dir = sorted_dirs[0]
        class_1_dir = sorted_dirs[1]
        logger.warning(f"Could not auto-detect class types. Using alphabetical order:")

    logger.info(f"✅ Class Detection:")
    logger.info(f"   Class 0 (Non-Violent): {class_0_dir}")
    logger.info(f"   Class 1 (Violent): {class_1_dir}")

    # Load data for all splits
    data = {}
    for split in ['train', 'val', 'test']:
        video_paths = []
        labels = []

        split_dir = dataset_path / split

        if not split_dir.exists():
            logger.warning(f"⚠️  Split directory not found: {split_dir}")
            data[split] = {
                'paths': [],
                'labels': np.array([])
            }
            continue

        # Load class 0 (non-violent)
        class_0_path = split_dir / class_0_dir
        if class_0_path.exists():
            for video in class_0_path.glob('*.mp4'):
                video_paths.append(str(video))
                labels.append(0)

        # Load class 1 (violent)
        class_1_path = split_dir / class_1_dir
        if class_1_path.exists():
            for video in class_1_path.glob('*.mp4'):
                video_paths.append(str(video))
                labels.append(1)

        data[split] = {
            'paths': video_paths,
            'labels': np.array(labels)
        }

        logger.info(f"   {split}: {len(video_paths)} videos ({np.sum(np.array(labels)==0)} normal, {np.sum(np.array(labels)==1)} violent)")

    return data


def print_dataset_summary(dataset: Dict):
    """Print dataset statistics"""
    print("\n" + "=" * 80)
    print("📊 DATASET SUMMARY")
    print("=" * 80)

    for split in ['train', 'val', 'test']:
        if split in dataset:
            paths = dataset[split]['paths']
            labels = dataset[split]['labels']

            if len(paths) > 0:
                class_0_count = np.sum(labels == 0)
                class_1_count = np.sum(labels == 1)

                print(f"\n{split.upper()}:")
                print(f"  Total: {len(paths)} videos")
                print(f"  Class 0 (Normal): {class_0_count} videos ({class_0_count/len(paths)*100:.1f}%)")
                print(f"  Class 1 (Violent): {class_1_count} videos ({class_1_count/len(paths)*100:.1f}%)")

                # Show sample paths
                if len(paths) > 0:
                    print(f"  Sample: {Path(paths[0]).parent.name}/{Path(paths[0]).name}")
            else:
                print(f"\n{split.upper()}: Empty or not found")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "/workspace/data"

    print(f"Testing flexible dataset loader with: {dataset_path}\n")

    try:
        dataset = load_dataset_flexible(dataset_path)
        print_dataset_summary(dataset)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nUsage: python3 flexible_dataset_loader.py /path/to/dataset")
