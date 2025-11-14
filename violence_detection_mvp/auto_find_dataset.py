#!/usr/bin/env python3
"""
Auto-detect dataset path on Vast.ai and update training config
"""

import os
from pathlib import Path


def find_dataset_path(start_path="/workspace"):
    """
    Find dataset directory by looking for train/val/test structure
    """
    print(f"🔍 Searching for dataset in {start_path}...")

    # Look for common dataset patterns
    patterns = [
        # Pattern 1: train/val/test with Fight/NonFight or Fight/Normal
        lambda p: (p / "train" / "Fight").exists() or (p / "train" / "NonFight").exists(),
        # Pattern 2: train/val/test with violent/nonviolent
        lambda p: (p / "train" / "violent").exists() or (p / "train" / "nonviolent").exists(),
    ]

    # Search recursively
    for root, dirs, files in os.walk(start_path):
        # Skip cache/checkpoint/model directories
        if any(skip in root for skip in ['cache', 'checkpoint', 'model', '.git']):
            continue

        current = Path(root)

        # Check if this looks like a dataset root
        for pattern in patterns:
            if pattern(current):
                print(f"✅ Found dataset at: {current}")
                return str(current)

    # Fallback: look for any directory with MP4 files
    for root, dirs, files in os.walk(start_path):
        if any(skip in root for skip in ['cache', 'checkpoint', 'model']):
            continue

        mp4_files = [f for f in files if f.endswith('.mp4')]
        if len(mp4_files) > 10:  # At least 10 videos
            parent = Path(root).parent
            print(f"✅ Found videos at: {parent}")
            return str(parent)

    return None


def detect_class_names(dataset_path):
    """
    Detect class names from directory structure
    Returns tuple: (class_0_name, class_1_name)
    """
    dataset_path = Path(dataset_path)

    # Check train directory
    train_dir = dataset_path / "train"
    if not train_dir.exists():
        return None, None

    subdirs = [d.name for d in train_dir.iterdir() if d.is_dir()]

    # Common patterns
    if "Fight" in subdirs and "NonFight" in subdirs:
        return "NonFight", "Fight"
    elif "Fight" in subdirs and "Normal" in subdirs:
        return "Normal", "Fight"
    elif "violent" in subdirs and "nonviolent" in subdirs:
        return "nonviolent", "violent"
    elif "violence" in subdirs and "normal" in subdirs:
        return "normal", "violence"
    else:
        return subdirs[0] if len(subdirs) > 0 else None, subdirs[1] if len(subdirs) > 1 else None


def print_dataset_info(dataset_path):
    """Print dataset structure info"""
    dataset_path = Path(dataset_path)

    print("\n" + "=" * 80)
    print("📊 DATASET STRUCTURE")
    print("=" * 80)

    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            print(f"\n{split.upper()}:")
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    video_count = len(list(class_dir.glob('*.mp4')))
                    print(f"  ✅ {class_dir.name}: {video_count} videos")
        else:
            print(f"\n{split.upper()}: ❌ Not found")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("🔍 AUTO-DETECTING DATASET PATH")
    print("=" * 80 + "\n")

    # Find dataset
    dataset_path = find_dataset_path()

    if dataset_path is None:
        print("\n❌ Could not find dataset!")
        print("\nPlease manually check your dataset location:")
        print("  find /workspace -name '*.mp4' -type f | head -10")
        exit(1)

    # Detect class names
    class_0, class_1 = detect_class_names(dataset_path)

    # Print info
    print_dataset_info(dataset_path)

    print("\n" + "=" * 80)
    print("📝 CONFIGURATION")
    print("=" * 80)
    print(f"Dataset Path: {dataset_path}")
    print(f"Class 0 (Non-Violent): {class_0}")
    print(f"Class 1 (Violent): {class_1}")
    print("\n" + "=" * 80)

    print("\n💡 UPDATE YOUR TRAINING SCRIPT:")
    print(f"   dataset_path=\"{dataset_path}\"")
    print("\n" + "=" * 80)
