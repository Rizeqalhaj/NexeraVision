#!/usr/bin/env python3
"""
Create Sample Training Data for Violence Detection MVP
Generates synthetic videos for immediate model training and testing
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path

def create_violence_video(filename, duration=5, fps=30):
    """Create a sample violence video with aggressive movements"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))

    total_frames = fps * duration

    for i in range(total_frames):
        # Create base frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Red background (violence indicator)
        frame[:, :, 2] = 150 + int(50 * np.sin(i * 0.5))  # Pulsing red

        # Simulate aggressive movements
        # Moving objects with erratic patterns
        center_x = 320 + int(100 * np.sin(i * 0.3) + 50 * np.cos(i * 0.7))
        center_y = 240 + int(80 * np.cos(i * 0.4) + 40 * np.sin(i * 0.8))

        # Multiple moving objects (simulating people fighting)
        cv2.circle(frame, (center_x, center_y), 30, (0, 0, 255), -1)  # Person 1
        cv2.circle(frame, (640-center_x, 480-center_y), 25, (0, 100, 255), -1)  # Person 2

        # Rapid movements (punching motions)
        if i % 10 < 3:  # Fast movements every 10 frames
            cv2.rectangle(frame,
                         (center_x-20, center_y-20),
                         (center_x+20, center_y+20),
                         (255, 255, 255), -1)

        # Add random "impact" flashes
        if random.random() < 0.1:
            cv2.circle(frame, (random.randint(100, 540), random.randint(100, 380)),
                      random.randint(20, 50), (255, 255, 255), -1)

        # Violence text indicator
        cv2.putText(frame, "VIOLENCE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ Created violence video: {filename}")

def create_nonviolence_video(filename, duration=5, fps=30):
    """Create a sample non-violence video with calm movements"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))

    total_frames = fps * duration

    for i in range(total_frames):
        # Create base frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Blue/green background (calm indicator)
        frame[:, :, 0] = 100 + int(30 * np.sin(i * 0.1))  # Calm blue
        frame[:, :, 1] = 80 + int(20 * np.cos(i * 0.15))   # Gentle green

        # Simulate calm movements
        # Slow, predictable movements
        center_x = 320 + int(50 * np.sin(i * 0.05))
        center_y = 240 + int(30 * np.cos(i * 0.08))

        # People walking calmly
        cv2.circle(frame, (center_x, center_y), 25, (255, 200, 100), -1)  # Person 1
        cv2.circle(frame, (center_x + 100, center_y + 50), 20, (200, 255, 100), -1)  # Person 2

        # Gentle movements
        cv2.rectangle(frame, (200, 200), (440, 280), (100, 150, 200), 2)  # Building outline

        # Normal activity indicators
        cv2.putText(frame, "NORMAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ Created non-violence video: {filename}")

def create_training_dataset(num_videos_per_class=50):
    """Create a complete training dataset"""
    print("🎬 Creating Sample Training Dataset for Violence Detection")
    print("=" * 60)

    # Create directories
    dirs = [
        "data/raw/sample_training/train/Fight",
        "data/raw/sample_training/train/NonFight",
        "data/raw/sample_training/val/Fight",
        "data/raw/sample_training/val/NonFight"
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

    print(f"\n🎯 Creating {num_videos_per_class} videos per class...")

    # Training set (80%)
    train_count = int(num_videos_per_class * 0.8)
    val_count = num_videos_per_class - train_count

    print(f"📊 Train: {train_count} per class, Validation: {val_count} per class")

    # Create training violence videos
    print("\n🔴 Creating training violence videos...")
    for i in range(train_count):
        filename = f"data/raw/sample_training/train/Fight/fight_{i:03d}.mp4"
        create_violence_video(filename)

    # Create training non-violence videos
    print("\n🟢 Creating training non-violence videos...")
    for i in range(train_count):
        filename = f"data/raw/sample_training/train/NonFight/normal_{i:03d}.mp4"
        create_nonviolence_video(filename)

    # Create validation violence videos
    print("\n🔴 Creating validation violence videos...")
    for i in range(val_count):
        filename = f"data/raw/sample_training/val/Fight/fight_{i:03d}.mp4"
        create_violence_video(filename)

    # Create validation non-violence videos
    print("\n🟢 Creating validation non-violence videos...")
    for i in range(val_count):
        filename = f"data/raw/sample_training/val/NonFight/normal_{i:03d}.mp4"
        create_nonviolence_video(filename)

    # Summary
    total_videos = num_videos_per_class * 2
    print(f"\n📊 Dataset Creation Summary:")
    print(f"  Training: {train_count * 2} videos ({train_count} Fight + {train_count} NonFight)")
    print(f"  Validation: {val_count * 2} videos ({val_count} Fight + {val_count} NonFight)")
    print(f"  Total: {total_videos} videos")

    return total_videos

def create_dataset_info():
    """Create dataset information file"""
    info = """# Sample Violence Detection Dataset

## Dataset Information
- **Purpose**: Training and testing the Violence Detection MVP
- **Type**: Synthetic videos for immediate development
- **Structure**: RWF-2000 compatible format

## Directory Structure
```
data/raw/sample_training/
├── train/
│   ├── Fight/          # Violence videos
│   └── NonFight/       # Non-violence videos
└── val/
    ├── Fight/          # Violence validation videos
    └── NonFight/       # Non-violence validation videos
```

## Video Specifications
- **Duration**: 5 seconds per video
- **FPS**: 30 frames per second
- **Resolution**: 640x480 pixels
- **Format**: MP4 (H.264)

## Usage
```bash
# Train with sample dataset
python3 run.py train --data-dir data/raw/sample_training

# Evaluate with sample dataset
python3 run.py evaluate --data-dir data/raw/sample_training
```

## Characteristics
- **Violence videos**: Red-dominant, erratic movements, impact flashes
- **Non-violence videos**: Blue/green-dominant, calm movements, predictable patterns

This synthetic dataset allows immediate testing of the violence detection pipeline while real datasets are being downloaded.
"""

    with open("data/raw/sample_training/README.md", "w") as f:
        f.write(info)

    print("✅ Created dataset info: data/raw/sample_training/README.md")

def main():
    """Main function to create sample training data"""
    print("🚀 Sample Training Data Creator")
    print("Creating synthetic videos for immediate model training")
    print("=" * 60)

    # Create the dataset
    num_videos = 20  # Start with smaller dataset for quick testing
    total_created = create_training_dataset(num_videos)

    # Create info file
    create_dataset_info()

    print("\n" + "=" * 60)
    print("🎉 SUCCESS: Sample training dataset created!")
    print(f"📹 Total videos: {total_created}")
    print("\n🚀 Ready to start training:")
    print("   python3 run.py train --data-dir data/raw/sample_training")
    print("   python3 run.py info")

    print("\n💡 This synthetic dataset allows you to:")
    print("   ✅ Test the complete training pipeline")
    print("   ✅ Validate model architecture")
    print("   ✅ Verify preprocessing and feature extraction")
    print("   ✅ Debug any issues before using real data")

    print(f"\n📁 Dataset location: data/raw/sample_training/")

if __name__ == "__main__":
    main()