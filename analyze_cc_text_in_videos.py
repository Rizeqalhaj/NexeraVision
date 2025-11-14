#!/usr/bin/env python3
"""
Analyze if CC text/white overlays differ between violence and non-violence videos
Uses edge detection to find text-like regions
"""

import cv2
import numpy as np
from pathlib import Path
import random

VIOLENCE_DIRS = [
    "/workspace/violence_videos",
    "/workspace/fight_videos",
    "/workspace/rwf2000/train/Fight",
]

NONVIOLENCE_DIRS = [
    "/workspace/youtube_nonviolence_videos",
    "/workspace/ucf_crime_dropbox/normal_videos",
    "/workspace/rwf2000/train/NonFight",
]

SAMPLE_SIZE = 50  # Sample 50 videos from each category
FRAMES_TO_CHECK = 10  # Check 10 random frames per video

print("=" * 80)
print("CC TEXT ANALYSIS - Violence vs Non-Violence")
print("=" * 80)
print()

def has_text_overlay(frame):
    """
    Detect if frame has white text overlay (like CC)
    Returns: (has_text, text_percentage)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect white regions (CC text is usually white)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Focus on bottom 25% of frame (where CC usually appears)
    h, w = frame.shape[:2]
    bottom_region = white_mask[int(h * 0.75):, :]

    # Calculate white pixel percentage in bottom region
    white_pixels = np.sum(bottom_region == 255)
    total_pixels = bottom_region.size
    white_percentage = (white_pixels / total_pixels) * 100

    # If >5% of bottom region is white, likely has text
    has_text = white_percentage > 5.0

    return has_text, white_percentage

def analyze_video(video_path):
    """
    Analyze single video for text overlays
    Returns: average text percentage
    """
    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            return None

        # Sample random frames
        frame_indices = random.sample(range(total_frames), min(FRAMES_TO_CHECK, total_frames))

        text_percentages = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            has_text, text_pct = has_text_overlay(frame)
            text_percentages.append(text_pct)

        cap.release()

        return np.mean(text_percentages) if text_percentages else None

    except Exception as e:
        return None

def analyze_category(dirs, category_name):
    """Analyze all videos in category directories"""
    print(f"Analyzing {category_name} videos...")

    # Collect all video files
    all_videos = []
    for dir_path in dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            all_videos.extend(list(dir_path.rglob("*.mp4")))
            all_videos.extend(list(dir_path.rglob("*.avi")))

    if not all_videos:
        print(f"  ⚠️  No videos found in {category_name} directories")
        return None

    # Sample videos
    sample_videos = random.sample(all_videos, min(SAMPLE_SIZE, len(all_videos)))

    print(f"  Found {len(all_videos)} videos, sampling {len(sample_videos)}")

    text_percentages = []
    videos_with_text = 0

    for idx, video in enumerate(sample_videos, 1):
        if idx % 10 == 0:
            print(f"    Progress: {idx}/{len(sample_videos)}")

        text_pct = analyze_video(video)

        if text_pct is not None:
            text_percentages.append(text_pct)
            if text_pct > 5.0:
                videos_with_text += 1

    if not text_percentages:
        return None

    avg_text_pct = np.mean(text_percentages)
    text_video_ratio = (videos_with_text / len(text_percentages)) * 100

    print(f"  ✓ Average text overlay: {avg_text_pct:.2f}%")
    print(f"  ✓ Videos with text: {videos_with_text}/{len(text_percentages)} ({text_video_ratio:.1f}%)")
    print()

    return {
        'avg_text_pct': avg_text_pct,
        'videos_with_text': videos_with_text,
        'total_analyzed': len(text_percentages),
        'text_video_ratio': text_video_ratio
    }

# Analyze both categories
print()
violence_stats = analyze_category(VIOLENCE_DIRS, "VIOLENCE")
nonviolence_stats = analyze_category(NONVIOLENCE_DIRS, "NON-VIOLENCE")

print("=" * 80)
print("ANALYSIS RESULTS")
print("=" * 80)
print()

if violence_stats and nonviolence_stats:
    print(f"VIOLENCE:")
    print(f"  Average text overlay: {violence_stats['avg_text_pct']:.2f}%")
    print(f"  Videos with text: {violence_stats['text_video_ratio']:.1f}%")
    print()

    print(f"NON-VIOLENCE:")
    print(f"  Average text overlay: {nonviolence_stats['avg_text_pct']:.2f}%")
    print(f"  Videos with text: {nonviolence_stats['text_video_ratio']:.1f}%")
    print()

    # Calculate difference
    text_diff = abs(violence_stats['text_video_ratio'] - nonviolence_stats['text_video_ratio'])
    avg_diff = abs(violence_stats['avg_text_pct'] - nonviolence_stats['avg_text_pct'])

    print("=" * 80)
    print("IMPACT ASSESSMENT")
    print("=" * 80)
    print()

    print(f"Difference in text overlay: {avg_diff:.2f}%")
    print(f"Difference in videos with text: {text_diff:.1f}%")
    print()

    if text_diff < 10:
        print("✅ LOW RISK: Text overlay distribution is similar between categories")
        print("   The model is unlikely to learn spurious text-based patterns")
    elif text_diff < 25:
        print("⚠️  MODERATE RISK: Some difference in text overlay distribution")
        print("   Recommendation: Monitor model performance, consider data augmentation")
    else:
        print("🚨 HIGH RISK: Significant difference in text overlay distribution")
        print("   Recommendation: Either remove CC text or add text augmentation during training")

    print()
    print("RECOMMENDATIONS:")
    print()

    if text_diff >= 10:
        print("1. DATA AUGMENTATION (Recommended):")
        print("   - Add random text overlays during training to both categories")
        print("   - This makes the model robust to text presence/absence")
        print()
        print("2. PREPROCESSING (If text is severe):")
        print("   - Use inpainting to remove text regions before training")
        print("   - Focus model on motion patterns, not static overlays")
        print()

    print("3. TRAINING STRATEGIES:")
    print("   - Use 3D CNNs or LSTM to focus on temporal motion patterns")
    print("   - Text is static, violence is motion-based")
    print("   - Your current architecture should handle this well")
    print()

else:
    print("❌ Unable to complete analysis - check if video directories exist")

print()
