#!/usr/bin/env python3
"""
Automatic Violent/Non-Violent Separation
Separates already downloaded datasets into violent and non-violent categories
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Keywords for violent content
VIOLENT_KEYWORDS = [
    'fight', 'violence', 'violent', 'assault', 'punch', 'kick', 'hit',
    'boxing', 'wrestl', 'combat', 'attack', 'aggression', 'brawl',
    'abuse', 'robbery', 'burglary', 'vandalism', 'arson', 'explosion',
    'shooting', 'stabbing', 'riot', 'anomaly'
]

# Keywords for non-violent content
NONVIOLENT_KEYWORDS = [
    'normal', 'nonviolent', 'non-violent', 'non_violent', 'daily',
    'walk', 'sit', 'stand', 'talk', 'eat', 'drink', 'read', 'write',
    'cook', 'clean', 'shop', 'play', 'dance', 'sing', 'exercise'
]

def classify_video(filepath, parent_folder_name):
    """Classify video as violent (1) or non-violent (0)"""

    # Get full path for context
    full_path = str(filepath).lower()
    folder_name = parent_folder_name.lower()
    filename = filepath.name.lower()

    # Check folder name first (most reliable)
    for keyword in VIOLENT_KEYWORDS:
        if keyword in folder_name or keyword in filename:
            return 1  # Violent

    for keyword in NONVIOLENT_KEYWORDS:
        if keyword in folder_name or keyword in filename:
            return 0  # Non-violent

    # Default: if from known violent dataset but no explicit label, assume violent
    return 1

def separate_datasets(source_dir, violent_out, nonviolent_out):
    """Separate videos into violent and non-violent"""

    source_dir = Path(source_dir)
    violent_out = Path(violent_out)
    nonviolent_out = Path(nonviolent_out)

    violent_out.mkdir(parents=True, exist_ok=True)
    nonviolent_out.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SEPARATING VIOLENT AND NON-VIOLENT VIDEOS")
    print("="*80)
    print(f"Source: {source_dir}")
    print(f"Violent output: {violent_out}")
    print(f"Non-violent output: {nonviolent_out}")
    print("")

    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm'}

    print("🔍 Scanning for videos...")
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(source_dir.rglob(f'*{ext}'))

    print(f"Found {len(all_videos)} videos")
    print("")

    # Separate videos
    violent_count = 0
    nonviolent_count = 0

    print("📂 Separating videos...")
    for video_path in tqdm(all_videos, desc="Processing"):
        # Get parent folder name for context
        parent_folder = video_path.parent.name

        # Classify
        is_violent = classify_video(video_path, parent_folder)

        # Determine output directory
        if is_violent:
            out_dir = violent_out / parent_folder
            violent_count += 1
        else:
            out_dir = nonviolent_out / parent_folder
            nonviolent_count += 1

        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy video (don't delete original yet)
        dest_path = out_dir / video_path.name

        if not dest_path.exists():
            try:
                shutil.copy2(video_path, dest_path)
            except Exception as e:
                print(f"Error copying {video_path}: {e}")

    print("")
    print("="*80)
    print("SEPARATION COMPLETE!")
    print("="*80)
    print(f"✅ Violent videos: {violent_count}")
    print(f"✅ Non-violent videos: {nonviolent_count}")
    print(f"✅ Total processed: {violent_count + nonviolent_count}")
    print("")
    print(f"📁 Violent location: {violent_out}")
    print(f"📁 Non-violent location: {nonviolent_out}")
    print("")

    return violent_count, nonviolent_count

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Separate Violent/Non-Violent Videos')
    parser.add_argument('--source', required=True,
                       help='Source directory with mixed videos')
    parser.add_argument('--violent-out', default='/workspace/datasets/separated/violent',
                       help='Output directory for violent videos')
    parser.add_argument('--nonviolent-out', default='/workspace/datasets/separated/nonviolent',
                       help='Output directory for non-violent videos')

    args = parser.parse_args()

    separate_datasets(args.source, args.violent_out, args.nonviolent_out)

if __name__ == "__main__":
    main()
