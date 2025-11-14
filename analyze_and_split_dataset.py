#!/usr/bin/env python3
"""
Complete Dataset Analysis and Train/Val/Test Split Script
Analyzes all videos, categorizes as violent/non-violent, and creates proper splits.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory (adjust for Vast.ai or local)
BASE_DIR = "/workspace/datasets"  # Change to your actual path

# Output directory
OUTPUT_DIR = "/workspace/organized_dataset"

# Split ratios
TRAIN_RATIO = 0.70  # 70%
VAL_RATIO = 0.15    # 15%
TEST_RATIO = 0.15   # 15%

# Random seed for reproducibility
RANDOM_SEED = 42

# Video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']

# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================

# Violent categories (based on your folder names)
VIOLENT_CATEGORIES = {
    # Reddit subreddits with violence
    'r_fights': 'violent',
    'r_Justiceserved': 'violent',  # Often contains violence
    'r_StreetFighting': 'violent',
    'r_RealFights': 'violent',
    'r_BrutalFights': 'violent',
    'r_femalemma': 'violent',  # MMA is violent
    'r_ActualPublicFreakouts': 'violent',  # Often contains violence
    'r_fightclub': 'violent',
    'r_instantkarma': 'mixed',  # Some violent, some not
    'r_instant_regret': 'mixed',
    'r_NoahGetTheBoat': 'mixed',
    'r_iamatotalpieceofshit': 'mixed',

    # YouTube fight videos
    'UFC_fight_highlights': 'violent',
    'UFC_fight_highlights_2023': 'violent',
    'MMA_knockouts_compilation': 'violent',
    'Boxing_match_highlights': 'violent',
    'Street_fight_caught_on_camera': 'violent',
    'Wrestling_highlights': 'violent',
    'Martial_arts_demonstration': 'violent',  # Often contains sparring/fighting
    'Kickboxing_highlights': 'violent',
    'Muay_Thai_fight': 'violent',
    'Karate_tournament': 'violent',
    'Judo_competition': 'violent',
    'youtube_fights': 'violent',

    # Non-violent categories
    'cctv_surveillance': 'nonviolent',
    'nonviolent': 'nonviolent',
    'nonviolent_kaggle': 'nonviolent',
    'nonviolent_safe': 'nonviolent',
    'phase1': 'unknown',  # Need to check
    'phase3': 'unknown',  # Need to check
    'reddit_videos': 'mixed',  # Need to check
    'reddit_videos_pushshift': 'mixed',
}


def get_category(folder_name):
    """Determine if folder contains violent or non-violent videos."""
    folder_lower = folder_name.lower()

    for key, category in VIOLENT_CATEGORIES.items():
        if key.lower() in folder_lower:
            return category

    return 'unknown'


def analyze_dataset(base_dir):
    """
    Analyze all videos in dataset and categorize them.
    Returns detailed statistics.
    """
    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    print(f"\nScanning: {base_dir}\n")

    stats = {
        'violent': defaultdict(int),
        'nonviolent': defaultdict(int),
        'mixed': defaultdict(int),
        'unknown': defaultdict(int),
    }

    all_videos = {
        'violent': [],
        'nonviolent': [],
        'mixed': [],
        'unknown': [],
    }

    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"❌ ERROR: Directory not found: {base_dir}")
        return None, None

    # Scan all directories
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)

        # Get videos in this directory
        videos = [f for f in files if Path(f).suffix.lower() in VIDEO_EXTENSIONS]

        if not videos:
            continue

        # Determine category
        relative_path = root_path.relative_to(base_path)
        folder_name = str(relative_path).split('/')[0] if str(relative_path) != '.' else root_path.name

        category = get_category(folder_name)

        # Count videos
        video_count = len(videos)
        stats[category][folder_name] += video_count

        # Store video paths
        for video in videos:
            video_path = root_path / video
            all_videos[category].append(str(video_path))

    # Print statistics
    print("\n📊 DATASET STATISTICS BY CATEGORY:\n")

    for category in ['violent', 'nonviolent', 'mixed', 'unknown']:
        if stats[category]:
            total = sum(stats[category].values())
            print(f"\n{'='*80}")
            print(f"🏷️  {category.upper()} ({total:,} videos)")
            print(f"{'='*80}")

            for folder, count in sorted(stats[category].items(), key=lambda x: -x[1]):
                print(f"  📂 {folder:40s} → {count:6,} videos")

    # Overall summary
    print(f"\n{'='*80}")
    print("📈 OVERALL SUMMARY")
    print(f"{'='*80}")

    violent_total = sum(sum(stats['violent'].values()) for _ in [1])
    nonviolent_total = sum(sum(stats['nonviolent'].values()) for _ in [1])
    mixed_total = sum(sum(stats['mixed'].values()) for _ in [1])
    unknown_total = sum(sum(stats['unknown'].values()) for _ in [1])

    total_videos = violent_total + nonviolent_total + mixed_total + unknown_total

    print(f"  ✅ Violent videos:     {violent_total:8,} ({violent_total/total_videos*100:5.1f}%)")
    print(f"  ✅ Non-violent videos: {nonviolent_total:8,} ({nonviolent_total/total_videos*100:5.1f}%)")
    print(f"  ⚠️  Mixed videos:       {mixed_total:8,} ({mixed_total/total_videos*100:5.1f}%)")
    print(f"  ❓ Unknown videos:     {unknown_total:8,} ({unknown_total/total_videos*100:5.1f}%)")
    print(f"  {'─'*40}")
    print(f"  📊 TOTAL:              {total_videos:8,}")
    print()

    return stats, all_videos


def create_splits(all_videos, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits from categorized videos.
    """
    print("\n" + "="*80)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("="*80)

    random.seed(RANDOM_SEED)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits_info = {}

    for category in ['violent', 'nonviolent']:
        if category not in all_videos or not all_videos[category]:
            print(f"\n⚠️  No videos in '{category}' category, skipping...")
            continue

        videos = all_videos[category].copy()
        random.shuffle(videos)

        total = len(videos)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_videos = videos[:train_size]
        val_videos = videos[train_size:train_size + val_size]
        test_videos = videos[train_size + val_size:]

        print(f"\n📂 {category.upper()}:")
        print(f"   Total: {total:,} videos")
        print(f"   Train: {len(train_videos):,} ({len(train_videos)/total*100:.1f}%)")
        print(f"   Val:   {len(val_videos):,} ({len(val_videos)/total*100:.1f}%)")
        print(f"   Test:  {len(test_videos):,} ({len(test_videos)/total*100:.1f}%)")

        splits_info[category] = {
            'total': total,
            'train': len(train_videos),
            'val': len(val_videos),
            'test': len(test_videos),
        }

        # Create directory structure
        for split_name, split_videos in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
            split_dir = output_path / split_name / ('Violence' if category == 'violent' else 'NonViolence')
            split_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n   Copying {len(split_videos):,} videos to {split_name}/{category}...")

            for i, video_path in enumerate(split_videos):
                if i % 100 == 0 and i > 0:
                    print(f"      {i}/{len(split_videos)}...", end='\r')

                src = Path(video_path)
                if not src.exists():
                    continue

                # Generate unique filename
                dst_name = f"{category}_{src.stem}_{i:05d}{src.suffix}"
                dst = split_dir / dst_name

                # Copy file
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"\n      ⚠️  Error copying {src.name}: {e}")

            print(f"      ✅ Done: {len(split_videos):,} videos copied")

    # Save split info
    info_file = output_path / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'splits_info': splits_info,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'random_seed': RANDOM_SEED,
        }, f, indent=2)

    print(f"\n✅ Dataset info saved to: {info_file}")

    return splits_info


def print_final_summary(output_dir):
    """Print final directory structure summary."""
    print("\n" + "="*80)
    print("FINAL DATASET STRUCTURE")
    print("="*80)

    output_path = Path(output_dir)

    for split in ['train', 'val', 'test']:
        split_dir = output_path / split
        if not split_dir.exists():
            continue

        print(f"\n📁 {split}/")

        for category in ['Violence', 'NonViolence']:
            cat_dir = split_dir / category
            if cat_dir.exists():
                count = len(list(cat_dir.glob('*.*')))
                print(f"   └─ {category:15s} → {count:6,} videos")

    print(f"\n✅ Organized dataset location: {output_dir}")
    print("\n📝 Ready for training!")


def main():
    """Main execution function."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     VIOLENCE DETECTION DATASET ORGANIZER                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Analyze dataset
    stats, all_videos = analyze_dataset(BASE_DIR)

    if stats is None:
        print("\n❌ Analysis failed. Please check the BASE_DIR path.")
        return

    # Step 2: Handle mixed/unknown categories
    if all_videos['mixed'] or all_videos['unknown']:
        print("\n" + "="*80)
        print("⚠️  MIXED/UNKNOWN CATEGORIES DETECTED")
        print("="*80)
        print("\nSome folders contain mixed or unknown content.")
        print("These will be EXCLUDED from the organized dataset.")
        print("\nYou can manually review and categorize them later.")

        mixed_count = len(all_videos['mixed'])
        unknown_count = len(all_videos['unknown'])

        if mixed_count:
            print(f"\n  Mixed videos: {mixed_count:,}")
        if unknown_count:
            print(f"  Unknown videos: {unknown_count:,}")

    # Step 3: Check if we have enough data
    violent_count = len(all_videos['violent'])
    nonviolent_count = len(all_videos['nonviolent'])

    if violent_count == 0 or nonviolent_count == 0:
        print("\n❌ ERROR: Need both violent and non-violent videos!")
        print(f"   Violent: {violent_count}")
        print(f"   Non-violent: {nonviolent_count}")
        return

    # Step 4: Check class balance
    ratio = min(violent_count, nonviolent_count) / max(violent_count, nonviolent_count)

    print(f"\n📊 Class Balance Analysis:")
    print(f"   Violent:     {violent_count:8,}")
    print(f"   Non-violent: {nonviolent_count:8,}")
    print(f"   Balance:     {ratio*100:5.1f}% (1:{1/ratio:.1f})")

    if ratio < 0.5:
        print(f"\n⚠️  WARNING: Significant class imbalance!")
        print(f"   Consider balancing by:")
        print(f"   1. Using only {min(violent_count, nonviolent_count)*2:,} videos (balanced)")
        print(f"   2. Applying class weights during training")
        print(f"   3. Using data augmentation on minority class")

    # Step 5: Ask user to proceed
    print(f"\n{'='*80}")
    response = input("\nProceed with creating train/val/test splits? (y/n): ")

    if response.lower() != 'y':
        print("\n❌ Operation cancelled.")
        print("\nTo run again, adjust BASE_DIR and OUTPUT_DIR in the script.")
        return

    # Step 6: Create splits
    splits_info = create_splits(
        all_videos,
        OUTPUT_DIR,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO
    )

    # Step 7: Print final summary
    print_final_summary(OUTPUT_DIR)

    print("\n" + "="*80)
    print("✅ DATASET ORGANIZATION COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Verify the splits look correct")
    print(f"  2. Upload to your training platform (RunPod/Vast.ai)")
    print(f"  3. Start training with:")
    print(f"     python runpod_train_l40s.py --dataset-path {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
