#!/usr/bin/env python3
"""
Ultimate Dataset Combiner - All Sources
Combines Phase 1 (Kaggle + Academic + Archive) + Phase 3 (Kinetics-700)
Target: 95,000+ videos after combination
"""

import os
import sys
import shutil
import json
from pathlib import Path
from collections import defaultdict
import hashlib
from tqdm import tqdm

# Paths
PHASE1_DIR = Path("/workspace/datasets/phase1")
PHASE3_DIR = Path("/workspace/datasets/phase3/kinetics")
OUTPUT_DIR = Path("/workspace/data/combined_ultimate")

# Labels for violence detection
VIOLENCE_LABELS = {
    'fight': 1,
    'violence': 1,
    'boxing': 1,
    'wrestling': 1,
    'martial_arts': 1,
    'kickboxing': 1,
    'punching': 1,
    'kicking': 1,
    'assault': 1,
    'aggression': 1,

    'non_violence': 0,
    'normal': 0,
    'neutral': 0,
}

def get_file_hash(filepath, sample_size=8192):
    """Fast hash using first and last chunks"""
    hasher = hashlib.md5()

    try:
        with open(filepath, 'rb') as f:
            # Hash first chunk
            hasher.update(f.read(sample_size))

            # Hash last chunk
            f.seek(-sample_size, 2)
            hasher.update(f.read(sample_size))

            return hasher.hexdigest()
    except:
        return None

def classify_video(filepath, parent_dirs):
    """Classify video as fight (1) or non-fight (0)"""
    filepath_lower = str(filepath).lower()
    parent_dirs_lower = [d.lower() for d in parent_dirs]

    # Check for violence indicators in path
    violence_keywords = [
        'fight', 'violence', 'boxing', 'wrestling', 'martial', 'kick',
        'punch', 'assault', 'aggression', 'combat', 'mma', 'ufc',
        'attack', 'hitting', 'striking'
    ]

    non_violence_keywords = [
        'non', 'normal', 'neutral', 'peaceful', 'background'
    ]

    # Check parent directories and filename
    path_text = filepath_lower + ' '.join(parent_dirs_lower)

    # Check for non-violence first
    if any(keyword in path_text for keyword in non_violence_keywords):
        return 0

    # Check for violence
    if any(keyword in path_text for keyword in violence_keywords):
        return 1

    # Default: If from Kinetics or academic datasets, assume violence-related
    if 'kinetics' in path_text or 'ucf' in path_text or 'xd-violence' in path_text:
        return 1

    # Default to violence for combat sports
    return 1

def find_all_videos(base_dir):
    """Find all video files recursively"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm'}
    videos = []

    print(f"\n🔍 Scanning {base_dir}...")

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                filepath = Path(root) / file
                videos.append(filepath)

    return videos

def process_dataset(name, source_dir, output_base, video_mapping, seen_hashes):
    """Process a single dataset directory"""
    print(f"\n{'='*80}")
    print(f"Processing: {name}")
    print(f"{'='*80}")

    videos = find_all_videos(source_dir)
    print(f"Found {len(videos)} video files")

    if not videos:
        print("⚠️  No videos found, skipping")
        return 0, 0

    fight_count = 0
    non_fight_count = 0
    duplicate_count = 0

    for video in tqdm(videos, desc=f"Processing {name}"):
        # Get file hash for deduplication
        file_hash = get_file_hash(video)
        if file_hash and file_hash in seen_hashes:
            duplicate_count += 1
            continue

        if file_hash:
            seen_hashes.add(file_hash)

        # Classify video
        parent_dirs = [p.name for p in video.parents]
        label = classify_video(video, parent_dirs)

        # Determine output directory
        if label == 1:
            output_dir = output_base / 'Fight'
            fight_count += 1
        else:
            output_dir = output_base / 'NonFight'
            non_fight_count += 1

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename
        dataset_prefix = name.replace(' ', '_').replace('-', '_').lower()
        new_filename = f"{dataset_prefix}_{video.stem}{video.suffix}"
        output_path = output_dir / new_filename

        # Handle filename conflicts
        counter = 1
        while output_path.exists():
            new_filename = f"{dataset_prefix}_{video.stem}_{counter}{video.suffix}"
            output_path = output_dir / new_filename
            counter += 1

        # Copy video
        try:
            shutil.copy2(video, output_path)

            # Track in mapping
            video_mapping.append({
                'original_path': str(video),
                'new_path': str(output_path),
                'dataset': name,
                'label': label,
                'hash': file_hash
            })
        except Exception as e:
            print(f"❌ Error copying {video}: {e}")

    print(f"✅ {name} complete:")
    print(f"   Fight videos: {fight_count}")
    print(f"   Non-fight videos: {non_fight_count}")
    print(f"   Duplicates skipped: {duplicate_count}")

    return fight_count, non_fight_count

def main():
    print("="*80)
    print("ULTIMATE DATASET COMBINER")
    print("Combining Phase 1 + Phase 3 datasets")
    print("="*80)

    # Create output directories
    train_dir = OUTPUT_DIR / 'train'
    val_dir = OUTPUT_DIR / 'val'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    video_mapping = []
    seen_hashes = set()

    total_fight = 0
    total_non_fight = 0

    # Process Phase 1 datasets
    print("\n" + "="*80)
    print("PHASE 1: KAGGLE + ACADEMIC + ARCHIVE")
    print("="*80)

    if PHASE1_DIR.exists():
        # Process Kaggle datasets
        kaggle_dir = PHASE1_DIR / 'kaggle'
        if kaggle_dir.exists():
            for dataset_dir in kaggle_dir.iterdir():
                if dataset_dir.is_dir():
                    f, nf = process_dataset(
                        f"Kaggle-{dataset_dir.name}",
                        dataset_dir,
                        train_dir,
                        video_mapping,
                        seen_hashes
                    )
                    total_fight += f
                    total_non_fight += nf

        # Process Academic datasets
        academic_dir = PHASE1_DIR / 'academic'
        if academic_dir.exists():
            for dataset_dir in academic_dir.iterdir():
                if dataset_dir.is_dir():
                    f, nf = process_dataset(
                        f"Academic-{dataset_dir.name}",
                        dataset_dir,
                        train_dir,
                        video_mapping,
                        seen_hashes
                    )
                    total_fight += f
                    total_non_fight += nf

        # Process Internet Archive
        archive_dir = PHASE1_DIR / 'archive'
        if archive_dir.exists():
            for dataset_dir in archive_dir.iterdir():
                if dataset_dir.is_dir():
                    f, nf = process_dataset(
                        f"Archive-{dataset_dir.name}",
                        dataset_dir,
                        train_dir,
                        video_mapping,
                        seen_hashes
                    )
                    total_fight += f
                    total_non_fight += nf
    else:
        print("⚠️  Phase 1 directory not found, skipping")

    # Process Phase 3: Kinetics-700
    print("\n" + "="*80)
    print("PHASE 3: KINETICS-700")
    print("="*80)

    if PHASE3_DIR.exists():
        f, nf = process_dataset(
            "Kinetics-700",
            PHASE3_DIR,
            train_dir,
            video_mapping,
            seen_hashes
        )
        total_fight += f
        total_non_fight += nf
    else:
        print("⚠️  Phase 3 (Kinetics) directory not found")
        print("Run download_phase3_kinetics.sh first")

    # Create validation split (10% of data)
    print("\n" + "="*80)
    print("CREATING VALIDATION SPLIT (10%)")
    print("="*80)

    import random
    random.seed(42)

    train_fight_dir = train_dir / 'Fight'
    train_nonfight_dir = train_dir / 'NonFight'
    val_fight_dir = val_dir / 'Fight'
    val_nonfight_dir = val_dir / 'NonFight'

    val_fight_dir.mkdir(parents=True, exist_ok=True)
    val_nonfight_dir.mkdir(parents=True, exist_ok=True)

    # Move 10% to validation
    for class_dir, val_class_dir in [(train_fight_dir, val_fight_dir),
                                      (train_nonfight_dir, val_nonfight_dir)]:
        if class_dir.exists():
            videos = list(class_dir.glob('*.*'))
            val_count = int(len(videos) * 0.1)
            val_videos = random.sample(videos, val_count)

            print(f"Moving {val_count} videos from {class_dir.name} to validation...")
            for video in tqdm(val_videos):
                shutil.move(str(video), str(val_class_dir / video.name))

    # Save video mapping
    mapping_file = OUTPUT_DIR / 'video_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(video_mapping, f, indent=2)

    # Final statistics
    print("\n" + "="*80)
    print("COMBINATION COMPLETE!")
    print("="*80)

    train_fight_count = len(list((train_dir / 'Fight').glob('*.*'))) if (train_dir / 'Fight').exists() else 0
    train_nonfight_count = len(list((train_dir / 'NonFight').glob('*.*'))) if (train_dir / 'NonFight').exists() else 0
    val_fight_count = len(list((val_dir / 'Fight').glob('*.*'))) if (val_dir / 'Fight').exists() else 0
    val_nonfight_count = len(list((val_dir / 'NonFight').glob('*.*'))) if (val_dir / 'NonFight').exists() else 0

    total = train_fight_count + train_nonfight_count + val_fight_count + val_nonfight_count

    print(f"\n📊 FINAL STATISTICS:")
    print(f"{'='*80}")
    print(f"Train set:")
    print(f"  Fight videos:     {train_fight_count:,}")
    print(f"  Non-fight videos: {train_nonfight_count:,}")
    print(f"  Total train:      {train_fight_count + train_nonfight_count:,}")
    print(f"")
    print(f"Validation set:")
    print(f"  Fight videos:     {val_fight_count:,}")
    print(f"  Non-fight videos: {val_nonfight_count:,}")
    print(f"  Total val:        {val_fight_count + val_nonfight_count:,}")
    print(f"")
    print(f"{'='*80}")
    print(f"TOTAL VIDEOS:       {total:,}")
    print(f"Duplicates removed: {len([m for m in video_mapping if 'duplicate' in str(m).lower()])}")
    print(f"{'='*80}")

    if total >= 95000:
        print("🎉 EXCELLENT: 95,000+ videos achieved!")
    elif total >= 80000:
        print("✅ GREAT: 80,000+ videos achieved!")
    elif total >= 50000:
        print("✅ GOOD: 50,000+ videos achieved!")
    else:
        print(f"⚠️  {total:,} videos (may need Phase 4 for 100K+ target)")

    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"📋 Video mapping saved to: {mapping_file}")
    print(f"\n🔄 NEXT STEP:")
    print(f"python /home/admin/Desktop/NexaraVision/runpod_train_ultimate.py")

if __name__ == "__main__":
    main()
