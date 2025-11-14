#!/usr/bin/env python3
"""
Combine ALL datasets into violent/nonviolent categories
Includes: phase1, youtube_fights, reddit_videos_massive, reddit_videos, nonviolent_safe
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base dataset directory
DATASETS_BASE = "/workspace/datasets"

# Source datasets
SOURCES = {
    'violent': [
        {
            'name': 'phase1_violent',
            'path': f'{DATASETS_BASE}/phase1',
            'need_separation': True,  # Needs to run separation first
            'use_path': f'{DATASETS_BASE}/phase1_categorized/violent'  # After separation
        },
        {
            'name': 'youtube_fights',
            'path': f'{DATASETS_BASE}/youtube_fights',
            'need_separation': False,
            'recursive': True  # Search subdirectories
        },
        {
            'name': 'reddit_videos_massive',
            'path': f'{DATASETS_BASE}/reddit_videos_massive',
            'need_separation': False,
            'recursive': True
        },
        {
            'name': 'reddit_videos',
            'path': f'{DATASETS_BASE}/reddit_videos',
            'need_separation': False,
            'recursive': True
        },
        {
            'name': 'reddit_videos_pushshift',
            'path': f'{DATASETS_BASE}/reddit_videos_pushshift',
            'need_separation': False,
            'recursive': True
        },
    ],
    'nonviolent': [
        {
            'name': 'phase1_nonviolent',
            'path': f'{DATASETS_BASE}/phase1',
            'need_separation': True,
            'use_path': f'{DATASETS_BASE}/phase1_categorized/nonviolent'  # After separation
        },
        {
            'name': 'nonviolent_safe',
            'path': f'{DATASETS_BASE}/nonviolent_safe',
            'need_separation': False,
            'recursive': True
        },
        {
            'name': 'nonviolent',
            'path': f'{DATASETS_BASE}/nonviolent',
            'need_separation': False,
            'recursive': True
        },
        {
            'name': 'nonviolent_kaggle',
            'path': f'{DATASETS_BASE}/nonviolent_kaggle',
            'need_separation': False,
            'recursive': True
        },
        {
            'name': 'cctv_surveillance',
            'path': f'{DATASETS_BASE}/cctv_surveillance',
            'need_separation': False,
            'recursive': True
        },
    ]
}

# Output directory
OUTPUT_DIR = f"{DATASETS_BASE}/combined_dataset"

# Video extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_videos(directory: Path, recursive=True):
    """Find all videos in directory."""
    videos = []

    if recursive:
        for ext in VIDEO_EXTENSIONS:
            videos.extend(list(directory.rglob(f'*{ext}')))
    else:
        for ext in VIDEO_EXTENSIONS:
            videos.extend(list(directory.glob(f'*{ext}')))

    return videos


def check_phase1_separation():
    """Check if phase1 has been separated."""
    phase1_categorized = Path(f"{DATASETS_BASE}/phase1_categorized")

    if not phase1_categorized.exists():
        return False, "Directory doesn't exist"

    violent_dir = phase1_categorized / "violent"
    nonviolent_dir = phase1_categorized / "nonviolent"

    if not violent_dir.exists() or not nonviolent_dir.exists():
        return False, "Subdirectories missing"

    violent_count = len(find_videos(violent_dir, recursive=False))
    nonviolent_count = len(find_videos(nonviolent_dir, recursive=False))

    if violent_count == 0 and nonviolent_count == 0:
        return False, "No videos found"

    return True, f"{violent_count} violent, {nonviolent_count} nonviolent"


def analyze_datasets():
    """Analyze all source datasets before combining."""
    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    print()

    analysis = {
        'violent': {},
        'nonviolent': {},
        'total_violent': 0,
        'total_nonviolent': 0
    }

    # Check phase1 separation status
    print("📊 Checking phase1 separation status...")
    separated, status = check_phase1_separation()

    if separated:
        print(f"   ✅ phase1 separated: {status}")
    else:
        print(f"   ⚠️  phase1 NOT separated: {status}")
        print(f"   ⚠️  Run: python3 separate_phase1_optimized.py")
        print()

    # Analyze violent datasets
    print("\n⚠️  VIOLENT DATASETS:")
    print("-" * 80)

    for source in SOURCES['violent']:
        name = source['name']

        if source['need_separation']:
            if separated:
                path = Path(source['use_path'])
            else:
                print(f"   ⚠️  {name}: Needs separation first")
                analysis['violent'][name] = {'status': 'needs_separation', 'count': 0}
                continue
        else:
            path = Path(source['path'])

        if not path.exists():
            print(f"   ❌ {name}: NOT FOUND at {path}")
            analysis['violent'][name] = {'status': 'not_found', 'count': 0}
            continue

        videos = find_videos(path, recursive=source.get('recursive', True))
        count = len(videos)
        analysis['violent'][name] = {
            'status': 'ready',
            'count': count,
            'path': str(path)
        }
        analysis['total_violent'] += count

        print(f"   ✅ {name:30s} → {count:6,} videos")

    # Analyze nonviolent datasets
    print("\n✅ NON-VIOLENT DATASETS:")
    print("-" * 80)

    for source in SOURCES['nonviolent']:
        name = source['name']

        if source['need_separation']:
            if separated:
                path = Path(source['use_path'])
            else:
                print(f"   ⚠️  {name}: Needs separation first")
                analysis['nonviolent'][name] = {'status': 'needs_separation', 'count': 0}
                continue
        else:
            path = Path(source['path'])

        if not path.exists():
            print(f"   ❌ {name}: NOT FOUND at {path}")
            analysis['nonviolent'][name] = {'status': 'not_found', 'count': 0}
            continue

        videos = find_videos(path, recursive=source.get('recursive', True))
        count = len(videos)
        analysis['nonviolent'][name] = {
            'status': 'ready',
            'count': count,
            'path': str(path)
        }
        analysis['total_nonviolent'] += count

        print(f"   ✅ {name:30s} → {count:6,} videos")

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"\n⚠️  Total Violent:     {analysis['total_violent']:7,} videos")
    print(f"✅ Total Non-Violent: {analysis['total_nonviolent']:7,} videos")
    print(f"📊 Total:             {analysis['total_violent'] + analysis['total_nonviolent']:7,} videos")

    if analysis['total_violent'] > 0 and analysis['total_nonviolent'] > 0:
        ratio = analysis['total_nonviolent'] / analysis['total_violent']
        print(f"\n⚖️  CLASS BALANCE:")
        print(f"   Violent:Non-Violent = {analysis['total_violent']:,}:{analysis['total_nonviolent']:,}")
        print(f"   Ratio: {ratio:.2%}")

        if ratio < 0.3:
            shortage = int(analysis['total_violent'] * 0.5 - analysis['total_nonviolent'])
            print(f"\n   ⚠️  SEVERE IMBALANCE!")
            print(f"   ⚠️  Need ~{shortage:,} more non-violent videos for 50/50 balance")
        elif ratio < 0.7:
            shortage = int(analysis['total_violent'] * 0.8 - analysis['total_nonviolent'])
            print(f"\n   ⚠️  Moderate imbalance")
            print(f"   Suggestion: Add ~{shortage:,} more non-violent videos")
        else:
            print(f"\n   ✅ Good balance!")

    return analysis


def combine_datasets(analysis, mode='copy'):
    """
    Combine all datasets into violent/nonviolent folders.

    mode: 'copy' (safe, preserves originals) or 'move' (saves space)
    """
    print("\n" + "="*80)
    print("COMBINING DATASETS")
    print("="*80)
    print(f"\nMode: {mode}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Check if we can proceed
    ready_count = sum(1 for v in analysis['violent'].values() if v['status'] == 'ready')
    ready_count += sum(1 for v in analysis['nonviolent'].values() if v['status'] == 'ready')

    if ready_count == 0:
        print("❌ No datasets ready to combine!")
        return False

    # Create output directories
    output_path = Path(OUTPUT_DIR)
    violent_dir = output_path / "violent"
    nonviolent_dir = output_path / "nonviolent"

    violent_dir.mkdir(parents=True, exist_ok=True)
    nonviolent_dir.mkdir(parents=True, exist_ok=True)

    stats = {'violent': 0, 'nonviolent': 0, 'errors': 0}

    # Process violent datasets
    print("⚠️  Processing VIOLENT datasets...")
    print("-" * 80)

    violent_counter = 0

    for source in SOURCES['violent']:
        name = source['name']

        if analysis['violent'][name]['status'] != 'ready':
            print(f"   ⏭️  Skipping {name} (not ready)")
            continue

        if source['need_separation']:
            source_path = Path(source['use_path'])
        else:
            source_path = Path(source['path'])

        videos = find_videos(source_path, recursive=source.get('recursive', True))

        print(f"\n   📁 {name}: {len(videos):,} videos")

        for video in tqdm(videos, desc=f"   Copying {name}"):
            try:
                dst = violent_dir / f"violent_{violent_counter:07d}{video.suffix}"

                if mode == 'move':
                    shutil.move(str(video), str(dst))
                else:
                    shutil.copy2(str(video), str(dst))

                violent_counter += 1
                stats['violent'] += 1

            except Exception as e:
                print(f"\n      ⚠️  Error with {video.name}: {e}")
                stats['errors'] += 1

    print(f"\n   ✅ Total violent: {stats['violent']:,} videos")

    # Process nonviolent datasets
    print("\n✅ Processing NON-VIOLENT datasets...")
    print("-" * 80)

    nonviolent_counter = 0

    for source in SOURCES['nonviolent']:
        name = source['name']

        if analysis['nonviolent'][name]['status'] != 'ready':
            print(f"   ⏭️  Skipping {name} (not ready)")
            continue

        if source['need_separation']:
            source_path = Path(source['use_path'])
        else:
            source_path = Path(source['path'])

        videos = find_videos(source_path, recursive=source.get('recursive', True))

        print(f"\n   📁 {name}: {len(videos):,} videos")

        for video in tqdm(videos, desc=f"   Copying {name}"):
            try:
                dst = nonviolent_dir / f"nonviolent_{nonviolent_counter:07d}{video.suffix}"

                if mode == 'move':
                    shutil.move(str(video), str(dst))
                else:
                    shutil.copy2(str(video), str(dst))

                nonviolent_counter += 1
                stats['nonviolent'] += 1

            except Exception as e:
                print(f"\n      ⚠️  Error with {video.name}: {e}")
                stats['errors'] += 1

    print(f"\n   ✅ Total non-violent: {stats['nonviolent']:,} videos")

    # Save metadata
    metadata = {
        'created': str(Path(__file__).name),
        'mode': mode,
        'sources': analysis,
        'output': {
            'violent': stats['violent'],
            'nonviolent': stats['nonviolent'],
            'errors': stats['errors']
        }
    }

    metadata_file = output_path / "combination_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("✅ COMBINATION COMPLETE")
    print("="*80)
    print(f"\n📊 Results:")
    print(f"   Violent videos:     {stats['violent']:7,}")
    print(f"   Non-violent videos: {stats['nonviolent']:7,}")
    print(f"   Total combined:     {stats['violent'] + stats['nonviolent']:7,}")
    print(f"   Errors:             {stats['errors']:7,}")

    print(f"\n📁 Output location:")
    print(f"   {OUTPUT_DIR}/violent/")
    print(f"   {OUTPUT_DIR}/nonviolent/")

    # Balance check
    if stats['violent'] > 0 and stats['nonviolent'] > 0:
        ratio = stats['nonviolent'] / stats['violent']
        print(f"\n⚖️  CLASS BALANCE:")
        print(f"   Ratio: {ratio:.2%}")

        if ratio < 0.3:
            shortage = int(stats['violent'] * 0.5 - stats['nonviolent'])
            print(f"\n   ⚠️  SEVERE IMBALANCE!")
            print(f"   ⚠️  Recommend collecting {shortage:,} more non-violent videos")
            print(f"   ⚠️  For 50/50 balance, need: {int(stats['violent'] * 0.5):,} non-violent total")
        elif ratio < 0.7:
            shortage = int(stats['violent'] - stats['nonviolent'])
            print(f"\n   ⚠️  Moderate imbalance")
            print(f"   Suggestion: Add ~{shortage:,} more non-violent videos")
        else:
            print(f"\n   ✅ Good balance!")

    print()
    return True


def main():
    """Main execution."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     COMBINE ALL DATASETS INTO VIOLENT/NONVIOLENT            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Analyze
    analysis = analyze_datasets()

    # Check if phase1 needs separation
    needs_separation = any(
        v['status'] == 'needs_separation'
        for v in list(analysis['violent'].values()) + list(analysis['nonviolent'].values())
    )

    if needs_separation:
        print("\n" + "="*80)
        print("⚠️  PHASE1 NEEDS SEPARATION FIRST!")
        print("="*80)
        print("\nRun these commands first:")
        print("   1. python3 analyze_phase1_optimized.py")
        print("   2. python3 separate_phase1_optimized.py")
        print("\nThen run this script again.")
        print()

        proceed = input("Continue anyway (skip phase1)? [y/N]: ").strip().lower()
        if proceed != 'y':
            print("\nExiting. Run phase1 separation first.")
            return

    # Check if any datasets are ready
    total_ready = analysis['total_violent'] + analysis['total_nonviolent']
    if total_ready == 0:
        print("\n❌ No datasets ready to combine!")
        return

    # Step 2: Choose mode
    print("\n" + "="*80)
    print("COMBINATION MODE")
    print("="*80)
    print("\nAvailable modes:")
    print("  1. copy - Copy files (safe, preserves originals, needs 2x space)")
    print("  2. move - Move files (saves space, original locations will be empty)")
    print()

    mode_choice = input("Select mode (1/2) [default: 1]: ").strip() or "1"
    mode = 'move' if mode_choice == '2' else 'copy'

    if mode == 'move':
        print("\n⚠️  WARNING: MOVE mode will empty the original dataset folders!")
        confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
        if confirm != 'yes':
            print("\nCancelled. Using copy mode instead.")
            mode = 'copy'

    # Step 3: Combine
    success = combine_datasets(analysis, mode=mode)

    if success:
        print("="*80)
        print("🎯 NEXT STEPS")
        print("="*80)
        print(f"\n1. Review combined dataset in: {OUTPUT_DIR}")
        print(f"2. Collect more non-violent data if needed")
        print(f"3. Create train/val/test splits:")
        print(f"   python3 analyze_and_split_dataset.py")
        print(f"4. Train model:")
        print(f"   python3 train_dual_rtx5000.py")
        print()


if __name__ == "__main__":
    main()
