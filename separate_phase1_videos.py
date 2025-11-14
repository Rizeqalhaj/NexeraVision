#!/usr/bin/env python3
"""
Intelligent Video Categorization Script for phase1
Separates mixed violent/non-violent videos using multiple detection methods.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE1_DIR = "/workspace/datasets/phase1"
OUTPUT_DIR = "/workspace/datasets/phase1_categorized"

# ============================================================================
# PATTERN-BASED CATEGORIZATION
# ============================================================================

# Violent keywords (in filenames or folder names)
VIOLENT_KEYWORDS = [
    # Fight-related
    'fight', 'fighting', 'fighter', 'brawl', 'combat',
    'punch', 'punching', 'kick', 'kicking', 'hit', 'hitting',
    'beat', 'beating', 'attack', 'assault',

    # Martial arts
    'ufc', 'mma', 'boxing', 'boxer', 'kickbox', 'muay thai',
    'karate', 'judo', 'jujitsu', 'wrestling', 'wrestle',
    'martial', 'sparring', 'knockout', 'ko',

    # Street violence
    'street fight', 'streetfight', 'bully', 'gang',
    'riot', 'protest', 'clash',

    # Violent actions
    'violence', 'violent', 'brutal', 'bloody',
    'slam', 'smash', 'knock', 'takedown',
    'submission', 'choke', 'strangle',

    # Accidents and injuries (MOVED FROM AMBIGUOUS)
    'accident', 'crash', 'collision', 'wreck', 'smash up',
    'injury', 'hurt', 'wounded', 'bleeding', 'fatal',
    'road accident', 'roadaccident', 'car crash', 'vehicle crash',
    'train crash', 'plane crash', 'motorcycle accident',

    # Reddit violence indicators
    'r_fight', 'r_street', 'r_brutal', 'r_real',
    'r_femalemma', 'r_publicfreakout', 'freakout',

    # Generic violence
    'aggress', 'hostile', 'confrontation',
    'altercation', 'scuffle', 'tussle',
]

# Non-violent keywords (REMOVED GENERIC TERMS: work, working, road, car, vehicle, drive, traffic)
NONVIOLENT_KEYWORDS = [
    # Normal activities
    'walk', 'walking', 'sit', 'sitting', 'stand', 'standing',
    'talk', 'talking', 'conversation', 'chat',
    'shop', 'shopping', 'store', 'mall',
    'office', 'desk', 'meeting', 'conference',

    # CCTV/Surveillance
    'cctv', 'surveillance', 'camera', 'security',
    'monitor', 'parking', 'lobby', 'entrance',
    'corridor', 'hallway', 'street view',

    # Daily life
    'daily', 'normal', 'routine', 'activity',
    'crowd', 'people', 'pedestrian', 'passerby',

    # Safe activities
    'dance', 'dancing', 'play', 'playing', 'game',
    'sport', 'exercise', 'run', 'running',
    'eat', 'eating', 'drink', 'drinking',

    # Positive
    'smile', 'laugh', 'happy', 'joy', 'celebrate',
    'hug', 'kiss', 'love', 'friend',

    # Non-violent Reddit
    'r_wholesome', 'r_aww', 'r_mademesmile',
    'r_uplift', 'r_eyebleach', 'r_happy',

    # Events (usually non-violent)
    'concert', 'show', 'performance', 'music',
    'wedding', 'party', 'birthday', 'celebration',
]

# Ambiguous keywords (could be either - need manual review)
AMBIGUOUS_KEYWORDS = [
    'police', 'cop', 'arrest', 'law enforcement',
    'court', 'judge', 'trial', 'justice',
    'argument', 'dispute', 'disagree',
    'push', 'shove', 'grab', 'fall',
]


def categorize_by_pattern(file_path: str) -> tuple:
    """
    Categorize video based on filename and folder path patterns.
    Returns: (category, confidence, matched_keywords)
    """
    path_lower = str(file_path).lower()

    violent_matches = []
    nonviolent_matches = []
    ambiguous_matches = []

    # Check for violent keywords
    for keyword in VIOLENT_KEYWORDS:
        if keyword in path_lower:
            violent_matches.append(keyword)

    # Check for non-violent keywords
    for keyword in NONVIOLENT_KEYWORDS:
        if keyword in path_lower:
            nonviolent_matches.append(keyword)

    # Check for ambiguous keywords
    for keyword in AMBIGUOUS_KEYWORDS:
        if keyword in path_lower:
            ambiguous_matches.append(keyword)

    # Decision logic
    if violent_matches and not nonviolent_matches:
        return 'violent', 'high', violent_matches

    if nonviolent_matches and not violent_matches:
        return 'nonviolent', 'high', nonviolent_matches

    if len(violent_matches) > len(nonviolent_matches):
        return 'violent', 'medium', violent_matches

    if len(nonviolent_matches) > len(violent_matches):
        return 'nonviolent', 'medium', nonviolent_matches

    if ambiguous_matches:
        return 'ambiguous', 'low', ambiguous_matches

    return 'unknown', 'none', []


def analyze_phase1(phase1_dir: str):
    """
    Analyze phase1 directory and categorize all videos.
    """
    print("="*80)
    print("PHASE1 VIDEO CATEGORIZATION ANALYSIS")
    print("="*80)
    print(f"\nScanning: {phase1_dir}\n")

    phase1_path = Path(phase1_dir)

    if not phase1_path.exists():
        print(f"❌ ERROR: Directory not found: {phase1_dir}")
        return None

    # Find all videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    all_videos = []

    print("📁 Finding all videos...")
    for ext in video_extensions:
        all_videos.extend(list(phase1_path.rglob(f'*{ext}')))

    print(f"✅ Found {len(all_videos):,} videos\n")

    # Categorize each video
    categories = {
        'violent': {'high': [], 'medium': [], 'low': []},
        'nonviolent': {'high': [], 'medium': [], 'low': []},
        'ambiguous': {'high': [], 'medium': [], 'low': []},
        'unknown': {'none': []},
    }

    print("🔍 Categorizing videos by pattern matching...")
    for i, video_path in enumerate(all_videos):
        if i % 1000 == 0 and i > 0:
            print(f"   Processed {i:,}/{len(all_videos):,}...", end='\r')

        category, confidence, keywords = categorize_by_pattern(video_path)
        categories[category][confidence].append({
            'path': str(video_path),
            'keywords': keywords
        })

    print(f"   Processed {len(all_videos):,}/{len(all_videos):,}... Done!")

    # Print statistics
    print("\n" + "="*80)
    print("📊 CATEGORIZATION RESULTS")
    print("="*80)

    violent_total = sum(len(categories['violent'][conf]) for conf in categories['violent'])
    nonviolent_total = sum(len(categories['nonviolent'][conf]) for conf in categories['nonviolent'])
    ambiguous_total = sum(len(categories['ambiguous'][conf]) for conf in categories['ambiguous'])
    unknown_total = len(categories['unknown']['none'])

    print(f"\n⚠️  VIOLENT: {violent_total:,} videos")
    for conf in ['high', 'medium', 'low']:
        count = len(categories['violent'][conf])
        if count > 0:
            print(f"   - {conf.capitalize()} confidence: {count:,}")

    print(f"\n✅ NON-VIOLENT: {nonviolent_total:,} videos")
    for conf in ['high', 'medium', 'low']:
        count = len(categories['nonviolent'][conf])
        if count > 0:
            print(f"   - {conf.capitalize()} confidence: {count:,}")

    print(f"\n🔀 AMBIGUOUS: {ambiguous_total:,} videos")
    print(f"   (Contains both violent and non-violent indicators)")

    print(f"\n❓ UNKNOWN: {unknown_total:,} videos")
    print(f"   (No clear indicators - needs manual review or ML)")

    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    auto_categorizable = violent_total + nonviolent_total
    needs_review = ambiguous_total + unknown_total

    print(f"\n✅ Auto-categorizable: {auto_categorizable:,} ({auto_categorizable/len(all_videos)*100:.1f}%)")
    print(f"⚠️  Needs review: {needs_review:,} ({needs_review/len(all_videos)*100:.1f}%)")

    if unknown_total > 1000:
        print(f"\n🎯 STRATEGY FOR {unknown_total:,} UNKNOWN VIDEOS:")
        print("   Option 1: Exclude from training (safest)")
        print("   Option 2: Manual sample review (50-100 videos) + extrapolate")
        print("   Option 3: Use filename structure patterns")
        print("   Option 4: Assume based on folder majority")

    return categories


def organize_videos(categories: dict, output_dir: str, mode='high_confidence_only'):
    """
    Organize videos into violent/nonviolent folders.

    Modes:
    - high_confidence_only: Only use high confidence categorizations
    - all_auto: Use all automatic categorizations (high + medium + low)
    - include_ambiguous_as_violent: Treat ambiguous as violent
    - include_unknown_based_on_folder: Use folder patterns for unknown
    """
    print("\n" + "="*80)
    print("ORGANIZING VIDEOS")
    print("="*80)
    print(f"\nMode: {mode}\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    violent_dir = output_path / "violent"
    nonviolent_dir = output_path / "nonviolent"
    review_dir = output_path / "needs_review"

    violent_dir.mkdir(exist_ok=True)
    nonviolent_dir.mkdir(exist_ok=True)
    review_dir.mkdir(exist_ok=True)

    stats = {'violent': 0, 'nonviolent': 0, 'needs_review': 0}

    # Determine which videos to include
    if mode == 'high_confidence_only':
        violent_videos = categories['violent']['high']
        nonviolent_videos = categories['nonviolent']['high']
        review_videos = (
            categories['violent']['medium'] +
            categories['violent']['low'] +
            categories['nonviolent']['medium'] +
            categories['nonviolent']['low'] +
            sum(categories['ambiguous'].values(), []) +
            categories['unknown']['none']
        )

    elif mode == 'all_auto':
        violent_videos = sum(categories['violent'].values(), [])
        nonviolent_videos = sum(categories['nonviolent'].values(), [])
        review_videos = (
            sum(categories['ambiguous'].values(), []) +
            categories['unknown']['none']
        )

    elif mode == 'include_ambiguous_as_violent':
        violent_videos = (
            sum(categories['violent'].values(), []) +
            sum(categories['ambiguous'].values(), [])
        )
        nonviolent_videos = sum(categories['nonviolent'].values(), [])
        review_videos = categories['unknown']['none']

    else:  # Default: high_confidence_only
        violent_videos = categories['violent']['high']
        nonviolent_videos = categories['nonviolent']['high']
        review_videos = []

    # Copy violent videos
    print(f"📁 Organizing {len(violent_videos):,} VIOLENT videos...")
    for i, video_info in enumerate(violent_videos):
        if i % 100 == 0 and i > 0:
            print(f"   {i}/{len(violent_videos)}...", end='\r')

        src = Path(video_info['path'])
        if not src.exists():
            continue

        dst = violent_dir / f"violent_{i:06d}{src.suffix}"
        try:
            shutil.copy2(src, dst)
            stats['violent'] += 1
        except Exception as e:
            print(f"\n   ⚠️  Error copying {src.name}: {e}")

    print(f"   ✅ {stats['violent']:,} violent videos organized")

    # Copy non-violent videos
    print(f"\n📁 Organizing {len(nonviolent_videos):,} NON-VIOLENT videos...")
    for i, video_info in enumerate(nonviolent_videos):
        if i % 100 == 0 and i > 0:
            print(f"   {i}/{len(nonviolent_videos)}...", end='\r')

        src = Path(video_info['path'])
        if not src.exists():
            continue

        dst = nonviolent_dir / f"nonviolent_{i:06d}{src.suffix}"
        try:
            shutil.copy2(src, dst)
            stats['nonviolent'] += 1
        except Exception as e:
            print(f"\n   ⚠️  Error copying {src.name}: {e}")

    print(f"   ✅ {stats['nonviolent']:,} non-violent videos organized")

    # Save review list (don't copy, just list paths)
    if review_videos:
        review_file = review_dir / "videos_to_review.txt"
        print(f"\n📝 Saving {len(review_videos):,} videos needing review to {review_file}")
        with open(review_file, 'w') as f:
            for video_info in review_videos:
                f.write(f"{video_info['path']}\n")

    # Save categorization metadata
    metadata_file = output_path / "categorization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'mode': mode,
            'stats': stats,
            'total_reviewed': len(violent_videos) + len(nonviolent_videos),
            'needs_review': len(review_videos),
        }, f, indent=2)

    print(f"\n✅ Metadata saved to {metadata_file}")

    return stats


def print_summary(stats: dict, output_dir: str):
    """Print final summary."""
    print("\n" + "="*80)
    print("✅ CATEGORIZATION COMPLETE")
    print("="*80)

    print(f"\n📊 Results:")
    print(f"   Violent videos:     {stats['violent']:6,}")
    print(f"   Non-violent videos: {stats['nonviolent']:6,}")
    print(f"   Total organized:    {stats['violent'] + stats['nonviolent']:6,}")

    print(f"\n📁 Output location: {output_dir}")
    print(f"   - {output_dir}/violent/")
    print(f"   - {output_dir}/nonviolent/")
    print(f"   - {output_dir}/needs_review/")

    # Balance check
    if stats['violent'] and stats['nonviolent']:
        ratio = min(stats['violent'], stats['nonviolent']) / max(stats['violent'], stats['nonviolent'])
        print(f"\n⚖️  Class Balance: {ratio*100:.1f}%")

        if ratio < 0.5:
            print(f"   ⚠️  Significant imbalance!")
            print(f"   Consider downsampling to {min(stats['violent'], stats['nonviolent']):,} per class")
        else:
            print(f"   ✅ Good balance!")


def main():
    """Main execution."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     PHASE1 VIDEO CATEGORIZATION (Pattern-Based)             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Analyze
    categories = analyze_phase1(PHASE1_DIR)

    if categories is None:
        return

    # Step 2: Choose mode
    print("\n" + "="*80)
    print("ORGANIZATION MODE SELECTION")
    print("="*80)
    print("\nAvailable modes:")
    print("  1. high_confidence_only - Safest, only high-confidence categorizations")
    print("  2. all_auto - Use all automatic categorizations (high + medium + low)")
    print("  3. include_ambiguous_as_violent - Treat ambiguous as violent (conservative)")
    print()

    mode_choice = input("Select mode (1/2/3) [default: 1]: ").strip() or "1"

    mode_map = {
        '1': 'high_confidence_only',
        '2': 'all_auto',
        '3': 'include_ambiguous_as_violent',
    }

    mode = mode_map.get(mode_choice, 'high_confidence_only')

    # Step 3: Organize
    stats = organize_videos(categories, OUTPUT_DIR, mode=mode)

    # Step 4: Summary
    print_summary(stats, OUTPUT_DIR)

    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print(f"\n1. Review results in: {OUTPUT_DIR}")
    print(f"2. Check a few sample videos to verify categorization")
    print(f"3. Combine with your other datasets:")
    print(f"   - youtube_fights (105 violent)")
    print(f"   - reddit_videos_massive (2,667 violent)")
    print(f"   - reddit_videos (1,669 violent)")
    print(f"   - nonviolent_safe (30 non-violent)")
    print(f"4. Create final train/val/test splits")
    print(f"5. Start training!")
    print()


if __name__ == "__main__":
    main()
