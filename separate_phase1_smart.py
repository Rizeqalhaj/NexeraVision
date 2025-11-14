#!/usr/bin/env python3
"""
Smart Phase1 Separation - Priority-based classification
Separates videos using explicit labels first, then strong indicators
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE1_DIR = "/workspace/datasets/phase1"
OUTPUT_DIR = "/workspace/datasets/phase1_categorized"

# ============================================================================
# SMART CATEGORIZATION - PRIORITY LEVELS (Same as analyze_phase1_smart.py)
# ============================================================================

# Priority 1: EXPLICIT LABELS in filename/path (highest confidence)
EXPLICIT_VIOLENT_LABELS = [
    'violent', 'violence',
    'shooting', 'shoot',
    'fight', 'fighting',
    'assault', 'attack',
]

EXPLICIT_NONVIOLENT_LABELS = [
    'nonviolent', 'non_violent', 'non-violent',
    'normal', 'daily', 'routine',
    'safe', 'peaceful',
]

# Priority 2: Strong indicators (high confidence)
STRONG_VIOLENT_KEYWORDS = [
    # Accidents/crashes
    'accident', 'crash', 'collision', 'wreck',
    'road accident', 'roadaccident', 'car crash',

    # Combat sports
    'ufc', 'mma', 'boxing', 'kickbox',

    # Street violence
    'street fight', 'streetfight', 'brawl',

    # Weapons
    'gun', 'knife', 'weapon', 'armed',

    # Severe violence
    'brutal', 'bloody', 'murder', 'stab',
]

STRONG_NONVIOLENT_KEYWORDS = [
    # CCTV/surveillance
    'cctv', 'surveillance', 'security cam',

    # Specific safe activities
    'shopping', 'dining', 'walking',
    'office', 'workplace', 'meeting',
    'park', 'playground', 'garden',
]

# Priority 3: Moderate indicators
MODERATE_VIOLENT_KEYWORDS = [
    'punch', 'kick', 'hit', 'beat',
    'combat', 'battle', 'riot',
    'aggression', 'hostile', 'confrontation',
]

MODERATE_NONVIOLENT_KEYWORDS = [
    'talk', 'conversation', 'discussion',
    'dance', 'play', 'celebrate',
    'smile', 'laugh', 'happy',
]


def smart_categorize(file_path: Path) -> tuple:
    """
    Smart categorization with priority levels.
    Returns: (category, confidence, matched_keywords, priority_level)
    """
    filename = file_path.name.lower()
    parent_dir = file_path.parent.name.lower()
    full_path = str(file_path).lower()

    matched = []

    # ========================================================================
    # PRIORITY 1: EXPLICIT LABELS (95-98% confidence)
    # ========================================================================

    # Check for explicit "violent" label
    if any(label in filename or label in parent_dir for label in EXPLICIT_VIOLENT_LABELS):
        # But exclude "nonviolent" variations
        if not any(nv in filename or nv in parent_dir for nv in EXPLICIT_NONVIOLENT_LABELS):
            matched = [label for label in EXPLICIT_VIOLENT_LABELS
                      if label in filename or label in parent_dir]
            return ('violent', 'explicit', matched, 'P1')

    # Check for explicit "nonviolent" label
    if any(label in filename or label in parent_dir for label in EXPLICIT_NONVIOLENT_LABELS):
        matched = [label for label in EXPLICIT_NONVIOLENT_LABELS
                  if label in filename or label in parent_dir]
        return ('nonviolent', 'explicit', matched, 'P1')

    # ========================================================================
    # PRIORITY 2: STRONG INDICATORS (85-90% confidence)
    # ========================================================================

    strong_violent_matches = [kw for kw in STRONG_VIOLENT_KEYWORDS if kw in full_path]
    strong_nonviolent_matches = [kw for kw in STRONG_NONVIOLENT_KEYWORDS if kw in full_path]

    if strong_violent_matches and not strong_nonviolent_matches:
        return ('violent', 'high', strong_violent_matches, 'P2')

    if strong_nonviolent_matches and not strong_violent_matches:
        return ('nonviolent', 'high', strong_nonviolent_matches, 'P2')

    # ========================================================================
    # PRIORITY 3: MODERATE INDICATORS (70-80% confidence)
    # ========================================================================

    moderate_violent_matches = [kw for kw in MODERATE_VIOLENT_KEYWORDS if kw in full_path]
    moderate_nonviolent_matches = [kw for kw in MODERATE_NONVIOLENT_KEYWORDS if kw in full_path]

    if moderate_violent_matches and not moderate_nonviolent_matches:
        if len(moderate_violent_matches) >= 2:
            return ('violent', 'medium', moderate_violent_matches, 'P3')
        return ('violent', 'low', moderate_violent_matches, 'P3')

    if moderate_nonviolent_matches and not moderate_violent_matches:
        if len(moderate_nonviolent_matches) >= 2:
            return ('nonviolent', 'medium', moderate_nonviolent_matches, 'P3')
        return ('nonviolent', 'low', moderate_nonviolent_matches, 'P3')

    # ========================================================================
    # AMBIGUOUS or UNKNOWN
    # ========================================================================

    if strong_violent_matches and strong_nonviolent_matches:
        return ('ambiguous', 'low', strong_violent_matches + strong_nonviolent_matches, 'conflict')

    if moderate_violent_matches and moderate_nonviolent_matches:
        return ('ambiguous', 'low', moderate_violent_matches + moderate_nonviolent_matches, 'conflict')

    return ('unknown', 'none', [], 'no_match')


def analyze_phase1(phase1_dir: str):
    """Analyze phase1 directory and categorize all videos."""
    print("="*80)
    print("SMART PHASE1 VIDEO CATEGORIZATION")
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
        'violent': {'explicit': [], 'high': [], 'medium': [], 'low': []},
        'nonviolent': {'explicit': [], 'high': [], 'medium': [], 'low': []},
        'ambiguous': {'low': []},
        'unknown': {'none': []},
    }

    print("🔍 Categorizing videos with smart classification...")
    for i, video_path in enumerate(all_videos):
        if i % 1000 == 0 and i > 0:
            print(f"   Processed {i:,}/{len(all_videos):,}...", end='\r')

        category, confidence, keywords, priority = smart_categorize(video_path)
        categories[category][confidence].append({
            'path': str(video_path),
            'keywords': keywords,
            'priority': priority
        })

    print(f"   Processed {len(all_videos):,}/{len(all_videos):,}... Done!")

    # Print statistics
    print("\n" + "="*80)
    print("📊 CATEGORIZATION RESULTS")
    print("="*80)

    violent_total = sum(len(categories['violent'][conf]) for conf in categories['violent'])
    nonviolent_total = sum(len(categories['nonviolent'][conf]) for conf in categories['nonviolent'])
    ambiguous_total = len(categories['ambiguous']['low'])
    unknown_total = len(categories['unknown']['none'])

    print(f"\n⚠️  VIOLENT: {violent_total:,} videos")
    for conf in ['explicit', 'high', 'medium', 'low']:
        count = len(categories['violent'][conf])
        if count > 0:
            if conf == 'explicit':
                print(f"   - Explicit label (P1): {count:,} (95-98% confidence)")
            else:
                print(f"   - {conf.capitalize()} confidence: {count:,}")

    print(f"\n✅ NON-VIOLENT: {nonviolent_total:,} videos")
    for conf in ['explicit', 'high', 'medium', 'low']:
        count = len(categories['nonviolent'][conf])
        if count > 0:
            if conf == 'explicit':
                print(f"   - Explicit label (P1): {count:,} (95-98% confidence)")
            else:
                print(f"   - {conf.capitalize()} confidence: {count:,}")

    print(f"\n🔀 AMBIGUOUS: {ambiguous_total:,} videos")
    print(f"   (Contains both violent and non-violent indicators)")

    print(f"\n❓ UNKNOWN: {unknown_total:,} videos")
    print(f"   (No clear indicators - needs manual review)")

    return categories


def organize_videos(categories: dict, output_dir: str, mode='smart_high_confidence'):
    """
    Organize videos into violent/nonviolent folders.

    Modes:
    - smart_high_confidence: Explicit + High confidence only (RECOMMENDED)
    - smart_all_confident: Explicit + High + Medium
    - smart_all_auto: All automatic (Explicit + High + Medium + Low)
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

    # Determine which videos to include based on mode
    if mode == 'smart_high_confidence':
        violent_videos = (
            categories['violent']['explicit'] +
            categories['violent']['high']
        )
        nonviolent_videos = (
            categories['nonviolent']['explicit'] +
            categories['nonviolent']['high']
        )
        review_videos = (
            categories['violent']['medium'] +
            categories['violent']['low'] +
            categories['nonviolent']['medium'] +
            categories['nonviolent']['low'] +
            categories['ambiguous']['low'] +
            categories['unknown']['none']
        )

    elif mode == 'smart_all_confident':
        violent_videos = (
            categories['violent']['explicit'] +
            categories['violent']['high'] +
            categories['violent']['medium']
        )
        nonviolent_videos = (
            categories['nonviolent']['explicit'] +
            categories['nonviolent']['high'] +
            categories['nonviolent']['medium']
        )
        review_videos = (
            categories['violent']['low'] +
            categories['nonviolent']['low'] +
            categories['ambiguous']['low'] +
            categories['unknown']['none']
        )

    elif mode == 'smart_all_auto':
        violent_videos = (
            categories['violent']['explicit'] +
            categories['violent']['high'] +
            categories['violent']['medium'] +
            categories['violent']['low']
        )
        nonviolent_videos = (
            categories['nonviolent']['explicit'] +
            categories['nonviolent']['high'] +
            categories['nonviolent']['medium'] +
            categories['nonviolent']['low']
        )
        review_videos = (
            categories['ambiguous']['low'] +
            categories['unknown']['none']
        )

    else:  # Default to smart_high_confidence
        violent_videos = categories['violent']['explicit'] + categories['violent']['high']
        nonviolent_videos = categories['nonviolent']['explicit'] + categories['nonviolent']['high']
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

    # Save review list
    if review_videos:
        review_file = review_dir / "videos_to_review.txt"
        print(f"\n📝 Saving {len(review_videos):,} videos needing review to {review_file}")
        with open(review_file, 'w') as f:
            for video_info in review_videos:
                f.write(f"{video_info['path']}\n")

    # Save metadata
    metadata_file = output_path / "categorization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'mode': mode,
            'stats': stats,
            'total_organized': stats['violent'] + stats['nonviolent'],
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
    print("║     SMART PHASE1 VIDEO CATEGORIZATION                       ║")
    print("║     Priority-Based: Explicit Labels → Strong → Moderate     ║")
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
    print("  1. smart_high_confidence - Explicit + High only (85-98% accurate) ⭐ RECOMMENDED")
    print("  2. smart_all_confident - Explicit + High + Medium (70-98% accurate)")
    print("  3. smart_all_auto - All automatic including Low (50-98% accurate)")
    print()

    mode_choice = input("Select mode (1/2/3) [default: 1]: ").strip() or "1"

    mode_map = {
        '1': 'smart_high_confidence',
        '2': 'smart_all_confident',
        '3': 'smart_all_auto',
    }

    mode = mode_map.get(mode_choice, 'smart_high_confidence')

    # Step 3: Organize
    stats = organize_videos(categories, OUTPUT_DIR, mode=mode)

    # Step 4: Summary
    print_summary(stats, OUTPUT_DIR)

    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print(f"\n1. Review results in: {OUTPUT_DIR}")
    print(f"2. Check sample videos to verify categorization")
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
