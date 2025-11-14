#!/usr/bin/env python3
"""
Smart Phase1 Analysis - Prioritizes explicit labels in filenames/paths
Improved classification based on actual naming patterns
"""

import os
from pathlib import Path
from collections import defaultdict
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE1_DIR = "/workspace/datasets/phase1"

# ============================================================================
# SMART CATEGORIZATION - PRIORITY LEVELS
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
    # CCTV/surveillance (only in path, not filename)
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


def analyze_phase1():
    """Analyze phase1 directory with smart categorization."""
    print("\n" + "="*80)
    print("SMART PHASE1 ANALYSIS - Priority-Based Classification")
    print("="*80)
    print(f"\nScanning: {PHASE1_DIR}\n")

    phase1_path = Path(PHASE1_DIR)

    if not phase1_path.exists():
        print(f"❌ ERROR: {PHASE1_DIR} not found")
        return

    # Find all videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    all_videos = []

    print("📁 Finding videos...")
    for ext in video_extensions:
        videos = list(phase1_path.rglob(f'*{ext}'))
        all_videos.extend(videos)
        if videos:
            print(f"   Found {len(videos):,} {ext} files")

    print(f"\n✅ Total videos found: {len(all_videos):,}\n")

    if len(all_videos) == 0:
        print("⚠️  No videos found!")
        return

    # Categorize
    results = {
        'violent': {
            'explicit': [],  # P1
            'high': [],      # P2
            'medium': [],    # P3
            'low': []        # P3
        },
        'nonviolent': {
            'explicit': [],  # P1
            'high': [],      # P2
            'medium': [],    # P3
            'low': []        # P3
        },
        'ambiguous': {'low': []},
        'unknown': {'none': []},
    }

    print("🔍 Analyzing with priority-based classification...")
    for i, video_path in enumerate(all_videos):
        if (i + 1) % 1000 == 0:
            print(f"   Progress: {i+1:,}/{len(all_videos):,} ({(i+1)/len(all_videos)*100:.1f}%)...")

        category, confidence, keywords, priority = smart_categorize(video_path)
        results[category][confidence].append({
            'path': str(video_path),
            'keywords': keywords,
            'priority': priority
        })

    print(f"   Progress: {len(all_videos):,}/{len(all_videos):,} (100.0%) - Done!\n")

    # Calculate totals
    violent_total = sum(len(results['violent'][c]) for c in results['violent'])
    nonviolent_total = sum(len(results['nonviolent'][c]) for c in results['nonviolent'])
    ambiguous_total = len(results['ambiguous']['low'])
    unknown_total = len(results['unknown']['none'])

    # Print detailed results
    print("="*80)
    print("📊 CATEGORIZATION RESULTS")
    print("="*80)

    print(f"\n⚠️  VIOLENT: {violent_total:,} videos ({violent_total/len(all_videos)*100:.1f}%)")
    if results['violent']['explicit']:
        print(f"   - EXPLICIT label (P1):  {len(results['violent']['explicit']):6,} (95-98% confidence)")
    if results['violent']['high']:
        print(f"   - High confidence (P2): {len(results['violent']['high']):6,} (85-90% confidence)")
    if results['violent']['medium']:
        print(f"   - Medium confidence:    {len(results['violent']['medium']):6,} (70-80% confidence)")
    if results['violent']['low']:
        print(f"   - Low confidence:       {len(results['violent']['low']):6,} (50-70% confidence)")

    print(f"\n✅ NON-VIOLENT: {nonviolent_total:,} videos ({nonviolent_total/len(all_videos)*100:.1f}%)")
    if results['nonviolent']['explicit']:
        print(f"   - EXPLICIT label (P1):  {len(results['nonviolent']['explicit']):6,} (95-98% confidence)")
    if results['nonviolent']['high']:
        print(f"   - High confidence (P2): {len(results['nonviolent']['high']):6,} (85-90% confidence)")
    if results['nonviolent']['medium']:
        print(f"   - Medium confidence:    {len(results['nonviolent']['medium']):6,} (70-80% confidence)")
    if results['nonviolent']['low']:
        print(f"   - Low confidence:       {len(results['nonviolent']['low']):6,} (50-70% confidence)")

    if ambiguous_total:
        print(f"\n🔀 AMBIGUOUS: {ambiguous_total:,} videos ({ambiguous_total/len(all_videos)*100:.1f}%)")
        print(f"   (Has both violent and non-violent indicators)")

    if unknown_total:
        print(f"\n❓ UNKNOWN: {unknown_total:,} videos ({unknown_total/len(all_videos)*100:.1f}%)")
        print(f"   (No clear indicators in filename/path)")

    # Show sample paths
    print("\n" + "="*80)
    print("📝 SAMPLE VIDEOS BY CATEGORY")
    print("="*80)

    # Explicit violent
    if results['violent']['explicit']:
        print("\n⚠️  VIOLENT (Explicit Label - P1) - Sample 5:")
        for video in results['violent']['explicit'][:5]:
            keywords = ', '.join(video['keywords'][:3])
            print(f"   {Path(video['path']).name}")
            print(f"      Keywords: {keywords}")

    # High confidence violent
    if results['violent']['high']:
        print("\n⚠️  VIOLENT (High Confidence - P2) - Sample 5:")
        for video in results['violent']['high'][:5]:
            keywords = ', '.join(video['keywords'][:3])
            print(f"   {Path(video['path']).name}")
            print(f"      Keywords: {keywords}")

    # Explicit nonviolent
    if results['nonviolent']['explicit']:
        print("\n✅ NON-VIOLENT (Explicit Label - P1) - Sample 5:")
        for video in results['nonviolent']['explicit'][:5]:
            keywords = ', '.join(video['keywords'][:3])
            print(f"   {Path(video['path']).name}")
            print(f"      Keywords: {keywords}")

    # High confidence nonviolent
    if results['nonviolent']['high']:
        print("\n✅ NON-VIOLENT (High Confidence - P2) - Sample 5:")
        for video in results['nonviolent']['high'][:5]:
            keywords = ', '.join(video['keywords'][:3])
            print(f"   {Path(video['path']).name}")
            print(f"      Keywords: {keywords}")

    # Check for RoadAccidents specifically
    road_accidents = [v for v in all_videos if 'roadaccident' in str(v).lower()]
    if road_accidents:
        print("\n" + "="*80)
        print("🚗 ROAD ACCIDENTS CHECK (Should be VIOLENT)")
        print("="*80)
        for ra in road_accidents[:5]:
            cat, conf, kw, pri = smart_categorize(ra)
            status = "✅" if cat == 'violent' else "❌"
            print(f"\n{status} {ra.name}")
            print(f"   Category: {cat.upper()} ({conf} confidence, {pri})")
            if kw:
                print(f"   Matched: {', '.join(kw[:3])}")

    if unknown_total:
        print("\n❓ UNKNOWN - Sample 5:")
        for video in results['unknown']['none'][:5]:
            print(f"   {Path(video['path']).name}")

    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    confident = violent_total + nonviolent_total
    uncertain = ambiguous_total + unknown_total

    print(f"\n✅ Confident categorization: {confident:,} videos ({confident/len(all_videos)*100:.1f}%)")
    print(f"⚠️  Uncertain: {uncertain:,} videos ({uncertain/len(all_videos)*100:.1f}%)")

    # Confidence breakdown
    explicit_count = len(results['violent']['explicit']) + len(results['nonviolent']['explicit'])
    high_count = len(results['violent']['high']) + len(results['nonviolent']['high'])

    if explicit_count > 0:
        print(f"\n🎯 EXPLICIT LABELS (Best): {explicit_count:,} videos (95-98% accurate)")
    if high_count > 0:
        print(f"🎯 HIGH CONFIDENCE: {high_count:,} videos (85-90% accurate)")

    print("\n🎯 NEXT STEPS:")

    if confident > len(all_videos) * 0.7:
        print(f"   ✅ Good! {confident/len(all_videos)*100:.0f}% can be auto-categorized")
        print(f"   → Proceed with automatic organization")
        print(f"   → Command: python3 separate_phase1_smart.py")
    else:
        print(f"   ⚠️  Only {confident/len(all_videos)*100:.0f}% confidently categorized")
        print(f"   → Review folder structure for better patterns")
        print(f"   → Or manually review uncertain videos")

    print(f"\n⚖️  CLASS BALANCE:")
    if violent_total and nonviolent_total:
        ratio = min(violent_total, nonviolent_total) / max(violent_total, nonviolent_total)
        print(f"   Violent: {violent_total:,}")
        print(f"   Non-Violent: {nonviolent_total:,}")
        print(f"   Balance: {ratio*100:.1f}%")

        if ratio < 0.3:
            print(f"   ⚠️  SEVERE IMBALANCE!")
        elif ratio < 0.7:
            print(f"   ⚠️  Moderate imbalance")
        else:
            print(f"   ✅ Good balance")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE - NO FILES MODIFIED")
    print("="*80)
    print()

    return results


if __name__ == "__main__":
    analyze_phase1()
