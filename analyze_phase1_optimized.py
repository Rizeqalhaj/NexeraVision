#!/usr/bin/env python3
"""
Optimized Phase1 Analysis - Uses actual path and filename structure
Based on investigation showing clear dataset labels
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
# OPTIMIZED CATEGORIZATION - Based on Actual Data Structure
# ============================================================================

# Priority 1: PATH STRUCTURE (99% confidence)
# These are folder names in the path that clearly indicate category
VIOLENT_PATH_LABELS = [
    '/violence/',
    '/fight/',
    '/fighting/',
    '/weaponized/',
    '/vandalism/',
    '/assault/',
    '/robbery/',
    '/burglary/',
    '/shooting/',
    '/stabbing/',
    '/arrest/',
    '/abuse/',
    '/explosion/',
]

NONVIOLENT_PATH_LABELS = [
    '/nonviolence/',
    '/nonfight/',
    '/normal/',
    '/safe/',
    '/regular/',
]

# Priority 2: FILENAME PATTERNS (95% confidence)
# Explicit patterns in filenames
VIOLENT_FILENAME_PATTERNS = [
    '_violence',
    '_fighting',
    '_shooting',
    '_fight',
    '_assault',
    '_robbery',
    '_burglary',
    '_vandalism',
    '_weaponized',
    '_explosion',
    '_abuse',
    '_arrest',
    'violence_',
    'fighting_',
    'fight_',
    'v_',  # V_535.mp4 pattern from Real Life Violence
]

NONVIOLENT_FILENAME_PATTERNS = [
    '_normal',
    '_nonfight',
    '_nonviolence',
    '_safe',
    'normal_',
    'nonfight_',
    'nv_',  # NV_44.mp4 pattern from Real Life Violence
]

# Priority 3: STRONG KEYWORDS (85% confidence)
STRONG_VIOLENT_KEYWORDS = [
    'accident', 'crash', 'roadaccident',
    'ufc', 'mma', 'boxing',
    'gun', 'knife', 'weapon',
]

STRONG_NONVIOLENT_KEYWORDS = [
    'cctv', 'surveillance',
    'shopping', 'office', 'meeting',
]


def optimized_categorize(file_path: Path) -> tuple:
    """
    Optimized categorization based on actual data structure.
    Returns: (category, confidence, matched_pattern, priority_level)
    """
    path_lower = str(file_path).lower()
    filename_lower = file_path.name.lower()

    # ========================================================================
    # PRIORITY 1: PATH STRUCTURE (99% confidence)
    # ========================================================================

    # Check for violent path labels
    for label in VIOLENT_PATH_LABELS:
        if label in path_lower:
            # Exclude if it's actually nonviolent
            if not any(nv in path_lower for nv in NONVIOLENT_PATH_LABELS):
                return ('violent', 'path', label.strip('/'), 'P1_PATH')

    # Check for nonviolent path labels
    for label in NONVIOLENT_PATH_LABELS:
        if label in path_lower:
            return ('nonviolent', 'path', label.strip('/'), 'P1_PATH')

    # ========================================================================
    # PRIORITY 2: FILENAME PATTERNS (95% confidence)
    # ========================================================================

    # Check for violent filename patterns
    for pattern in VIOLENT_FILENAME_PATTERNS:
        if pattern in filename_lower:
            # Special handling for v_ and nv_ patterns
            if pattern == 'v_':
                # Make sure it's V_### pattern, not part of another word
                if re.search(r'\bv_\d+', filename_lower):
                    return ('violent', 'filename', pattern, 'P2_FILENAME')
            else:
                # Exclude nonviolent patterns
                if not any(nv in filename_lower for nv in NONVIOLENT_FILENAME_PATTERNS):
                    return ('violent', 'filename', pattern, 'P2_FILENAME')

    # Check for nonviolent filename patterns
    for pattern in NONVIOLENT_FILENAME_PATTERNS:
        if pattern in filename_lower:
            # Special handling for nv_ pattern
            if pattern == 'nv_':
                if re.search(r'\bnv_\d+', filename_lower):
                    return ('nonviolent', 'filename', pattern, 'P2_FILENAME')
            else:
                return ('nonviolent', 'filename', pattern, 'P2_FILENAME')

    # ========================================================================
    # PRIORITY 3: STRONG KEYWORDS (85% confidence)
    # ========================================================================

    strong_violent = [kw for kw in STRONG_VIOLENT_KEYWORDS if kw in path_lower]
    strong_nonviolent = [kw for kw in STRONG_NONVIOLENT_KEYWORDS if kw in path_lower]

    if strong_violent and not strong_nonviolent:
        return ('violent', 'keyword', strong_violent[0], 'P3_KEYWORD')

    if strong_nonviolent and not strong_violent:
        return ('nonviolent', 'keyword', strong_nonviolent[0], 'P3_KEYWORD')

    # ========================================================================
    # AMBIGUOUS or UNKNOWN
    # ========================================================================

    if strong_violent and strong_nonviolent:
        return ('ambiguous', 'conflict', 'mixed_signals', 'CONFLICT')

    return ('unknown', 'none', 'no_clear_label', 'UNKNOWN')


def analyze_phase1():
    """Analyze phase1 directory with optimized categorization."""
    print("\n" + "="*80)
    print("OPTIMIZED PHASE1 ANALYSIS - Path & Filename Structure")
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
            'path': [],
            'filename': [],
            'keyword': [],
        },
        'nonviolent': {
            'path': [],
            'filename': [],
            'keyword': [],
        },
        'ambiguous': {'conflict': []},
        'unknown': {'none': []},
    }

    print("🔍 Analyzing with optimized structure-based classification...")
    for i, video_path in enumerate(all_videos):
        if (i + 1) % 1000 == 0:
            print(f"   Progress: {i+1:,}/{len(all_videos):,} ({(i+1)/len(all_videos)*100:.1f}%)...")

        category, confidence, matched, priority = optimized_categorize(video_path)
        results[category][confidence].append({
            'path': str(video_path),
            'matched': matched,
            'priority': priority
        })

    print(f"   Progress: {len(all_videos):,}/{len(all_videos):,} (100.0%) - Done!\n")

    # Calculate totals
    violent_total = sum(len(results['violent'][c]) for c in results['violent'])
    nonviolent_total = sum(len(results['nonviolent'][c]) for c in results['nonviolent'])
    ambiguous_total = len(results['ambiguous']['conflict'])
    unknown_total = len(results['unknown']['none'])

    # Print detailed results
    print("="*80)
    print("📊 CATEGORIZATION RESULTS")
    print("="*80)

    print(f"\n⚠️  VIOLENT: {violent_total:,} videos ({violent_total/len(all_videos)*100:.1f}%)")
    if results['violent']['path']:
        print(f"   - Path structure (P1):  {len(results['violent']['path']):6,} (99% confidence)")
    if results['violent']['filename']:
        print(f"   - Filename pattern (P2): {len(results['violent']['filename']):6,} (95% confidence)")
    if results['violent']['keyword']:
        print(f"   - Keyword match (P3):    {len(results['violent']['keyword']):6,} (85% confidence)")

    print(f"\n✅ NON-VIOLENT: {nonviolent_total:,} videos ({nonviolent_total/len(all_videos)*100:.1f}%)")
    if results['nonviolent']['path']:
        print(f"   - Path structure (P1):  {len(results['nonviolent']['path']):6,} (99% confidence)")
    if results['nonviolent']['filename']:
        print(f"   - Filename pattern (P2): {len(results['nonviolent']['filename']):6,} (95% confidence)")
    if results['nonviolent']['keyword']:
        print(f"   - Keyword match (P3):    {len(results['nonviolent']['keyword']):6,} (85% confidence)")

    if ambiguous_total:
        print(f"\n🔀 AMBIGUOUS: {ambiguous_total:,} videos ({ambiguous_total/len(all_videos)*100:.1f}%)")

    if unknown_total:
        print(f"\n❓ UNKNOWN: {unknown_total:,} videos ({unknown_total/len(all_videos)*100:.1f}%)")

    # Show sample paths
    print("\n" + "="*80)
    print("📝 SAMPLE VIDEOS BY DETECTION METHOD")
    print("="*80)

    # Path-based violent
    if results['violent']['path']:
        print("\n⚠️  VIOLENT (Path Structure - P1) - Sample 5:")
        for video in results['violent']['path'][:5]:
            print(f"   {Path(video['path']).name}")
            print(f"      Path contains: /{video['matched']}/")

    # Filename-based violent
    if results['violent']['filename']:
        print("\n⚠️  VIOLENT (Filename Pattern - P2) - Sample 5:")
        for video in results['violent']['filename'][:5]:
            print(f"   {Path(video['path']).name}")
            print(f"      Pattern: {video['matched']}")

    # Path-based nonviolent
    if results['nonviolent']['path']:
        print("\n✅ NON-VIOLENT (Path Structure - P1) - Sample 5:")
        for video in results['nonviolent']['path'][:5]:
            print(f"   {Path(video['path']).name}")
            print(f"      Path contains: /{video['matched']}/")

    # Filename-based nonviolent
    if results['nonviolent']['filename']:
        print("\n✅ NON-VIOLENT (Filename Pattern - P2) - Sample 5:")
        for video in results['nonviolent']['filename'][:5]:
            print(f"   {Path(video['path']).name}")
            print(f"      Pattern: {video['matched']}")

    # Check specific datasets
    print("\n" + "="*80)
    print("🔍 DATASET-SPECIFIC CHECKS")
    print("="*80)

    # RoadAccidents check
    road_accidents = [v for v in all_videos if 'roadaccident' in str(v).lower()]
    if road_accidents:
        print("\n🚗 Road Accidents (should be VIOLENT):")
        for ra in road_accidents[:3]:
            cat, conf, match, pri = optimized_categorize(ra)
            status = "✅" if cat == 'violent' else "❌"
            print(f"   {status} {ra.name} → {cat.upper()} ({conf})")

    # Real Life Violence dataset check
    rlv_violent = [v for v in all_videos[:100] if '/violence/' in str(v).lower()]
    rlv_nonviolent = [v for v in all_videos[:100] if '/nonviolence/' in str(v).lower()]

    if rlv_violent:
        print("\n📊 Real Life Violence Dataset - Violence folder:")
        v = rlv_violent[0]
        cat, conf, match, pri = optimized_categorize(v)
        status = "✅" if cat == 'violent' else "❌"
        print(f"   {status} {v.name} → {cat.upper()} ({conf})")

    if rlv_nonviolent:
        print("\n📊 Real Life Violence Dataset - NonViolence folder:")
        v = rlv_nonviolent[0]
        cat, conf, match, pri = optimized_categorize(v)
        status = "✅" if cat == 'nonviolent' else "❌"
        print(f"   {status} {v.name} → {cat.upper()} ({conf})")

    # XD-Violence dataset check
    xd_fighting = [v for v in all_videos[:100] if '_fighting' in str(v).lower()]
    xd_normal = [v for v in all_videos[:100] if '_normal' in str(v).lower()]

    if xd_fighting:
        print("\n📊 XD-Violence Dataset - Fighting videos:")
        v = xd_fighting[0]
        cat, conf, match, pri = optimized_categorize(v)
        status = "✅" if cat == 'violent' else "❌"
        print(f"   {status} {v.name} → {cat.upper()} ({conf})")

    if xd_normal:
        print("\n📊 XD-Violence Dataset - Normal videos:")
        v = xd_normal[0]
        cat, conf, match, pri = optimized_categorize(v)
        status = "✅" if cat == 'nonviolent' else "❌"
        print(f"   {status} {v.name} → {cat.upper()} ({conf})")

    if unknown_total:
        print("\n❓ UNKNOWN - Sample 5:")
        for video in results['unknown']['none'][:5]:
            rel_path = Path(video['path']).relative_to(PHASE1_DIR)
            print(f"   {rel_path}")

    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    confident = violent_total + nonviolent_total
    uncertain = ambiguous_total + unknown_total

    print(f"\n✅ Confident categorization: {confident:,} videos ({confident/len(all_videos)*100:.1f}%)")
    print(f"⚠️  Uncertain: {uncertain:,} videos ({uncertain/len(all_videos)*100:.1f}%)")

    # Detection method breakdown
    path_count = len(results['violent']['path']) + len(results['nonviolent']['path'])
    filename_count = len(results['violent']['filename']) + len(results['nonviolent']['filename'])
    keyword_count = len(results['violent']['keyword']) + len(results['nonviolent']['keyword'])

    if path_count > 0:
        print(f"\n🎯 PATH STRUCTURE (Best): {path_count:,} videos (99% accurate)")
    if filename_count > 0:
        print(f"🎯 FILENAME PATTERNS: {filename_count:,} videos (95% accurate)")
    if keyword_count > 0:
        print(f"🎯 KEYWORD MATCHING: {keyword_count:,} videos (85% accurate)")

    print("\n🎯 NEXT STEPS:")

    if confident > len(all_videos) * 0.95:
        print(f"   ✅ EXCELLENT! {confident/len(all_videos)*100:.1f}% can be auto-categorized")
        print(f"   → Your data has clear structure-based labels")
        print(f"   → Proceed with automatic organization")
        print(f"   → Command: python3 separate_phase1_optimized.py")
    elif confident > len(all_videos) * 0.8:
        print(f"   ✅ Good! {confident/len(all_videos)*100:.1f}% can be auto-categorized")
        print(f"   → Proceed with automatic organization")
    else:
        print(f"   ⚠️  Only {confident/len(all_videos)*100:.0f}% confidently categorized")
        print(f"   → Review unknown videos")

    print(f"\n⚖️  CLASS BALANCE:")
    if violent_total and nonviolent_total:
        ratio = min(violent_total, nonviolent_total) / max(violent_total, nonviolent_total)
        print(f"   Violent: {violent_total:,}")
        print(f"   Non-Violent: {nonviolent_total:,}")
        print(f"   Balance: {ratio*100:.1f}%")

        if ratio < 0.3:
            print(f"   ⚠️  SEVERE IMBALANCE! Need to collect more {('violent' if nonviolent_total > violent_total else 'non-violent')} data")
        elif ratio < 0.7:
            print(f"   ⚠️  Moderate imbalance - consider downsampling or collecting more data")
        else:
            print(f"   ✅ Good balance - ready for training!")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE - NO FILES MODIFIED")
    print("="*80)
    print()

    return results


if __name__ == "__main__":
    analyze_phase1()
