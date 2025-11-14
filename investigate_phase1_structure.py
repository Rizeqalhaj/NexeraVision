#!/usr/bin/env python3
"""
Investigate actual phase1 directory structure and naming patterns
to build accurate classifier based on real filenames
"""

import os
from pathlib import Path
from collections import defaultdict
import re

PHASE1_DIR = "/workspace/datasets/phase1"

def analyze_structure():
    """Analyze actual directory and filename patterns."""
    print("="*80)
    print("PHASE1 STRUCTURE INVESTIGATION")
    print("="*80)
    print(f"\nScanning: {PHASE1_DIR}\n")

    phase1_path = Path(PHASE1_DIR)

    if not phase1_path.exists():
        print(f"❌ ERROR: {PHASE1_DIR} not found")
        print("\n💡 TIP: Update PHASE1_DIR variable to your actual path")
        return

    # Find all videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    all_videos = []

    print("📁 Finding all videos...")
    for ext in video_extensions:
        videos = list(phase1_path.rglob(f'*{ext}'))
        all_videos.extend(videos)

    print(f"✅ Found {len(all_videos):,} videos\n")

    if len(all_videos) == 0:
        print("⚠️  No videos found!")
        return

    # Analyze directory structure
    print("="*80)
    print("📊 DIRECTORY STRUCTURE")
    print("="*80)

    subdirs = defaultdict(int)
    for video in all_videos:
        # Get relative path from phase1
        rel_path = video.relative_to(phase1_path)
        if len(rel_path.parts) > 1:
            subdir = rel_path.parts[0]
            subdirs[subdir] += 1

    if subdirs:
        print(f"\n📂 Found {len(subdirs)} subdirectories:\n")
        for subdir, count in sorted(subdirs.items(), key=lambda x: x[1], reverse=True):
            print(f"   {subdir:50s} → {count:6,} videos")
    else:
        print("\n📂 All videos are in root directory (no subdirectories)")

    # Analyze filename patterns
    print("\n" + "="*80)
    print("📝 FILENAME PATTERNS")
    print("="*80)

    # Look for explicit labels in filenames
    patterns = {
        'violent': 0,
        'violence': 0,
        'nonviolent': 0,
        'non_violent': 0,
        'non-violent': 0,
        'normal': 0,
        'shooting': 0,
        'fight': 0,
        'accident': 0,
        'crash': 0,
        'road': 0,
        'cctv': 0,
        'surveillance': 0,
    }

    pattern_examples = defaultdict(list)

    for video in all_videos:
        filename_lower = video.name.lower()
        path_lower = str(video).lower()

        for pattern in patterns.keys():
            if pattern in filename_lower or pattern in path_lower:
                patterns[pattern] += 1
                if len(pattern_examples[pattern]) < 5:
                    pattern_examples[pattern].append(video.name)

    print("\n🔍 Pattern Occurrence Counts:\n")
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"   {pattern:20s} → {count:6,} videos")

    # Show examples
    print("\n" + "="*80)
    print("📋 SAMPLE FILENAMES BY PATTERN")
    print("="*80)

    for pattern in ['violent', 'nonviolent', 'non_violent', 'normal', 'shooting', 'fight', 'accident']:
        if pattern in pattern_examples and pattern_examples[pattern]:
            print(f"\n🔹 '{pattern}' pattern - Sample 5:")
            for example in pattern_examples[pattern][:5]:
                print(f"   {example}")

    # Show random samples
    print("\n" + "="*80)
    print("📋 RANDOM SAMPLE FILENAMES (First 30)")
    print("="*80)
    print()

    import random
    sample_size = min(30, len(all_videos))
    samples = random.sample(all_videos, sample_size)

    for i, video in enumerate(samples, 1):
        # Show relative path from phase1
        rel_path = video.relative_to(phase1_path)
        print(f"{i:2d}. {rel_path}")

    # Analyze path structure
    print("\n" + "="*80)
    print("🗂️  PATH STRUCTURE ANALYSIS")
    print("="*80)

    path_patterns = {
        'contains_violent': [],
        'contains_nonviolent': [],
        'contains_normal': [],
        'contains_shooting': [],
        'contains_fight': [],
        'no_clear_indicator': []
    }

    for video in all_videos[:100]:  # Check first 100
        path_lower = str(video).lower()

        if 'violent' in path_lower and 'non' not in path_lower:
            if len(path_patterns['contains_violent']) < 3:
                path_patterns['contains_violent'].append(str(video.relative_to(phase1_path)))
        elif 'nonviolent' in path_lower or 'non_violent' in path_lower or 'non-violent' in path_lower:
            if len(path_patterns['contains_nonviolent']) < 3:
                path_patterns['contains_nonviolent'].append(str(video.relative_to(phase1_path)))
        elif 'normal' in path_lower:
            if len(path_patterns['contains_normal']) < 3:
                path_patterns['contains_normal'].append(str(video.relative_to(phase1_path)))
        elif 'shooting' in path_lower:
            if len(path_patterns['contains_shooting']) < 3:
                path_patterns['contains_shooting'].append(str(video.relative_to(phase1_path)))
        elif 'fight' in path_lower:
            if len(path_patterns['contains_fight']) < 3:
                path_patterns['contains_fight'].append(str(video.relative_to(phase1_path)))
        else:
            if len(path_patterns['no_clear_indicator']) < 3:
                path_patterns['no_clear_indicator'].append(str(video.relative_to(phase1_path)))

    print("\n📍 Sample paths by category:\n")
    for category, examples in path_patterns.items():
        if examples:
            print(f"\n{category.replace('_', ' ').title()}:")
            for example in examples:
                print(f"   {example}")

    print("\n" + "="*80)
    print("✅ INVESTIGATION COMPLETE")
    print("="*80)
    print("\n💡 RECOMMENDATION:")
    print("   Based on this analysis, we can create a smart classifier that:")
    print("   1. Checks directory names first (violent/, nonviolent/, normal/)")
    print("   2. Checks explicit labels in filenames (violent, nonviolent, shooting, etc.)")
    print("   3. Falls back to keyword matching for unclear cases")
    print()


if __name__ == "__main__":
    analyze_structure()
