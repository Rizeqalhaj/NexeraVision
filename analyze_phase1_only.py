#!/usr/bin/env python3
"""
Phase1 Analysis Only - No file operations
Shows violent vs non-violent breakdown without copying anything
"""

import os
from pathlib import Path
from collections import defaultdict
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE1_DIR = "/workspace/datasets/phase1"

# Violent keywords
VIOLENT_KEYWORDS = [
    # Fight-related
    'fight', 'fighting', 'fighter', 'brawl', 'combat', 'battle',
    'punch', 'punching', 'kick', 'kicking', 'hit', 'hitting', 'slap',
    'beat', 'beating', 'attack', 'assault', 'aggression', 'aggressive',

    # Martial arts
    'ufc', 'mma', 'boxing', 'boxer', 'kickbox', 'muay thai',
    'karate', 'judo', 'jujitsu', 'wrestling', 'wrestle', 'grappling',
    'martial', 'sparring', 'knockout', ' ko ', 'k.o',

    # Street violence
    'street fight', 'streetfight', 'bully', 'gang', 'mob',
    'riot', 'clash', 'violence', 'violent', 'brutal', 'bloody',

    # Violent actions
    'slam', 'smash', 'knock', 'takedown', 'submission',
    'choke', 'strangle',

    # Accidents and injuries (MOVED FROM AMBIGUOUS)
    'accident', 'crash', 'collision', 'wreck', 'smash up',
    'injury', 'hurt', 'wounded', 'bleeding', 'fatal',
    'road accident', 'roadaccident', 'car crash', 'vehicle crash',
    'train crash', 'plane crash', 'motorcycle accident',

    # Reddit violence indicators
    'r_fight', 'r_street', 'r_brutal', 'r_real',
    'r_femalemma', 'r_publicfreakout', 'freakout',

    # Generic violence
    'freak out', 'altercation', 'scuffle',
]

# Non-violent keywords (REMOVED GENERIC TERMS: work, working, road)
NONVIOLENT_KEYWORDS = [
    # Normal activities
    'walk', 'walking', 'sit', 'sitting', 'stand', 'standing',
    'talk', 'talking', 'conversation', 'chat', 'discuss',
    'shop', 'shopping', 'store', 'mall', 'market',
    'office', 'desk', 'meeting', 'conference',

    # CCTV/Surveillance
    'cctv', 'surveillance', 'camera', 'security', 'monitor',
    'parking', 'lobby', 'entrance', 'exit', 'corridor',
    'hallway', 'street view', 'pedestrian',

    # Daily life (REMOVED: road, traffic)
    'daily', 'normal', 'routine', 'activity', 'regular',
    'crowd', 'people', 'public', 'passerby', 'bystander',

    # Safe activities
    'dance', 'dancing', 'play', 'playing', 'game', 'sport',
    'exercise', 'run', 'running', 'jog', 'jogging',
    'eat', 'eating', 'drink', 'drinking', 'meal', 'food',

    # Positive
    'smile', 'laugh', 'happy', 'joy', 'celebrate', 'party',
    'wholesome', 'peaceful', 'calm', 'quiet', 'safe',
]

def categorize_video(file_path: str) -> tuple:
    """
    Categorize based on path and filename.
    Returns: (category, confidence, matched_keywords)
    """
    path_lower = str(file_path).lower()

    violent_matches = [kw for kw in VIOLENT_KEYWORDS if kw in path_lower]
    nonviolent_matches = [kw for kw in NONVIOLENT_KEYWORDS if kw in path_lower]

    # Strong indicators
    if violent_matches and not nonviolent_matches:
        if len(violent_matches) >= 2:
            return 'violent', 'high', violent_matches
        return 'violent', 'medium', violent_matches

    if nonviolent_matches and not violent_matches:
        if len(nonviolent_matches) >= 2:
            return 'nonviolent', 'high', nonviolent_matches
        return 'nonviolent', 'medium', nonviolent_matches

    if len(violent_matches) > len(nonviolent_matches):
        return 'violent', 'low', violent_matches

    if len(nonviolent_matches) > len(violent_matches):
        return 'nonviolent', 'low', nonviolent_matches

    if violent_matches and nonviolent_matches:
        return 'ambiguous', 'low', violent_matches + nonviolent_matches

    return 'unknown', 'none', []


def analyze_phase1():
    """Analyze phase1 directory."""
    print("\n" + "="*80)
    print("PHASE1 ANALYSIS - VIOLENT VS NON-VIOLENT")
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
        'violent': {'high': [], 'medium': [], 'low': []},
        'nonviolent': {'high': [], 'medium': [], 'low': []},
        'ambiguous': {'low': []},
        'unknown': {'none': []},
    }

    print("🔍 Analyzing videos...")
    for i, video_path in enumerate(all_videos):
        if (i + 1) % 1000 == 0:
            print(f"   Progress: {i+1:,}/{len(all_videos):,} ({(i+1)/len(all_videos)*100:.1f}%)...")

        category, confidence, keywords = categorize_video(video_path)
        results[category][confidence].append({
            'path': str(video_path),
            'keywords': keywords
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
    if results['violent']['high']:
        print(f"   - High confidence:   {len(results['violent']['high']):6,} (strong fight/violence indicators)")
    if results['violent']['medium']:
        print(f"   - Medium confidence: {len(results['violent']['medium']):6,} (1 violence indicator)")
    if results['violent']['low']:
        print(f"   - Low confidence:    {len(results['violent']['low']):6,} (weak indicators)")

    print(f"\n✅ NON-VIOLENT: {nonviolent_total:,} videos ({nonviolent_total/len(all_videos)*100:.1f}%)")
    if results['nonviolent']['high']:
        print(f"   - High confidence:   {len(results['nonviolent']['high']):6,} (strong normal activity indicators)")
    if results['nonviolent']['medium']:
        print(f"   - Medium confidence: {len(results['nonviolent']['medium']):6,} (1 normal indicator)")
    if results['nonviolent']['low']:
        print(f"   - Low confidence:    {len(results['nonviolent']['low']):6,} (weak indicators)")

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

    if results['violent']['high']:
        print("\n⚠️  VIOLENT (High Confidence) - Sample 5:")
        for video in results['violent']['high'][:5]:
            keywords = ', '.join(video['keywords'][:3])
            print(f"   {Path(video['path']).name}")
            print(f"      Keywords: {keywords}")

    if results['nonviolent']['high']:
        print("\n✅ NON-VIOLENT (High Confidence) - Sample 5:")
        for video in results['nonviolent']['high'][:5]:
            keywords = ', '.join(video['keywords'][:3])
            print(f"   {Path(video['path']).name}")
            print(f"      Keywords: {keywords}")

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

    print("\n🎯 NEXT STEPS:")

    if confident > len(all_videos) * 0.7:
        print(f"   ✅ Good! {confident/len(all_videos)*100:.0f}% can be auto-categorized")
        print(f"   → Proceed with automatic organization")
        print(f"   → Command: python3 separate_phase1_videos.py")
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
