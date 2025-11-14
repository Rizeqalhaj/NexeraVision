#!/usr/bin/env python3
"""
Balance and Split Dataset - For Vast.ai
Run this on your Vast.ai instance
"""

import os
import shutil
from pathlib import Path
import random

# Find the downloaded directories automatically
def find_dirs():
    """Auto-detect download directories on Vast.ai"""
    base = Path("/workspace/violence_detection_mvp")

    dirs = {
        'reddit': base / "downloaded_reddit_videos",
        'youtube': base / "downloaded_youtube_videos",
        'worldstar': None
    }

    # Try to find WorldStar directory
    for possible in ["downloaded_worldstar_videos", "downloaded_worldstar", "worldstar_videos"]:
        ws_path = base / possible
        if ws_path.exists():
            dirs['worldstar'] = ws_path
            break

    return dirs

# Count all videos
def count_videos():
    dirs = find_dirs()

    print("="*80)
    print("VIDEO COUNTER")
    print("="*80)
    print()

    sources = {
        'Reddit': dirs['reddit'],
        'YouTube': dirs['youtube'],
        'WorldStar': dirs['worldstar'],
        'Train Violent': Path("/workspace/organized_dataset/train/violent"),
        'Train NonViolent': Path("/workspace/organized_dataset/train/nonviolent"),
        'Val Violent': Path("/workspace/organized_dataset/val/violent"),
        'Val NonViolent': Path("/workspace/organized_dataset/val/nonviolent")
    }

    total = 0
    for name, path in sources.items():
        if path and path.exists():
            videos = list(path.glob('*.mp4'))
            count = len(videos)
            total += count
            size_gb = sum(f.stat().st_size for f in videos) / (1024**3)
            print(f"{name:20} {count:6} videos ({size_gb:.2f} GB)")
        else:
            print(f"{name:20}      0 videos (not found)")

    print()
    print(f"TOTAL: {total} videos")
    print("="*80)
    return total

# Balance and split dataset
def balance_dataset():
    dirs = find_dirs()

    print("="*80)
    print("BALANCE & SPLIT DATASET")
    print("="*80)
    print()

    # Collect all violent videos
    violent_videos = []

    if dirs['reddit'].exists():
        reddit = list(dirs['reddit'].glob('*.mp4'))
        violent_videos.extend(reddit)
        print(f"Reddit: {len(reddit)} violent videos")

    if dirs['youtube'].exists():
        youtube = list(dirs['youtube'].glob('*.mp4'))
        violent_videos.extend(youtube)
        print(f"YouTube: {len(youtube)} violent videos")

    if dirs['worldstar'] and dirs['worldstar'].exists():
        worldstar = list(dirs['worldstar'].glob('*.mp4'))
        violent_videos.extend(worldstar)
        print(f"WorldStar: {len(worldstar)} violent videos")

    # Add existing violent videos
    existing_train_v = Path("/workspace/organized_dataset/train/violent")
    existing_val_v = Path("/workspace/organized_dataset/val/violent")

    if existing_train_v.exists():
        etv = list(existing_train_v.glob('*.mp4'))
        violent_videos.extend(etv)
        print(f"Existing train/violent: {len(etv)} videos")

    if existing_val_v.exists():
        evv = list(existing_val_v.glob('*.mp4'))
        violent_videos.extend(evv)
        print(f"Existing val/violent: {len(evv)} videos")

    print()

    # Collect all non-violent videos
    nonviolent_videos = []

    existing_train_nv = Path("/workspace/organized_dataset/train/nonviolent")
    existing_val_nv = Path("/workspace/organized_dataset/val/nonviolent")

    if existing_train_nv.exists():
        etnv = list(existing_train_nv.glob('*.mp4'))
        nonviolent_videos.extend(etnv)
        print(f"Existing train/nonviolent: {len(etnv)} videos")

    if existing_val_nv.exists():
        evnv = list(existing_val_nv.glob('*.mp4'))
        nonviolent_videos.extend(evnv)
        print(f"Existing val/nonviolent: {len(evnv)} videos")

    print()
    print("="*80)
    print(f"TOTAL VIOLENT: {len(violent_videos)}")
    print(f"TOTAL NON-VIOLENT: {len(nonviolent_videos)}")
    print("="*80)
    print()

    # Balance
    target_count = min(len(violent_videos), len(nonviolent_videos))
    print(f"Target per class: {target_count} videos")
    print()

    random.shuffle(violent_videos)
    random.shuffle(nonviolent_videos)

    violent_balanced = violent_videos[:target_count]
    nonviolent_balanced = nonviolent_videos[:target_count]

    # Split 80/20
    train_size = int(target_count * 0.8)

    violent_train = violent_balanced[:train_size]
    violent_val = violent_balanced[train_size:]

    nonviolent_train = nonviolent_balanced[:train_size]
    nonviolent_val = nonviolent_balanced[train_size:]

    print("FINAL SPLIT:")
    print(f"  train/violent:     {len(violent_train)}")
    print(f"  train/nonviolent:  {len(nonviolent_train)}")
    print(f"  val/violent:       {len(violent_val)}")
    print(f"  val/nonviolent:    {len(nonviolent_val)}")
    print(f"  TOTAL:             {len(violent_train) + len(nonviolent_train) + len(violent_val) + len(nonviolent_val)}")
    print()

    response = input("Proceed with MOVING files? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled")
        return

    # Create directories
    output_base = Path("/workspace/organized_dataset")

    # NO BACKUP - not enough space on Vast.ai
    print("⚠️  No backup will be created (disk space constraints)")

    # Create structure
    train_v = output_base / "train" / "violent"
    train_nv = output_base / "train" / "nonviolent"
    val_v = output_base / "val" / "violent"
    val_nv = output_base / "val" / "nonviolent"

    # Just create directories - DON'T DELETE ANYTHING!
    for d in [train_v, train_nv, val_v, val_nv]:
        d.mkdir(parents=True, exist_ok=True)

    print()
    print("Moving files...")

    # Move train/violent
    for i, src in enumerate(violent_train, 1):
        dst = train_v / f"violent_{i:05d}.mp4"
        shutil.move(str(src), str(dst))
        if i % 100 == 0:
            print(f"  Moved {i}/{len(violent_train)} train/violent")
    print(f"✅ train/violent: {len(violent_train)}")

    # Move train/nonviolent
    for i, src in enumerate(nonviolent_train, 1):
        dst = train_nv / f"nonviolent_{i:05d}.mp4"
        shutil.move(str(src), str(dst))
        if i % 100 == 0:
            print(f"  Moved {i}/{len(nonviolent_train)} train/nonviolent")
    print(f"✅ train/nonviolent: {len(nonviolent_train)}")

    # Move val/violent
    for i, src in enumerate(violent_val, 1):
        dst = val_v / f"violent_{i:05d}.mp4"
        shutil.move(str(src), str(dst))
    print(f"✅ val/violent: {len(violent_val)}")

    # Move val/nonviolent
    for i, src in enumerate(nonviolent_val, 1):
        dst = val_nv / f"nonviolent_{i:05d}.mp4"
        shutil.move(str(src), str(dst))
    print(f"✅ val/nonviolent: {len(nonviolent_val)}")

    print()
    print("="*80)
    print("✅ DATASET BALANCED!")
    print("="*80)
    print(f"Output: {output_base}")
    print(f"Backup: {output_base}_backup")
    print("="*80)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Usage:")
        print("  python3 VASTAI_balance_dataset.py count")
        print("  python3 VASTAI_balance_dataset.py balance")
        print()
        mode = input("Mode (count/balance): ").strip().lower()

    if mode == 'count':
        count_videos()
    elif mode == 'balance':
        count_videos()
        print()
        balance_dataset()
    else:
        print(f"Unknown mode: {mode}")
