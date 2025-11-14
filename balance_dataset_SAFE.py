#!/usr/bin/env python3
"""
Balance and Split Dataset - SAFE VERSION
ONLY MOVES FILES - NO DELETION - NO BACKUP
"""

import shutil
from pathlib import Path
import random

def balance_dataset():
    print("="*80)
    print("BALANCE & SPLIT DATASET - SAFE VERSION")
    print("ONLY MOVES FILES - NO DELETION")
    print("="*80)
    print()

    # Paths
    base = Path("/workspace/violence_detection_mvp")
    reddit_dir = base / "downloaded_reddit_videos"
    youtube_dir = base / "downloaded_youtube_videos"
    worldstar_dir = base / "downloaded_worldstar_videos"

    existing_train_v = Path("/workspace/organized_dataset/train/violent")
    existing_train_nv = Path("/workspace/organized_dataset/train/nonviolent")
    existing_val_v = Path("/workspace/organized_dataset/val/violent")
    existing_val_nv = Path("/workspace/organized_dataset/val/nonviolent")

    # Collect all violent videos
    violent_videos = []

    if reddit_dir.exists():
        reddit = list(reddit_dir.glob('*.mp4'))
        violent_videos.extend(reddit)
        print(f"Reddit: {len(reddit)} violent videos")

    if youtube_dir.exists():
        youtube = list(youtube_dir.glob('*.mp4'))
        violent_videos.extend(youtube)
        print(f"YouTube: {len(youtube)} violent videos")

    if worldstar_dir.exists():
        worldstar = list(worldstar_dir.glob('*.mp4'))
        violent_videos.extend(worldstar)
        print(f"WorldStar: {len(worldstar)} violent videos")

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

    # Create directories - NO DELETION!
    output_base = Path("/workspace/organized_dataset")

    train_v = output_base / "train" / "violent"
    train_nv = output_base / "train" / "nonviolent"
    val_v = output_base / "val" / "violent"
    val_nv = output_base / "val" / "nonviolent"

    # Just create directories
    for d in [train_v, train_nv, val_v, val_nv]:
        d.mkdir(parents=True, exist_ok=True)

    print()
    print("Moving files...")
    print()

    # Move train/violent
    print(f"Moving {len(violent_train)} to train/violent...")
    for i, src in enumerate(violent_train, 1):
        dst = train_v / f"violent_{i:05d}.mp4"
        if src != dst:  # Don't move if already in place
            shutil.move(str(src), str(dst))
        if i % 100 == 0:
            print(f"  Moved {i}/{len(violent_train)}")
    print(f"✅ train/violent: {len(violent_train)}")

    # Move train/nonviolent
    print(f"Moving {len(nonviolent_train)} to train/nonviolent...")
    for i, src in enumerate(nonviolent_train, 1):
        dst = train_nv / f"nonviolent_{i:05d}.mp4"
        if src != dst:
            shutil.move(str(src), str(dst))
        if i % 100 == 0:
            print(f"  Moved {i}/{len(nonviolent_train)}")
    print(f"✅ train/nonviolent: {len(nonviolent_train)}")

    # Move val/violent
    print(f"Moving {len(violent_val)} to val/violent...")
    for i, src in enumerate(violent_val, 1):
        dst = val_v / f"violent_{i:05d}.mp4"
        if src != dst:
            shutil.move(str(src), str(dst))
    print(f"✅ val/violent: {len(violent_val)}")

    # Move val/nonviolent
    print(f"Moving {len(nonviolent_val)} to val/nonviolent...")
    for i, src in enumerate(nonviolent_val, 1):
        dst = val_nv / f"nonviolent_{i:05d}.mp4"
        if src != dst:
            shutil.move(str(src), str(dst))
    print(f"✅ val/nonviolent: {len(nonviolent_val)}")

    print()
    print("="*80)
    print("✅ DATASET BALANCED!")
    print("="*80)
    print(f"Output: {output_base}")
    print("="*80)

if __name__ == "__main__":
    balance_dataset()
