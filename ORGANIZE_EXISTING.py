#!/usr/bin/env python3
"""
Organize existing videos into proper train/val split
Uses what you already have + downloaded videos
"""

import shutil
from pathlib import Path
import random

print("="*80)
print("ORGANIZE DATASET - USING EXISTING VIDEOS")
print("="*80)
print()

# Collect all violent videos
violent_videos = []

# From existing test/violent
test_v = Path("/workspace/organized_dataset/test/violent")
if test_v.exists():
    tv = list(test_v.glob('*.mp4'))
    violent_videos.extend(tv)
    print(f"Existing test/violent: {len(tv)} videos")

# From downloaded sources
reddit_dir = Path("/workspace/violence_detection_mvp/downloaded_reddit_videos")
youtube_dir = Path("/workspace/violence_detection_mvp/downloaded_youtube_videos")
worldstar_dir = Path("/workspace/violence_detection_mvp/downloaded_worldstar_videos")

if reddit_dir.exists():
    reddit = list(reddit_dir.glob('*.mp4'))
    violent_videos.extend(reddit)
    print(f"Downloaded Reddit: {len(reddit)} violent videos")

if youtube_dir.exists():
    youtube = list(youtube_dir.glob('*.mp4'))
    violent_videos.extend(youtube)
    print(f"Downloaded YouTube: {len(youtube)} violent videos")

if worldstar_dir.exists():
    worldstar = list(worldstar_dir.glob('*.mp4'))
    violent_videos.extend(worldstar)
    print(f"Downloaded WorldStar: {len(worldstar)} violent videos")

print()

# Collect all non-violent videos
nonviolent_videos = []

test_nv = Path("/workspace/organized_dataset/test/nonviolent")
if test_nv.exists():
    tnv = list(test_nv.glob('*.mp4'))
    nonviolent_videos.extend(tnv)
    print(f"Existing test/nonviolent: {len(tnv)} videos")

train_nv = Path("/workspace/organized_dataset/train/nonviolent")
if train_nv.exists():
    trnv = list(train_nv.glob('*.mp4'))
    nonviolent_videos.extend(trnv)
    print(f"Existing train/nonviolent: {len(trnv)} videos")

print()
print("="*80)
print(f"TOTAL VIOLENT: {len(violent_videos)}")
print(f"TOTAL NON-VIOLENT: {len(nonviolent_videos)}")
print("="*80)
print()

# Balance
target_count = min(len(violent_videos), len(nonviolent_videos))
print(f"Balanced count per class: {target_count} videos")
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

print("FINAL SPLIT (80/20):")
print(f"  train/violent:     {len(violent_train)}")
print(f"  train/nonviolent:  {len(nonviolent_train)}")
print(f"  val/violent:       {len(violent_val)}")
print(f"  val/nonviolent:    {len(nonviolent_val)}")
print(f"  TOTAL:             {len(violent_train) + len(nonviolent_train) + len(violent_val) + len(nonviolent_val)}")
print()

response = input("Organize into /workspace/organized_dataset? (yes/no): ")
if response.lower() != 'yes':
    print("Cancelled")
    exit()

# Create clean structure
output = Path("/workspace/organized_dataset_NEW")

train_v = output / "train" / "violent"
train_nv = output / "train" / "nonviolent"
val_v = output / "val" / "violent"
val_nv = output / "val" / "nonviolent"

for d in [train_v, train_nv, val_v, val_nv]:
    d.mkdir(parents=True, exist_ok=True)

print()
print("Moving files to new structure...")
print()

# Move train/violent
for i, src in enumerate(violent_train, 1):
    dst = train_v / f"violent_{i:05d}.mp4"
    shutil.move(str(src), str(dst))
    if i % 500 == 0:
        print(f"  train/violent: {i}/{len(violent_train)}")
print(f"✅ train/violent: {len(violent_train)}")

# Move train/nonviolent
for i, src in enumerate(nonviolent_train, 1):
    dst = train_nv / f"nonviolent_{i:05d}.mp4"
    shutil.move(str(src), str(dst))
    if i % 500 == 0:
        print(f"  train/nonviolent: {i}/{len(nonviolent_train)}")
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
print("✅ ORGANIZED!")
print("="*80)
print(f"New dataset: {output}")
print()
print("Next step: Rename organized_dataset_NEW to organized_dataset")
print("  mv /workspace/organized_dataset /workspace/organized_dataset_old")
print("  mv /workspace/organized_dataset_NEW /workspace/organized_dataset")
print("="*80)
