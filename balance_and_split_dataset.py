#!/usr/bin/env python3
"""
Balance and Split Dataset Script
Combines all downloaded videos and creates balanced train/val splits
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

class DatasetBalancer:
    def __init__(self,
                 base_dir="/workspace/violence_detection_mvp",
                 output_dir="/workspace/organized_dataset",
                 train_ratio=0.8):

        base = Path(base_dir)

        # Auto-detect download directories
        self.reddit_dir = base / "downloaded_reddit_videos"
        self.youtube_dir = base / "downloaded_youtube_videos"

        # Try multiple possible names for WorldStar
        possible_worldstar = [
            base / "downloaded_worldstar_videos",
            base / "downloaded_worldstar",
            base / "worldstar_videos"
        ]

        self.worldstar_dir = None
        for ws_path in possible_worldstar:
            if ws_path.exists():
                self.worldstar_dir = ws_path
                break

        if self.worldstar_dir is None:
            self.worldstar_dir = base / "downloaded_worldstar_videos"  # Default

        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio

        # Output structure
        self.train_violent = self.output_dir / "train" / "violent"
        self.train_nonviolent = self.output_dir / "train" / "nonviolent"
        self.val_violent = self.output_dir / "val" / "violent"
        self.val_nonviolent = self.output_dir / "val" / "nonviolent"

    def count_videos(self):
        """Count all videos from all sources"""
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                    DATASET VIDEO COUNTER                                   ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()

        # Show paths being checked
        print("🔍 Checking directories:")
        print(f"   Reddit:     {self.reddit_dir}")
        print(f"   YouTube:    {self.youtube_dir}")
        print(f"   WorldStar:  {self.worldstar_dir}")
        print()

        sources = {
            'Reddit': self.reddit_dir,
            'YouTube': self.youtube_dir,
            'WorldStar': self.worldstar_dir,
            'Current Train Violent': self.train_violent,
            'Current Train NonViolent': self.train_nonviolent,
            'Current Val Violent': self.val_violent,
            'Current Val NonViolent': self.val_nonviolent
        }

        total_count = 0

        for name, path in sources.items():
            if path.exists():
                videos = list(path.glob('*.mp4')) + list(path.glob('*.webm')) + list(path.glob('*.mkv'))
                count = len(videos)
                total_count += count

                # Calculate size
                total_size = sum(f.stat().st_size for f in videos if f.exists())
                size_gb = total_size / (1024**3)

                print(f"📁 {name:25} {count:6} videos ({size_gb:.2f} GB)")
            else:
                print(f"📁 {name:25}      0 videos (not found: {path})")

        print()
        print("="*80)
        print(f"📊 TOTAL VIDEOS COLLECTED: {total_count}")
        print("="*80)
        print()

        return total_count

    def create_balanced_split(self):
        """Create balanced train/val split from all sources"""
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                    BALANCE & SPLIT DATASET                                 ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()

        # Step 1: Collect all violent videos from all sources
        print("📹 Collecting violent videos from all sources...")
        violent_videos = []

        # Reddit violent (all are violent - fight videos)
        if self.reddit_dir.exists():
            reddit_vids = list(self.reddit_dir.glob('*.mp4'))
            violent_videos.extend(reddit_vids)
            print(f"   ✅ Reddit: {len(reddit_vids)} violent videos")

        # YouTube violent (all are violent - fight videos)
        if self.youtube_dir.exists():
            youtube_vids = list(self.youtube_dir.glob('*.mp4'))
            violent_videos.extend(youtube_vids)
            print(f"   ✅ YouTube: {len(youtube_vids)} violent videos")

        # WorldStar violent (all are violent - fight videos)
        if self.worldstar_dir.exists():
            worldstar_vids = list(self.worldstar_dir.glob('*.mp4'))
            violent_videos.extend(worldstar_vids)
            print(f"   ✅ WorldStar: {len(worldstar_vids)} violent videos")

        # Existing violent videos
        if self.train_violent.exists():
            existing_train = list(self.train_violent.glob('*.mp4'))
            violent_videos.extend(existing_train)
            print(f"   ✅ Existing train/violent: {len(existing_train)} videos")

        if self.val_violent.exists():
            existing_val = list(self.val_violent.glob('*.mp4'))
            violent_videos.extend(existing_val)
            print(f"   ✅ Existing val/violent: {len(existing_val)} videos")

        print()

        # Step 2: Collect all non-violent videos
        print("📹 Collecting non-violent videos...")
        nonviolent_videos = []

        if self.train_nonviolent.exists():
            existing_train_nv = list(self.train_nonviolent.glob('*.mp4'))
            nonviolent_videos.extend(existing_train_nv)
            print(f"   ✅ Existing train/nonviolent: {len(existing_train_nv)} videos")

        if self.val_nonviolent.exists():
            existing_val_nv = list(self.val_nonviolent.glob('*.mp4'))
            nonviolent_videos.extend(existing_val_nv)
            print(f"   ✅ Existing val/nonviolent: {len(existing_val_nv)} videos")

        print()
        print("="*80)
        print(f"📊 TOTAL VIOLENT: {len(violent_videos)}")
        print(f"📊 TOTAL NON-VIOLENT: {len(nonviolent_videos)}")
        print("="*80)
        print()

        # Step 3: Balance the dataset
        print("⚖️  Balancing dataset...")

        # Determine target count (use the smaller class size)
        target_count = min(len(violent_videos), len(nonviolent_videos))

        if target_count == 0:
            print("❌ ERROR: No videos found!")
            return

        print(f"   Target per class: {target_count} videos")
        print()

        # Shuffle and sample
        random.shuffle(violent_videos)
        random.shuffle(nonviolent_videos)

        violent_balanced = violent_videos[:target_count]
        nonviolent_balanced = nonviolent_videos[:target_count]

        # Step 4: Split into train/val
        print(f"📊 Splitting with {int(self.train_ratio * 100)}% train, {int((1-self.train_ratio) * 100)}% val...")

        train_size = int(target_count * self.train_ratio)

        violent_train = violent_balanced[:train_size]
        violent_val = violent_balanced[train_size:]

        nonviolent_train = nonviolent_balanced[:train_size]
        nonviolent_val = nonviolent_balanced[train_size:]

        print()
        print("="*80)
        print("📊 FINAL SPLIT:")
        print("="*80)
        print(f"   train/violent:     {len(violent_train)} videos")
        print(f"   train/nonviolent:  {len(nonviolent_train)} videos")
        print(f"   val/violent:       {len(violent_val)} videos")
        print(f"   val/nonviolent:    {len(nonviolent_val)} videos")
        print(f"   TOTAL:             {len(violent_train) + len(nonviolent_train) + len(violent_val) + len(nonviolent_val)} videos")
        print("="*80)
        print()

        # Step 5: Confirm before proceeding
        response = input("Proceed with copying files? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cancelled")
            return

        # Step 6: Create directory structure
        print()
        print("📁 Creating directory structure...")

        # Backup existing data first
        if self.output_dir.exists():
            backup_dir = Path(str(self.output_dir) + "_backup")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            print(f"   💾 Backing up existing data to: {backup_dir}")
            shutil.copytree(self.output_dir, backup_dir)

        # Clear and recreate
        for dir_path in [self.train_violent, self.train_nonviolent, self.val_violent, self.val_nonviolent]:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Clear existing files
            for f in dir_path.glob('*'):
                f.unlink()

        print("   ✅ Directories created")
        print()

        # Step 7: Move files (not copy - saves space!)
        print("📋 Moving files to organized structure...")

        print("   Moving train/violent...")
        for i, src in enumerate(violent_train, 1):
            dst = self.train_violent / f"violent_{i:05d}.mp4"
            shutil.move(str(src), str(dst))
            if i % 100 == 0:
                print(f"      Moved {i}/{len(violent_train)}")
        print(f"   ✅ train/violent: {len(violent_train)} videos")

        print("   Moving train/nonviolent...")
        for i, src in enumerate(nonviolent_train, 1):
            dst = self.train_nonviolent / f"nonviolent_{i:05d}.mp4"
            shutil.move(str(src), str(dst))
            if i % 100 == 0:
                print(f"      Moved {i}/{len(nonviolent_train)}")
        print(f"   ✅ train/nonviolent: {len(nonviolent_train)} videos")

        print("   Moving val/violent...")
        for i, src in enumerate(violent_val, 1):
            dst = self.val_violent / f"violent_{i:05d}.mp4"
            shutil.move(str(src), str(dst))
            if i % 20 == 0:
                print(f"      Moved {i}/{len(violent_val)}")
        print(f"   ✅ val/violent: {len(violent_val)} videos")

        print("   Moving val/nonviolent...")
        for i, src in enumerate(nonviolent_val, 1):
            dst = self.val_nonviolent / f"nonviolent_{i:05d}.mp4"
            shutil.move(str(src), str(dst))
            if i % 20 == 0:
                print(f"      Moved {i}/{len(nonviolent_val)}")
        print(f"   ✅ val/nonviolent: {len(nonviolent_val)} videos")

        print()
        print("="*80)
        print("✅ DATASET BALANCED AND ORGANIZED!")
        print("="*80)
        print()
        print(f"📁 Output directory: {self.output_dir}")
        print()
        print("📊 Final structure:")
        print(f"   {self.output_dir}/")
        print(f"   ├── train/")
        print(f"   │   ├── violent/     ({len(violent_train)} videos)")
        print(f"   │   └── nonviolent/  ({len(nonviolent_train)} videos)")
        print(f"   └── val/")
        print(f"       ├── violent/     ({len(violent_val)} videos)")
        print(f"       └── nonviolent/  ({len(nonviolent_val)} videos)")
        print()
        print(f"💾 Backup saved to: {self.output_dir}_backup")
        print()
        print("🚀 Ready for training!")
        print("="*80)


def main():
    import sys

    # Parse arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Usage:")
        print("  python3 balance_and_split_dataset.py count          # Just count videos")
        print("  python3 balance_and_split_dataset.py balance        # Balance and split")
        print()
        mode = input("Choose mode (count/balance): ").strip().lower()

    balancer = DatasetBalancer()

    if mode == 'count':
        balancer.count_videos()
    elif mode == 'balance':
        balancer.count_videos()
        print()
        balancer.create_balanced_split()
    else:
        print(f"❌ Unknown mode: {mode}")
        print("   Use 'count' or 'balance'")


if __name__ == "__main__":
    main()
