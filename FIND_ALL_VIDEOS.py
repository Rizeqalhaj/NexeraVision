#!/usr/bin/env python3
"""
Find ALL videos on the system - emergency recovery
"""
import os
from pathlib import Path

print("="*80)
print("SEARCHING FOR ALL VIDEOS ON SYSTEM")
print("="*80)
print()

# Search entire workspace
print("Searching /workspace for all .mp4 files...")
print()

all_videos = {}

for root, dirs, files in os.walk("/workspace"):
    # Skip some directories to speed up
    dirs[:] = [d for d in dirs if not d.startswith('.')]

    for file in files:
        if file.endswith('.mp4'):
            full_path = os.path.join(root, file)
            directory = os.path.dirname(full_path)

            if directory not in all_videos:
                all_videos[directory] = []
            all_videos[directory].append(file)

print("="*80)
print("FOUND VIDEOS IN:")
print("="*80)
print()

total = 0
for directory, files in sorted(all_videos.items()):
    count = len(files)
    total += count

    # Calculate size
    size = 0
    for f in files:
        try:
            size += os.path.getsize(os.path.join(directory, f))
        except:
            pass
    size_gb = size / (1024**3)

    print(f"{directory}")
    print(f"  Videos: {count}")
    print(f"  Size: {size_gb:.2f} GB")
    print()

print("="*80)
print(f"TOTAL: {total} videos found")
print("="*80)
