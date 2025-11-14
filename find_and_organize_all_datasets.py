#!/usr/bin/env python3
"""
Find ALL video files and list their paths
NO classification, NO splitting, NO organizing
Just show me what files exist and where they are
"""

from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
SEARCH_DIRS = [
    "/workspace/datasets",
    "/workspace/exact_datasets",
    "/workspace/organized_dataset",
    "/workspace/ucf_crime_dropbox",
    "/workspace/youtube_nonviolence_videos",
    "/workspace/youtube_long_videos",
    "/workspace/nonviolence_clips_from_long",
]

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("FIND ALL VIDEO FILES - SIMPLE LIST")
print("=" * 80)
print()

all_videos = []
stats_by_dir = defaultdict(lambda: {'count': 0, 'size_mb': 0})

print("Searching directories...")
print()

# Search each directory
for search_dir in SEARCH_DIRS:
    search_path = Path(search_dir)

    if not search_path.exists():
        print(f"⚠️  Not found: {search_dir}")
        continue

    print(f"🔍 Scanning: {search_dir}")

    # Find all video files recursively
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(search_path.rglob(f"*{ext}"))

    print(f"   Found {len(video_files)} video files")

    # Collect info
    for video_file in video_files:
        try:
            size_mb = video_file.stat().st_size / (1024 * 1024)
            all_videos.append({
                'path': str(video_file),
                'size_mb': size_mb,
                'source_dir': search_dir
            })
            stats_by_dir[search_dir]['count'] += 1
            stats_by_dir[search_dir]['size_mb'] += size_mb
        except:
            pass

    print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

total_videos = len(all_videos)
total_size_mb = sum(v['size_mb'] for v in all_videos)
total_size_gb = total_size_mb / 1024

print(f"Total videos found: {total_videos:,}")
print(f"Total size: {total_size_gb:.2f} GB")
print()

print("By Directory:")
for search_dir, stats in sorted(stats_by_dir.items()):
    print(f"  {search_dir}")
    print(f"    Count: {stats['count']:,} videos")
    print(f"    Size:  {stats['size_mb']/1024:.2f} GB")
    print()

# ============================================================================
# SAVE ALL PATHS
# ============================================================================

output_dir = Path("/workspace/dataset_lists")
output_dir.mkdir(exist_ok=True)

# Save ALL video paths to single file
all_paths_file = output_dir / "all_video_paths.txt"
with open(all_paths_file, 'w') as f:
    for video in sorted(all_videos, key=lambda x: x['path']):
        f.write(f"{video['path']}\n")

print("=" * 80)
print("FILE PATHS SAVED")
print("=" * 80)
print()
print(f"✓ All {total_videos:,} video paths saved to:")
print(f"  {all_paths_file}")
print()

# Save paths grouped by source directory
for search_dir in SEARCH_DIRS:
    if search_dir in stats_by_dir:
        videos_in_dir = [v for v in all_videos if v['source_dir'] == search_dir]

        if videos_in_dir:
            dir_name = Path(search_dir).name
            dir_list_file = output_dir / f"paths_{dir_name}.txt"

            with open(dir_list_file, 'w') as f:
                for video in sorted(videos_in_dir, key=lambda x: x['path']):
                    f.write(f"{video['path']}\n")

            print(f"✓ {len(videos_in_dir):,} paths from {dir_name}:")
            print(f"  {dir_list_file}")

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print()
print("To view all paths:")
print(f"  cat {all_paths_file}")
print()
print("To count videos:")
print(f"  wc -l {all_paths_file}")
print()
