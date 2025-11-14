#!/usr/bin/env python3
"""
Show directory structure and identify which folders contain:
- Violence/Fight videos
- Non-Violence/Normal videos
- Unknown

Just show folder paths, not individual video files
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

# Violence keywords
VIOLENCE_KEYWORDS = [
    'fight', 'fighting', 'violence', 'violent', 'assault', 'abuse',
    'attack', 'punch', 'kick', 'hit', 'beat', 'aggression', 'aggressive',
    'combat', 'brawl', 'battle', 'struggle', 'anomaly'
]

NORMAL_KEYWORDS = [
    'normal', 'non', 'nonviolent', 'nonviolence', 'peaceful', 'safe',
    'neutral', 'regular', 'everyday', 'daily', 'routine'
]

# ============================================================================
# FUNCTIONS
# ============================================================================

def classify_directory(dir_path):
    """
    Classify directory based on its name
    Returns: 'violence', 'nonviolence', or 'unknown'
    """
    dir_name = dir_path.name.lower()
    full_path = str(dir_path).lower()

    # Check for violence keywords
    for keyword in VIOLENCE_KEYWORDS:
        if keyword in dir_name or keyword in full_path:
            return 'violence'

    # Check for normal keywords
    for keyword in NORMAL_KEYWORDS:
        if keyword in dir_name or keyword in full_path:
            return 'nonviolence'

    return 'unknown'

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("DIRECTORY STRUCTURE MAP")
print("Shows which folders contain violence/non-violence videos")
print("=" * 80)
print()

all_video_dirs = defaultdict(lambda: {'violence': [], 'nonviolence': [], 'unknown': []})

# Search each directory
for search_dir in SEARCH_DIRS:
    search_path = Path(search_dir)

    if not search_path.exists():
        continue

    print(f"🔍 Scanning: {search_dir}")

    # Find all directories that contain videos
    video_dirs = set()

    for ext in VIDEO_EXTENSIONS:
        for video_file in search_path.rglob(f"*{ext}"):
            video_dirs.add(video_file.parent)

    if not video_dirs:
        print(f"   No videos found")
        print()
        continue

    print(f"   Found {len(video_dirs)} directories with videos")

    # Classify each directory
    for video_dir in video_dirs:
        # Count videos in this directory
        video_count = sum(1 for ext in VIDEO_EXTENSIONS for _ in video_dir.glob(f"*{ext}"))

        category = classify_directory(video_dir)

        all_video_dirs[search_dir][category].append({
            'path': str(video_dir),
            'count': video_count,
            'relative': str(video_dir.relative_to(search_path))
        })

    print()

# ============================================================================
# DETAILED BREAKDOWN
# ============================================================================

print("=" * 80)
print("DIRECTORY MAP BY CATEGORY")
print("=" * 80)
print()

total_violence_dirs = 0
total_nonviolence_dirs = 0
total_unknown_dirs = 0

for search_dir in SEARCH_DIRS:
    if search_dir not in all_video_dirs:
        continue

    data = all_video_dirs[search_dir]

    if not data['violence'] and not data['nonviolence'] and not data['unknown']:
        continue

    print(f"\n{'='*80}")
    print(f"SOURCE: {search_dir}")
    print(f"{'='*80}")

    # Violence directories
    if data['violence']:
        print(f"\n🔴 VIOLENCE DIRECTORIES ({len(data['violence'])} folders):")
        print("-" * 80)
        for dir_info in sorted(data['violence'], key=lambda x: x['path']):
            print(f"  {dir_info['path']}")
            print(f"    └─ {dir_info['count']} videos")
        total_violence_dirs += len(data['violence'])

    # Non-violence directories
    if data['nonviolence']:
        print(f"\n🟢 NON-VIOLENCE DIRECTORIES ({len(data['nonviolence'])} folders):")
        print("-" * 80)
        for dir_info in sorted(data['nonviolence'], key=lambda x: x['path']):
            print(f"  {dir_info['path']}")
            print(f"    └─ {dir_info['count']} videos")
        total_nonviolence_dirs += len(data['nonviolence'])

    # Unknown directories
    if data['unknown']:
        print(f"\n⚪ UNKNOWN/UNCLEAR DIRECTORIES ({len(data['unknown'])} folders):")
        print("-" * 80)
        for dir_info in sorted(data['unknown'], key=lambda x: x['path']):
            print(f"  {dir_info['path']}")
            print(f"    └─ {dir_info['count']} videos")
        total_unknown_dirs += len(data['unknown'])

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"🔴 Violence directories:     {total_violence_dirs}")
print(f"🟢 Non-Violence directories: {total_nonviolence_dirs}")
print(f"⚪ Unknown directories:      {total_unknown_dirs}")
print(f"📁 Total directories:        {total_violence_dirs + total_nonviolence_dirs + total_unknown_dirs}")
print()

# ============================================================================
# SAVE TO FILE
# ============================================================================

output_dir = Path("/workspace/dataset_lists")
output_dir.mkdir(exist_ok=True)

# Save directory map
map_file = output_dir / "directory_map.txt"
with open(map_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DIRECTORY MAP - VIOLENCE vs NON-VIOLENCE\n")
    f.write("=" * 80 + "\n\n")

    for search_dir in SEARCH_DIRS:
        if search_dir not in all_video_dirs:
            continue

        data = all_video_dirs[search_dir]

        f.write(f"\nSOURCE: {search_dir}\n")
        f.write("=" * 80 + "\n")

        if data['violence']:
            f.write(f"\n🔴 VIOLENCE ({len(data['violence'])} folders):\n")
            for dir_info in sorted(data['violence'], key=lambda x: x['path']):
                f.write(f"  {dir_info['path']} ({dir_info['count']} videos)\n")

        if data['nonviolence']:
            f.write(f"\n🟢 NON-VIOLENCE ({len(data['nonviolence'])} folders):\n")
            for dir_info in sorted(data['nonviolence'], key=lambda x: x['path']):
                f.write(f"  {dir_info['path']} ({dir_info['count']} videos)\n")

        if data['unknown']:
            f.write(f"\n⚪ UNKNOWN ({len(data['unknown'])} folders):\n")
            for dir_info in sorted(data['unknown'], key=lambda x: x['path']):
                f.write(f"  {dir_info['path']} ({dir_info['count']} videos)\n")

print(f"✓ Directory map saved to: {map_file}")
print()

# Save simple lists
violence_dirs_file = output_dir / "violence_directories.txt"
with open(violence_dirs_file, 'w') as f:
    for search_dir, data in all_video_dirs.items():
        for dir_info in sorted(data['violence'], key=lambda x: x['path']):
            f.write(f"{dir_info['path']}\n")

nonviolence_dirs_file = output_dir / "nonviolence_directories.txt"
with open(nonviolence_dirs_file, 'w') as f:
    for search_dir, data in all_video_dirs.items():
        for dir_info in sorted(data['nonviolence'], key=lambda x: x['path']):
            f.write(f"{dir_info['path']}\n")

unknown_dirs_file = output_dir / "unknown_directories.txt"
with open(unknown_dirs_file, 'w') as f:
    for search_dir, data in all_video_dirs.items():
        for dir_info in sorted(data['unknown'], key=lambda x: x['path']):
            f.write(f"{dir_info['path']}\n")

print(f"✓ Violence directories: {violence_dirs_file}")
print(f"✓ Non-violence directories: {nonviolence_dirs_file}")
print(f"✓ Unknown directories: {unknown_dirs_file}")
print()

print("=" * 80)
print("COMPLETE")
print("=" * 80)
print()
print("View full map:")
print(f"  cat {map_file}")
print()
