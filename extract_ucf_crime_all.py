#!/usr/bin/env python3
"""
Extract all UCF Crime zip files with safe filename handling
"""

import zipfile
import shutil
from pathlib import Path
import sys

BASE_DIR = Path("/workspace/ucf_crime_dropbox")
OUTPUT_DIR = BASE_DIR / "extracted"
NORMAL_DIR = BASE_DIR / "normal_videos"

print("=" * 80)
print("UCF CRIME DATASET - EXTRACT ALL ZIP FILES")
print("=" * 80)
print()

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NORMAL_DIR.mkdir(parents=True, exist_ok=True)

def safe_extract_zip(zip_path, extract_to):
    """Extract zip with automatic renaming of long filenames"""
    file_counter = {}

    print(f"Extracting: {zip_path.name}")
    print(f"Size: {zip_path.stat().st_size / (1024**3):.2f} GB")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total_files = len(zf.namelist())
            print(f"Total files: {total_files}")

            for idx, member in enumerate(zf.namelist(), 1):
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{total_files} ({idx*100//total_files}%)")

                filename = Path(member).name
                parent_dirs = Path(member).parent

                # Check BYTE length (Unicode chars can be multiple bytes!)
                filename_bytes = len(filename.encode('utf-8'))

                # Linux filename limit is 255 BYTES, be safe with 200
                if filename_bytes > 200:
                    ext = Path(filename).suffix
                    parent_key = str(parent_dirs)

                    if parent_key not in file_counter:
                        file_counter[parent_key] = 1
                    else:
                        file_counter[parent_key] += 1

                    new_filename = f"file_{file_counter[parent_key]:05d}{ext}"
                    target_path = extract_to / parent_dirs / new_filename
                    print(f"  Renamed long filename: {filename[:50]}... → {new_filename}")
                else:
                    target_path = extract_to / member

                # Create parent directories
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                if not member.endswith('/'):
                    try:
                        with zf.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                    except OSError as e:
                        if 'File name too long' in str(e):
                            # Ultra-short fallback
                            short_name = f"v_{file_counter.get(parent_key, 1):05d}{ext}"
                            target_path = extract_to / parent_dirs / short_name
                            print(f"  Ultra-short rename: {short_name}")
                            with zf.open(member) as source, open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                        else:
                            raise

            print(f"✓ Extracted: {zip_path.name}")
            return True

    except Exception as e:
        print(f"❌ Error extracting {zip_path.name}: {e}")
        return False

# Find all zip files
zip_files = sorted(BASE_DIR.glob("*.zip"))

if not zip_files:
    print("❌ No zip files found in /workspace/ucf_crime_dropbox/")
    print()
    print("Expected files:")
    print("  - Anomaly-Detection-Dataset.zip")
    print("  - Anomaly-Videos-Part-1.zip")
    print("  - Anomaly-Videos-Part-2.zip")
    print("  - etc.")
    sys.exit(1)

print(f"Found {len(zip_files)} zip files:")
for zf in zip_files:
    size_gb = zf.stat().st_size / (1024**3)
    print(f"  - {zf.name} ({size_gb:.2f} GB)")
print()

# Extract all files
print("=" * 80)
print("EXTRACTING ALL FILES")
print("=" * 80)
print()

success_count = 0
for zip_file in zip_files:
    if safe_extract_zip(zip_file, OUTPUT_DIR):
        success_count += 1
    print()

print("=" * 80)
print("FILTERING NORMAL VIDEOS")
print("=" * 80)
print()

# Find all video files in extracted directories
video_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
all_videos = []

for ext in video_extensions:
    all_videos.extend(OUTPUT_DIR.rglob(f"*{ext}"))

print(f"Found {len(all_videos)} total videos")
print()

# Filter for normal/non-violence videos
normal_videos = []
for video in all_videos:
    # Check if parent directory contains "normal" or "non"
    parent_name = video.parent.name.lower()

    if 'normal' in parent_name or 'non' in parent_name:
        normal_videos.append(video)

print(f"Found {len(normal_videos)} normal/non-violence videos")
print()

# Copy normal videos to organized directory
print("Copying normal videos...")
for idx, video in enumerate(normal_videos, 1):
    if idx % 50 == 0:
        print(f"  Progress: {idx}/{len(normal_videos)}")

    # Preserve category structure
    category = video.parent.name
    category_dir = NORMAL_DIR / category
    category_dir.mkdir(exist_ok=True)

    target_path = category_dir / video.name
    shutil.copy2(video, target_path)

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print()
print(f"✓ Extracted {success_count}/{len(zip_files)} zip files")
print(f"✓ Normal videos: {len(normal_videos)}")
print(f"📁 Saved to: {NORMAL_DIR}")
print()

# Show directory structure
print("Directory structure:")
for category_dir in sorted(NORMAL_DIR.iterdir()):
    if category_dir.is_dir():
        count = len(list(category_dir.glob("*.*")))
        print(f"  {category_dir.name}: {count} videos")

print()
print("Next steps:")
print("  1. Verify videos: ls -lh /workspace/ucf_crime_dropbox/normal_videos/")
print("  2. Count videos: find /workspace/ucf_crime_dropbox/normal_videos/ -type f | wc -l")
print("  3. Delete extracted folder to save space: rm -rf /workspace/ucf_crime_dropbox/extracted/")
print()
