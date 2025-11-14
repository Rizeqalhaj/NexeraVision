#!/usr/bin/env python3
"""
Extract downloaded Kaggle zip files with SAFE filename handling
"""

from pathlib import Path
import zipfile
import shutil

def safe_extract_zip(zip_path, extract_to):
    """Extract zip with automatic renaming of long filenames"""
    print(f"Extracting {zip_path.name}...")

    file_counter = {}  # Track renamed files per directory

    with zipfile.ZipFile(zip_path, 'r') as zf:
        total_files = len(zf.namelist())
        extracted = 0

        for member in zf.namelist():
            # Get the filename and parent dirs
            filename = Path(member).name
            parent_dirs = Path(member).parent

            # Check BYTE length (Unicode chars can be multiple bytes!)
            filename_bytes = len(filename.encode('utf-8'))

            # Linux filename limit is 255 BYTES, be safe with 200
            if filename_bytes > 200:
                # Keep extension
                ext = Path(filename).suffix

                # Use counter
                parent_key = str(parent_dirs)
                if parent_key not in file_counter:
                    file_counter[parent_key] = 1
                else:
                    file_counter[parent_key] += 1

                # New simple name: file_001.avi, file_002.avi, etc.
                new_filename = f"file_{file_counter[parent_key]:05d}{ext}"

                print(f"  Renaming: {filename[:30]}... -> {new_filename}")

                target_path = extract_to / parent_dirs / new_filename
            else:
                target_path = extract_to / member

            # Create parent directories
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract
            if not member.endswith('/'):
                try:
                    with zf.open(member) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    extracted += 1

                    # Progress
                    if extracted % 100 == 0:
                        print(f"  Progress: {extracted}/{total_files} files")

                except OSError as e:
                    # If STILL too long (full path issue), use even shorter name
                    if 'File name too long' in str(e):
                        short_name = f"v_{file_counter.get(parent_key, 1):05d}{ext}"
                        target_path = extract_to / parent_dirs / short_name
                        print(f"  Using ultra-short: {short_name}")
                        with zf.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        extracted += 1
                    else:
                        raise

    print(f"✓ Extracted {extracted} files successfully")

# ============================================================================
# EXTRACT ALL ZIP FILES
# ============================================================================

output_base = Path("/workspace/exact_datasets")

print("="*80)
print("EXTRACTING KAGGLE ZIP FILES")
print("="*80)
print()

# Find all zip files
zip_files = list(output_base.rglob('*.zip'))

if not zip_files:
    print("❌ No zip files found in /workspace/exact_datasets/")
    exit(1)

print(f"Found {len(zip_files)} zip file(s)\n")

for i, zip_file in enumerate(zip_files, 1):
    print(f"\n[{i}/{len(zip_files)}] {zip_file.name}")
    print("-" * 60)

    # Extract to same directory
    extract_to = zip_file.parent

    try:
        safe_extract_zip(zip_file, extract_to)

        # Delete zip after extraction
        zip_file.unlink()
        print(f"✓ Deleted {zip_file.name}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)[:200]}")

# ============================================================================
# COUNT VIDEOS
# ============================================================================

print("\n" + "="*80)
print("COUNTING VIDEOS")
print("="*80)
print()

for dataset_dir in output_base.iterdir():
    if dataset_dir.is_dir():
        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
            video_count += len(list(dataset_dir.rglob(ext)))

        if video_count > 0:
            size_gb = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file()) / (1024**3)
            print(f"  {dataset_dir.name}: {video_count} videos ({size_gb:.2f} GB)")

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
