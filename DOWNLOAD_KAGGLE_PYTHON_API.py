#!/usr/bin/env python3
"""
Download Kaggle datasets using Python API directly (not CLI)
More reliable with better error handling + SAFE extraction for long filenames
"""

import os
from pathlib import Path
import shutil
import json
import zipfile
import hashlib

# Setup Kaggle credentials
kaggle_json = Path("kaggle.json")
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

if kaggle_json.exists():
    shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    print("✅ Kaggle credentials configured\n")

# Import Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi

def safe_extract_zip(zip_path, extract_to):
    """Extract zip with automatic renaming of long filenames"""
    print(f"Extracting {zip_path.name} with safe filename handling...")

    file_counter = {}  # Track renamed files per directory

    with zipfile.ZipFile(zip_path, 'r') as zf:
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

                # Use counter instead of hash for simplicity
                parent_key = str(parent_dirs)
                if parent_key not in file_counter:
                    file_counter[parent_key] = 1
                else:
                    file_counter[parent_key] += 1

                # New simple name: file_001.avi, file_002.avi, etc.
                new_filename = f"file_{file_counter[parent_key]:05d}{ext}"

                print(f"  Renaming long file: {filename[:50]}... -> {new_filename}")

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
                except OSError as e:
                    # If STILL too long (full path issue), use even shorter name
                    if 'File name too long' in str(e):
                        short_name = f"v_{file_counter.get(parent_key, 1):05d}{ext}"
                        target_path = extract_to / parent_dirs / short_name
                        print(f"  Using ultra-short name: {short_name}")
                        with zf.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                    else:
                        raise

    print(f"✓ Extracted successfully")

output_base = Path("/workspace/exact_datasets")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING KAGGLE DATASETS - PYTHON API")
print("="*80)
print()

# Initialize API
api = KaggleApi()
api.authenticate()
print("✅ Kaggle API authenticated\n")

results = []

# Dataset configurations
datasets = [
    {
        'name': 'RWF-2000',
        'owner': 'vulamnguyen',
        'dataset': 'rwf2000',
        'path': 'RWF2000'
    },
    {
        'name': 'RLVS+Hockey',
        'owner': 'inzilele',
        'dataset': 'rlvs-hockey-fight-dataset-mp4-files',
        'path': 'RLVS_Hockey'
    },
    {
        'name': 'UCF-Crime',
        'owner': 'odins0n',
        'dataset': 'ucf-crime-dataset',
        'path': 'UCF_Crime'
    }
]

for i, ds_config in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/3] {ds_config['name']}")
    print(f"Dataset: {ds_config['owner']}/{ds_config['dataset']}")
    print(f"{'='*80}")

    output_dir = output_base / ds_config['path']
    output_dir.mkdir(exist_ok=True)

    try:
        print("Downloading with Python API...")
        print("(This shows progress - may take several minutes)")

        # Download WITHOUT unzip (we'll extract manually with safe handling)
        api.dataset_download_files(
            f"{ds_config['owner']}/{ds_config['dataset']}",
            path=str(output_dir),
            unzip=False,
            quiet=False  # Show progress
        )

        print("\n✓ Download complete")

        # Extract with safe filename handling
        for zip_file in output_dir.glob('*.zip'):
            safe_extract_zip(zip_file, output_dir)
            zip_file.unlink()  # Delete zip after extraction

        print("Counting videos...")

        # Count videos
        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI']:
            video_count += len(list(output_dir.rglob(ext)))

        # Calculate size
        size_gb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / (1024**3)

        print(f"✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")

        results.append({
            'dataset': ds_config['name'],
            'status': 'success',
            'videos': video_count,
            'size_gb': round(size_gb, 2),
            'path': str(output_dir)
        })

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg[:200]}")

        results.append({
            'dataset': ds_config['name'],
            'status': 'error',
            'error': error_msg[:200]
        })

# ============================================================================
# UCF Crime Filtering (if downloaded successfully)
# ============================================================================

ucf_result = next((r for r in results if r['dataset'] == 'UCF-Crime' and r['status'] == 'success'), None)

if ucf_result:
    print("\n" + "="*80)
    print("FILTERING UCF CRIME - Abuse, Fighting, Assault, Normal only")
    print("="*80)

    ucf_dir = Path(ucf_result['path'])
    ucf_filtered = output_base / "UCF_Crime_filtered"
    ucf_filtered.mkdir(exist_ok=True)

    keep_categories = ['abuse', 'fighting', 'assault', 'normal']
    moved_count = 0

    print("Searching for matching categories...")

    for item in ucf_dir.rglob('*'):
        if item.is_dir():
            dir_name_lower = item.name.lower()

            if any(cat in dir_name_lower for cat in keep_categories):
                print(f"  Found: {item.name}")

                output_cat = ucf_filtered / item.name
                output_cat.mkdir(exist_ok=True)

                for video in item.rglob('*'):
                    if video.is_file() and video.suffix.lower() in ['.mp4', '.avi', '.mkv', '.webm', '.mov']:
                        dest = output_cat / video.name
                        shutil.copy(str(video), str(dest))
                        moved_count += 1

    print(f"✓ Copied {moved_count} videos to filtered dataset")

    # Count filtered
    filtered_count = sum(1 for _ in ucf_filtered.rglob('*.mp4')) + sum(1 for _ in ucf_filtered.rglob('*.avi'))
    filtered_size = sum(f.stat().st_size for f in ucf_filtered.rglob('*') if f.is_file()) / (1024**3)

    print(f"\nFiltered dataset: {filtered_count} videos ({filtered_size:.2f} GB)")
    print("Breakdown:")
    for subdir in ucf_filtered.iterdir():
        if subdir.is_dir():
            count = sum(1 for _ in subdir.rglob('*.mp4')) + sum(1 for _ in subdir.rglob('*.avi'))
            if count > 0:
                print(f"  {subdir.name}: {count} videos")

    # Update results
    for r in results:
        if r['dataset'] == 'UCF-Crime':
            r['filtered_videos'] = filtered_count
            r['filtered_size_gb'] = round(filtered_size, 2)
            r['filtered_path'] = str(ucf_filtered)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)
print()

successful = [r for r in results if r['status'] == 'success']
failed = [r for r in results if r['status'] != 'success']

total_videos = sum(r.get('videos', 0) for r in successful)
total_size = sum(r.get('size_gb', 0) for r in successful)

print(f"✅ Successful: {len(successful)}/3")
print(f"❌ Failed: {len(failed)}/3")
print()
print(f"📹 Total Videos: {total_videos:,}")
print(f"💾 Total Size: {total_size:.2f} GB")
print()

if successful:
    print("SUCCESSFUL DOWNLOADS:")
    for r in successful:
        print(f"  ✅ {r['dataset']}")
        print(f"     Videos: {r['videos']:,} ({r['size_gb']} GB)")
        print(f"     Path: {r['path']}")
        if 'filtered_path' in r:
            print(f"     Filtered: {r['filtered_videos']} videos ({r['filtered_size_gb']} GB)")
            print(f"     Filtered Path: {r['filtered_path']}")
        print()

if failed:
    print("FAILED DOWNLOADS:")
    for r in failed:
        print(f"  ❌ {r['dataset']}")
        print(f"     Error: {r.get('error', 'Unknown')}")
        print()

# Save report
report_file = output_base / "download_report.json"
with open(report_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"📄 Report: {report_file}")
print(f"📁 All datasets: {output_base}")
print("="*80)
