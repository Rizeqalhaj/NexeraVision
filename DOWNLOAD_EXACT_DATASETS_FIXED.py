#!/usr/bin/env python3
"""
Download EXACT Kaggle datasets with PROGRESS indicators
"""

import subprocess
import os
from pathlib import Path
import shutil
import json
import sys

# Setup Kaggle credentials
kaggle_json = Path("kaggle.json")
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

if kaggle_json.exists():
    shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    print("✅ Kaggle credentials configured\n")
else:
    print("❌ ERROR: kaggle.json not found!")
    exit(1)

output_base = Path("/workspace/exact_datasets")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING EXACT KAGGLE DATASETS - WITH PROGRESS")
print("="*80)
print()

results = []

# ============================================================================
# 1. RWF-2000
# ============================================================================
print("\n" + "="*80)
print("1. RWF-2000 Dataset")
print("Dataset: vulamnguyen/rwf2000")
print("="*80)

rwf_dir = output_base / "RWF2000"
rwf_dir.mkdir(exist_ok=True)

try:
    # Download WITHOUT --unzip first (shows progress)
    cmd = [
        'kaggle', 'datasets', 'download',
        '-d', 'vulamnguyen/rwf2000',
        '-p', str(rwf_dir)
    ]

    print("Downloading (this may take several minutes, watching for progress)...")

    # Run with live output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()

    process.wait()

    if process.returncode == 0:
        # Now unzip
        print("\nDownload complete! Extracting...")

        for zipfile in rwf_dir.glob('*.zip'):
            print(f"Unzipping {zipfile.name}...")
            subprocess.run(['unzip', '-q', str(zipfile), '-d', str(rwf_dir)])
            zipfile.unlink()
            print("✓ Extracted")

        # Count videos
        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI']:
            video_count += len(list(rwf_dir.rglob(ext)))

        size_gb = sum(f.stat().st_size for f in rwf_dir.rglob('*') if f.is_file()) / (1024**3)

        print(f"\n✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
        results.append({
            'dataset': 'RWF-2000',
            'status': 'success',
            'videos': video_count,
            'size_gb': round(size_gb, 2),
            'path': str(rwf_dir)
        })
    else:
        print(f"\n❌ FAILED: Return code {process.returncode}")
        results.append({
            'dataset': 'RWF-2000',
            'status': 'failed',
            'error': f'Return code {process.returncode}'
        })

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    results.append({
        'dataset': 'RWF-2000',
        'status': 'error',
        'error': str(e)[:200]
    })

# ============================================================================
# 2. RLVS + Hockey Fight
# ============================================================================
print("\n" + "="*80)
print("2. RLVS + Hockey Fight Dataset")
print("Dataset: inzilele/rlvs-hockey-fight-dataset-mp4-files")
print("="*80)

rlvs_dir = output_base / "RLVS_Hockey"
rlvs_dir.mkdir(exist_ok=True)

try:
    cmd = [
        'kaggle', 'datasets', 'download',
        '-d', 'inzilele/rlvs-hockey-fight-dataset-mp4-files',
        '-p', str(rlvs_dir)
    ]

    print("Downloading...")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()

    process.wait()

    if process.returncode == 0:
        print("\nDownload complete! Extracting...")

        for zipfile in rlvs_dir.glob('*.zip'):
            print(f"Unzipping {zipfile.name}...")
            subprocess.run(['unzip', '-q', str(zipfile), '-d', str(rlvs_dir)])
            zipfile.unlink()
            print("✓ Extracted")

        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI']:
            video_count += len(list(rlvs_dir.rglob(ext)))

        size_gb = sum(f.stat().st_size for f in rlvs_dir.rglob('*') if f.is_file()) / (1024**3)

        print(f"\n✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
        results.append({
            'dataset': 'RLVS+Hockey',
            'status': 'success',
            'videos': video_count,
            'size_gb': round(size_gb, 2),
            'path': str(rlvs_dir)
        })
    else:
        print(f"\n❌ FAILED: Return code {process.returncode}")
        results.append({
            'dataset': 'RLVS+Hockey',
            'status': 'failed',
            'error': f'Return code {process.returncode}'
        })

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    results.append({
        'dataset': 'RLVS+Hockey',
        'status': 'error',
        'error': str(e)[:200]
    })

# ============================================================================
# 3. UCF Crime (Abuse, Fighting, Assault, NormalVideos ONLY)
# ============================================================================
print("\n" + "="*80)
print("3. UCF Crime Dataset")
print("Dataset: odins0n/ucf-crime-dataset")
print("Filtering: Abuse, Fighting, Assault, NormalVideos")
print("="*80)

ucf_temp = output_base / "UCF_Crime_temp"
ucf_temp.mkdir(exist_ok=True)

ucf_final = output_base / "UCF_Crime_filtered"
ucf_final.mkdir(exist_ok=True)

try:
    cmd = [
        'kaggle', 'datasets', 'download',
        '-d', 'odins0n/ucf-crime-dataset',
        '-p', str(ucf_temp)
    ]

    print("Downloading...")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()

    process.wait()

    if process.returncode == 0:
        print("\nDownload complete! Extracting...")

        for zipfile in ucf_temp.glob('*.zip'):
            print(f"Unzipping {zipfile.name}...")
            subprocess.run(['unzip', '-q', str(zipfile), '-d', str(ucf_temp)])
            zipfile.unlink()
            print("✓ Extracted")

        print("\nFiltering categories...")

        # Categories to keep
        keep_categories = ['Abuse', 'Fighting', 'Assault', 'Normal', 'abuse', 'fighting', 'assault', 'normal']

        # Find and move videos
        moved_count = 0

        # Search for matching directories
        for item in ucf_temp.rglob('*'):
            if item.is_dir():
                dir_name = item.name.lower()

                # Check if directory matches our categories
                if any(cat.lower() in dir_name for cat in keep_categories):
                    print(f"  Found: {item.name}")

                    # Create output directory
                    output_cat = ucf_final / item.name
                    output_cat.mkdir(exist_ok=True)

                    # Move all videos
                    for video in item.rglob('*'):
                        if video.is_file() and video.suffix.lower() in ['.mp4', '.avi', '.mkv', '.webm', '.mov']:
                            dest = output_cat / video.name
                            shutil.move(str(video), str(dest))
                            moved_count += 1

        print(f"  Moved {moved_count} videos")

        # Clean up temp
        shutil.rmtree(ucf_temp, ignore_errors=True)

        # Count final videos
        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
            video_count += len(list(ucf_final.rglob(ext)))

        size_gb = sum(f.stat().st_size for f in ucf_final.rglob('*') if f.is_file()) / (1024**3)

        print(f"\n✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
        print("   Breakdown:")
        for subdir in ucf_final.iterdir():
            if subdir.is_dir():
                count = sum(1 for _ in subdir.rglob('*.mp4')) + sum(1 for _ in subdir.rglob('*.avi'))
                if count > 0:
                    print(f"     {subdir.name}: {count} videos")

        results.append({
            'dataset': 'UCF-Crime',
            'status': 'success',
            'videos': video_count,
            'size_gb': round(size_gb, 2),
            'path': str(ucf_final)
        })
    else:
        print(f"\n❌ FAILED: Return code {process.returncode}")
        results.append({
            'dataset': 'UCF-Crime',
            'status': 'failed',
            'error': f'Return code {process.returncode}'
        })
        shutil.rmtree(ucf_temp, ignore_errors=True)

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    results.append({
        'dataset': 'UCF-Crime',
        'status': 'error',
        'error': str(e)[:200]
    })
    shutil.rmtree(ucf_temp, ignore_errors=True)

# ============================================================================
# FINAL SUMMARY
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
        print()

if failed:
    print("FAILED DOWNLOADS:")
    for r in failed:
        print(f"  ❌ {r['dataset']}")
        print(f"     Status: {r['status']}")
        if 'error' in r:
            print(f"     Error: {r['error']}")
        print()

# Save report
report_file = output_base / "download_report.json"
with open(report_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"📄 Report: {report_file}")
print(f"📁 All datasets: {output_base}")
print("="*80)
