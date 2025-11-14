#!/usr/bin/env python3
"""
Download EXACT Kaggle datasets specified by user
1. RWF-2000
2. RLVS + Hockey Fight
3. UCF Crime (Abuse, Fighting, Assault, NormalVideos only)
"""

import subprocess
import os
from pathlib import Path
import shutil
import json

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
print("DOWNLOADING EXACT KAGGLE DATASETS")
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
    cmd = [
        'kaggle', 'datasets', 'download',
        '-d', 'vulamnguyen/rwf2000',
        '-p', str(rwf_dir),
        '--unzip'
    ]

    print("Downloading...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode == 0:
        # Count videos
        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
            video_count += len(list(rwf_dir.rglob(ext)))

        size_gb = sum(f.stat().st_size for f in rwf_dir.rglob('*') if f.is_file()) / (1024**3)

        print(f"✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
        results.append({
            'dataset': 'RWF-2000',
            'status': 'success',
            'videos': video_count,
            'size_gb': round(size_gb, 2),
            'path': str(rwf_dir)
        })
    else:
        error = result.stdout if result.stdout else result.stderr
        print(f"❌ FAILED: {error[:200]}")
        results.append({
            'dataset': 'RWF-2000',
            'status': 'failed',
            'error': error[:200]
        })

except Exception as e:
    print(f"❌ ERROR: {str(e)[:200]}")
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
        '-p', str(rlvs_dir),
        '--unzip'
    ]

    print("Downloading...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode == 0:
        # Count videos
        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
            video_count += len(list(rlvs_dir.rglob(ext)))

        size_gb = sum(f.stat().st_size for f in rlvs_dir.rglob('*') if f.is_file()) / (1024**3)

        print(f"✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
        results.append({
            'dataset': 'RLVS+Hockey',
            'status': 'success',
            'videos': video_count,
            'size_gb': round(size_gb, 2),
            'path': str(rlvs_dir)
        })
    else:
        error = result.stdout if result.stdout else result.stderr
        print(f"❌ FAILED: {error[:200]}")
        results.append({
            'dataset': 'RLVS+Hockey',
            'status': 'failed',
            'error': error[:200]
        })

except Exception as e:
    print(f"❌ ERROR: {str(e)[:200]}")
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
        '-p', str(ucf_temp),
        '--unzip'
    ]

    print("Downloading...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode == 0:
        print("Download complete, filtering categories...")

        # Categories to keep
        keep_categories = ['Abuse', 'Fighting', 'Assault', 'NormalVideos',
                          'abuse', 'fighting', 'assault', 'normal', 'Normal']

        # Find and move videos from these categories
        moved_count = 0
        for category in keep_categories:
            # Search for directories matching category
            for cat_dir in ucf_temp.rglob(f'*{category}*'):
                if cat_dir.is_dir():
                    print(f"  Found category: {cat_dir.name}")

                    # Create output directory
                    output_cat = ucf_final / cat_dir.name
                    output_cat.mkdir(exist_ok=True)

                    # Move all videos
                    for video in cat_dir.rglob('*'):
                        if video.is_file() and video.suffix.lower() in ['.mp4', '.avi', '.mkv', '.webm', '.mov']:
                            dest = output_cat / video.name
                            shutil.move(str(video), str(dest))
                            moved_count += 1

        # Clean up temp
        shutil.rmtree(ucf_temp, ignore_errors=True)

        # Count final videos
        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
            video_count += len(list(ucf_final.rglob(ext)))

        size_gb = sum(f.stat().st_size for f in ucf_final.rglob('*') if f.is_file()) / (1024**3)

        print(f"✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
        print(f"   Filtered to: Abuse, Fighting, Assault, NormalVideos")

        # Show breakdown
        print("   Breakdown:")
        for subdir in ucf_final.iterdir():
            if subdir.is_dir():
                count = len(list(subdir.rglob('*.mp4'))) + len(list(subdir.rglob('*.avi')))
                print(f"     {subdir.name}: {count} videos")

        results.append({
            'dataset': 'UCF-Crime',
            'status': 'success',
            'videos': video_count,
            'size_gb': round(size_gb, 2),
            'path': str(ucf_final)
        })
    else:
        error = result.stdout if result.stdout else result.stderr
        print(f"❌ FAILED: {error[:200]}")
        results.append({
            'dataset': 'UCF-Crime',
            'status': 'failed',
            'error': error[:200]
        })
        shutil.rmtree(ucf_temp, ignore_errors=True)

except Exception as e:
    print(f"❌ ERROR: {str(e)[:200]}")
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
