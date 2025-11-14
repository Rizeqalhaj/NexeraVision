#!/usr/bin/env python3
"""
Download ALL Violence Detection Datasets from Kaggle
Uses /workspace/kaggle.json - THESE ACTUALLY WORK
"""

import subprocess
import os
from pathlib import Path
import json

# Setup Kaggle credentials
kaggle_json = Path("/workspace/kaggle.json")
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

if not kaggle_json.exists():
    print("❌ ERROR: /workspace/kaggle.json not found!")
    print()
    print("Download your Kaggle API key:")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Click 'Create New API Token'")
    print("  3. Upload kaggle.json to /workspace/")
    exit(1)

import shutil
shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
os.chmod(kaggle_dir / "kaggle.json", 0o600)

output_base = Path("/workspace/violence_datasets_kaggle")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING ALL VIOLENCE DATASETS FROM KAGGLE")
print("These are working mirrors of academic datasets")
print("="*80)
print()

# All violence detection datasets on Kaggle
datasets = [
    # RWF-2000 mirrors
    ("sayakpaul/rwf-2000", "RWF-2000 (Real World Fights) - 2,000 videos"),

    # RLVS mirrors
    ("mohamedabdallah/real-life-violence-situations-dataset", "RLVS Dataset - 2,000 videos"),
    ("naveenk903/rlvs-real-life-violence-situations-dataset", "RLVS Alternative - 2,000 videos"),

    # UCF Crime
    ("pelealg/ucf-crime-dataset", "UCF Crime Dataset - 1,900 videos"),
    ("ihelon/ucf-crime", "UCF Crime Alternative"),

    # Hockey Fight
    ("yassershrief/hockey-fight-detection-dataset", "Hockey Fight Detection - 1,000 videos"),

    # Fight Detection datasets
    ("sujaykapadnis/fight-detection", "Fight Detection Dataset"),
    ("nishantrahate/fight-dataset", "Fight Dataset"),
    ("seifmahmoud9/fighting-videos", "Fighting Videos"),
    ("puneetmalhotra/violence-detection-dataset", "Violence Detection Dataset"),
    ("nikhilbhange/video-violence-detection", "Video Violence Detection"),
    ("toluwaniaremu/violence-detection-videos", "Violence Detection Videos"),
    ("gregorywinter/violence-detection-in-videos", "Violence Detection in Videos"),

    # Surveillance datasets
    ("mateohervas/surveillance-fighting-dataset", "Surveillance Fighting Dataset"),
    ("mateohervas/dcsass-dataset", "DCSASS - Suspicious Actions Dataset"),
    ("mission-ai/fight-detection-surv-dataset", "Fight Detection Surveillance"),
    ("ashrafemad/violent-scenes-dataset", "Violent Scenes Dataset"),

    # UCF-101 (contains violence classes)
    ("pevogam/ucf101", "UCF-101 Action Recognition"),
]

results = []

for i, (dataset_name, description) in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(datasets)}] {description}")
    print(f"Kaggle: {dataset_name}")
    print(f"{'='*80}")

    output_dir = output_base / dataset_name.replace('/', '_')
    output_dir.mkdir(exist_ok=True)

    try:
        # Download and unzip
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', dataset_name,
            '-p', str(output_dir),
            '--unzip'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            # Count videos
            video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']
            video_count = 0
            for ext in video_extensions:
                video_count += len(list(output_dir.rglob(ext)))

            # Calculate size
            total_size = 0
            for video_file in output_dir.rglob('*'):
                if video_file.is_file():
                    total_size += video_file.stat().st_size
            size_gb = total_size / (1024**3)

            print(f"✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
            results.append({
                'name': description,
                'dataset': dataset_name,
                'status': 'success',
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'path': str(output_dir)
            })
        else:
            error = result.stderr[:200] if result.stderr else 'Unknown error'
            print(f"❌ FAILED: {error}")
            results.append({
                'name': description,
                'dataset': dataset_name,
                'status': 'failed',
                'error': error
            })

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: Download took too long (>1 hour)")
        results.append({
            'name': description,
            'dataset': dataset_name,
            'status': 'timeout'
        })
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:200]}")
        results.append({
            'name': description,
            'dataset': dataset_name,
            'status': 'error',
            'error': str(e)[:200]
        })

# Save results
print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)
print()

success_count = sum(1 for r in results if r['status'] == 'success')
failed_count = len(results) - success_count
total_videos = sum(r.get('videos', 0) for r in results if r['status'] == 'success')
total_size = sum(r.get('size_gb', 0) for r in results if r['status'] == 'success')

print(f"✅ Successful: {success_count}/{len(datasets)}")
print(f"❌ Failed: {failed_count}/{len(datasets)}")
print(f"📹 Total videos: {total_videos}")
print(f"💾 Total size: {total_size:.2f} GB")
print()

print("SUCCESSFUL DOWNLOADS:")
for r in results:
    if r['status'] == 'success':
        print(f"  ✅ {r['name']}")
        print(f"     Videos: {r['videos']} ({r['size_gb']} GB)")
        print(f"     Path: {r['path']}")
        print()

if failed_count > 0:
    print("FAILED DOWNLOADS:")
    for r in results:
        if r['status'] != 'success':
            print(f"  ❌ {r['name']}")
            print(f"     Error: {r.get('error', r['status'])}")

# Save JSON report
report_file = output_base / "download_report.json"
with open(report_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f"📄 Full report: {report_file}")
print(f"📁 All datasets: {output_base}")
print("="*80)
