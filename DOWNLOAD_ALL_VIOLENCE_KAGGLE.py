#!/usr/bin/env python3
"""
Download ALL Violence Detection Datasets from Kaggle
Comprehensive collection of all available violence/fight detection datasets
"""

import subprocess
import os
import json
from pathlib import Path
import shutil

# Setup Kaggle credentials
kaggle_json = Path("kaggle.json")
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

if not kaggle_json.exists():
    print("❌ ERROR: kaggle.json not found in current directory!")
    print("Place your kaggle.json file here first")
    exit(1)

shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
os.chmod(kaggle_dir / "kaggle.json", 0o600)
print("✅ Kaggle credentials configured\n")

output_base = Path("/workspace/violence_datasets_kaggle")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING ALL VIOLENCE DETECTION DATASETS FROM KAGGLE")
print("="*80)
print()

# Comprehensive list of ALL violence detection datasets on Kaggle
datasets = [
    # ========== PRIMARY ACADEMIC DATASETS ==========
    ("sayakpaul/rwf-2000", "RWF-2000 (Real World Fights) - 2,000 videos"),
    ("mohamedabdallah/real-life-violence-situations-dataset", "RLVS Dataset - 2,000 videos"),
    ("naveenk903/rlvs-real-life-violence-situations-dataset", "RLVS Alternative Mirror"),
    ("pelealg/ucf-crime-dataset", "UCF Crime Dataset - 1,900 videos"),
    ("ihelon/ucf-crime", "UCF Crime Alternative Mirror"),
    ("yassershrief/hockey-fight-detection-dataset", "Hockey Fight Detection - 1,000 videos"),

    # ========== FIGHT DETECTION DATASETS ==========
    ("sujaykapadnis/fight-detection", "Fight Detection Dataset"),
    ("nishantrahate/fight-dataset", "Fight Dataset"),
    ("seifmahmoud9/fighting-videos", "Fighting Videos Dataset"),
    ("toluwaniaremu/violence-detection-videos", "Violence Detection Videos"),
    ("puneetmalhotra/violence-detection-dataset", "Violence Detection Dataset"),
    ("nikhilbhange/video-violence-detection", "Video Violence Detection"),
    ("gregorywinter/violence-detection-in-videos", "Violence Detection in Videos"),

    # ========== SURVEILLANCE DATASETS ==========
    ("mateohervas/surveillance-fighting-dataset", "Surveillance Fighting Dataset"),
    ("mateohervas/dcsass-dataset", "DCSASS - Suspicious Actions Dataset"),
    ("mission-ai/fight-detection-surv-dataset", "Fight Detection Surveillance"),
    ("ashrafemad/violent-scenes-dataset", "Violent Scenes Dataset"),

    # ========== ACTION RECOGNITION (Contains Violence) ==========
    ("pevogam/ucf101", "UCF-101 Action Recognition - 13,320 videos"),

    # ========== ADDITIONAL MIRRORS AND VARIATIONS ==========
    ("kmader/violence-recognition-dataset", "Violence Recognition Dataset"),
    ("davidcariboo/violence-detection-dataset", "Violence Detection Dataset"),
]

results = []
total_downloaded = 0
total_failed = 0

for i, (dataset_name, description) in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(datasets)}] {description}")
    print(f"Dataset: {dataset_name}")
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
            total_downloaded += 1

        else:
            error = result.stderr[:200] if result.stderr else 'Unknown error'
            print(f"❌ FAILED: {error}")
            results.append({
                'name': description,
                'dataset': dataset_name,
                'status': 'failed',
                'error': error
            })
            total_failed += 1

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: Download took >1 hour, skipping")
        results.append({
            'name': description,
            'dataset': dataset_name,
            'status': 'timeout'
        })
        total_failed += 1

    except Exception as e:
        print(f"❌ ERROR: {str(e)[:200]}")
        results.append({
            'name': description,
            'dataset': dataset_name,
            'status': 'error',
            'error': str(e)[:200]
        })
        total_failed += 1

# ========== FINAL SUMMARY ==========
print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)
print()

total_videos = sum(r.get('videos', 0) for r in results if r['status'] == 'success')
total_size = sum(r.get('size_gb', 0) for r in results if r['status'] == 'success')

print(f"✅ Successful: {total_downloaded}/{len(datasets)} datasets")
print(f"❌ Failed: {total_failed}/{len(datasets)} datasets")
print(f"📹 Total videos: {total_videos:,}")
print(f"💾 Total size: {total_size:.2f} GB")
print()

if total_downloaded > 0:
    print("SUCCESSFUL DOWNLOADS:")
    for r in results:
        if r['status'] == 'success':
            print(f"  ✅ {r['name']}")
            print(f"     Videos: {r['videos']:,} ({r['size_gb']} GB)")
            print(f"     Path: {r['path']}")
            print()

if total_failed > 0:
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
