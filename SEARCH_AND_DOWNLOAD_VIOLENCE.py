#!/usr/bin/env python3
"""
Search Kaggle for ALL violence detection datasets and download available ones
Skips datasets requiring manual terms acceptance (403 errors)
"""

import subprocess
import os
import json
from pathlib import Path
import shutil
import re

# Setup Kaggle credentials
kaggle_json = Path("kaggle.json")
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

if not kaggle_json.exists():
    print("❌ ERROR: kaggle.json not found!")
    exit(1)

shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
os.chmod(kaggle_dir / "kaggle.json", 0o600)
print("✅ Kaggle credentials configured\n")

output_base = Path("/workspace/violence_datasets_kaggle")
output_base.mkdir(exist_ok=True)

print("="*80)
print("SEARCHING KAGGLE FOR VIOLENCE DETECTION DATASETS")
print("="*80)
print()

# Search for datasets
search_terms = [
    "violence detection",
    "fight detection",
    "violence",
    "fighting",
    "aggressive behavior",
    "surveillance violence",
    "rwf-2000",
    "ucf crime",
    "hockey fight",
]

all_datasets = set()

for term in search_terms:
    print(f"🔍 Searching: '{term}'")
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'list', '-s', term, '--max-size', '100'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip():
                    # Parse: "owner/dataset-name" from first column
                    parts = line.split()
                    if len(parts) > 0:
                        dataset_ref = parts[0]
                        if '/' in dataset_ref:
                            all_datasets.add(dataset_ref)

    except Exception as e:
        print(f"  ⚠️  Search error: {e}")
        continue

print(f"\n✅ Found {len(all_datasets)} unique datasets\n")

# Save dataset list
datasets_file = output_base / "found_datasets.txt"
with open(datasets_file, 'w') as f:
    for ds in sorted(all_datasets):
        f.write(f"{ds}\n")

print(f"📄 Dataset list saved to: {datasets_file}")
print()

# Now try to download each one
print("="*80)
print("ATTEMPTING TO DOWNLOAD ALL FOUND DATASETS")
print("="*80)
print()

results = {
    'success': [],
    'forbidden': [],
    'error': [],
}

for i, dataset_name in enumerate(sorted(all_datasets), 1):
    print(f"\n[{i}/{len(all_datasets)}] {dataset_name}")
    print("-" * 60)

    output_dir = output_base / dataset_name.replace('/', '_')
    output_dir.mkdir(exist_ok=True)

    try:
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', dataset_name,
            '-p', str(output_dir),
            '--unzip'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            # Count videos
            video_count = 0
            for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
                video_count += len(list(output_dir.rglob(ext)))

            # Size
            total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)

            print(f"✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
            results['success'].append({
                'dataset': dataset_name,
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'path': str(output_dir)
            })

        elif '403' in result.stdout or 'Forbidden' in result.stdout:
            print(f"⚠️  FORBIDDEN: Requires manual acceptance on Kaggle website")
            results['forbidden'].append(dataset_name)
            # Clean up empty dir
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()

        else:
            error = result.stdout[:100]
            print(f"❌ ERROR: {error}")
            results['error'].append({
                'dataset': dataset_name,
                'error': error
            })

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT (>30min)")
        results['error'].append({
            'dataset': dataset_name,
            'error': 'timeout'
        })

    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)[:100]}")
        results['error'].append({
            'dataset': dataset_name,
            'error': str(e)[:100]
        })

# Final summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print()

total_videos = sum(r['videos'] for r in results['success'])
total_size = sum(r['size_gb'] for r in results['success'])

print(f"✅ Successfully downloaded: {len(results['success'])} datasets")
print(f"⚠️  Forbidden (need manual accept): {len(results['forbidden'])} datasets")
print(f"❌ Errors: {len(results['error'])} datasets")
print()
print(f"📹 Total videos: {total_videos:,}")
print(f"💾 Total size: {total_size:.2f} GB")
print()

if results['success']:
    print("SUCCESSFUL DOWNLOADS:")
    for r in results['success']:
        print(f"  ✅ {r['dataset']}")
        print(f"     Videos: {r['videos']:,} ({r['size_gb']} GB)")
        print(f"     Path: {r['path']}")
    print()

if results['forbidden']:
    print("DATASETS REQUIRING MANUAL ACCEPTANCE:")
    print("(Visit these URLs on Kaggle website, click 'Download', accept terms)")
    for ds in results['forbidden'][:10]:  # Show first 10
        print(f"  ⚠️  https://www.kaggle.com/datasets/{ds}")
    if len(results['forbidden']) > 10:
        print(f"  ... and {len(results['forbidden']) - 10} more")
    print()

# Save results
report_file = output_base / "download_results.json"
with open(report_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"📄 Full report: {report_file}")
print(f"📁 Downloaded datasets: {output_base}")
print("="*80)
