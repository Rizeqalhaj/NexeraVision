#!/usr/bin/env python3
"""
Search Kaggle for violence detection datasets and TEST which ones actually work
Verifies dataset structure and downloads working ones
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

output_base = Path("/workspace/violence_datasets_working")
output_base.mkdir(exist_ok=True)

print("="*80)
print("SEARCHING FOR WORKING VIOLENCE DATASETS ON KAGGLE")
print("Testing each one to verify it downloads without 403")
print("="*80)
print()

# Search for all violence-related datasets
search_terms = [
    "violence detection",
    "fight detection",
    "violence video",
    "fight video",
    "assault detection",
    "aggression detection",
]

all_datasets = set()

print("🔍 Searching Kaggle...")
for term in search_terms:
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'list', '-s', term],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n')[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) > 0 and '/' in parts[0]:
                        all_datasets.add(parts[0])
    except:
        continue

print(f"✅ Found {len(all_datasets)} datasets\n")

# Test each dataset
working_datasets = []
forbidden_datasets = []
other_errors = []

print("="*80)
print("TESTING EACH DATASET")
print("="*80)
print()

for i, dataset_name in enumerate(sorted(all_datasets), 1):
    print(f"[{i}/{len(all_datasets)}] Testing: {dataset_name}")

    test_dir = output_base / "test" / dataset_name.replace('/', '_')
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try to download
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(test_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Success! Check if it has videos
            has_videos = False
            try:
                # Unzip if needed
                for zipfile in test_dir.glob('*.zip'):
                    subprocess.run(['unzip', '-q', '-o', str(zipfile), '-d', str(test_dir)], timeout=60)
                    zipfile.unlink()

                # Check for videos
                video_count = 0
                for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
                    video_count += len(list(test_dir.rglob(ext)))

                if video_count > 0:
                    size_gb = sum(f.stat().st_size for f in test_dir.rglob('*') if f.is_file()) / (1024**3)
                    print(f"  ✅ WORKS! {video_count} videos ({size_gb:.2f} GB)")
                    working_datasets.append({
                        'dataset': dataset_name,
                        'videos': video_count,
                        'size_gb': round(size_gb, 2)
                    })
                    has_videos = True
                else:
                    print(f"  ⚠️  Downloaded but no videos found")
            except:
                pass

            # Clean up test if no videos
            if not has_videos:
                shutil.rmtree(test_dir, ignore_errors=True)

        elif '403' in result.stdout or 'Forbidden' in result.stdout:
            print(f"  ❌ 403 Forbidden (needs terms acceptance)")
            forbidden_datasets.append(dataset_name)
            shutil.rmtree(test_dir, ignore_errors=True)

        else:
            error = result.stdout[:100] if result.stdout else result.stderr[:100]
            print(f"  ❌ Error: {error}")
            other_errors.append({'dataset': dataset_name, 'error': error})
            shutil.rmtree(test_dir, ignore_errors=True)

    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Timeout")
        shutil.rmtree(test_dir, ignore_errors=True)
    except Exception as e:
        print(f"  ❌ Exception: {str(e)[:100]}")
        shutil.rmtree(test_dir, ignore_errors=True)

# Summary
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print()

print(f"✅ Working datasets: {len(working_datasets)}")
print(f"❌ Forbidden (403): {len(forbidden_datasets)}")
print(f"⚠️  Other errors: {len(other_errors)}")
print()

if working_datasets:
    print("WORKING DATASETS (NO TERMS NEEDED):")
    total_videos = 0
    total_size = 0
    for ds in working_datasets:
        print(f"  ✅ {ds['dataset']}")
        print(f"     Videos: {ds['videos']:,} ({ds['size_gb']} GB)")
        total_videos += ds['videos']
        total_size += ds['size_gb']

    print()
    print(f"Total: {total_videos:,} videos ({total_size:.2f} GB)")
    print()

    # Save working dataset list
    working_file = output_base / "working_datasets.txt"
    with open(working_file, 'w') as f:
        for ds in working_datasets:
            f.write(f"{ds['dataset']}\n")

    print(f"📄 Working dataset list: {working_file}")
    print()

    # Create download script
    download_script = output_base / "download_all_working.sh"
    with open(download_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Download all working violence datasets\n\n")
        f.write(f'BASE_DIR="/workspace/violence_datasets_working"\n')
        f.write('mkdir -p "$BASE_DIR"\n\n')

        for ds in working_datasets:
            safe_name = ds['dataset'].replace('/', '_')
            f.write(f'echo "Downloading {ds["dataset"]}..."\n')
            f.write(f'kaggle datasets download -d {ds["dataset"]} -p "$BASE_DIR/{safe_name}" --unzip\n')
            f.write(f'echo "✓ {ds["dataset"]} complete"\n\n')

        f.write('echo "All downloads complete!"\n')

    os.chmod(download_script, 0o755)
    print(f"📜 Download script: {download_script}")

if forbidden_datasets:
    print()
    print(f"FORBIDDEN DATASETS ({len(forbidden_datasets)} - need manual acceptance):")
    for ds in forbidden_datasets[:10]:
        print(f"  ❌ {ds}")
    if len(forbidden_datasets) > 10:
        print(f"  ... and {len(forbidden_datasets) - 10} more")

# Save results
results_file = output_base / "search_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'working': working_datasets,
        'forbidden': forbidden_datasets,
        'errors': other_errors
    }, f, indent=2)

print()
print(f"📄 Full results: {results_file}")
print("="*80)
