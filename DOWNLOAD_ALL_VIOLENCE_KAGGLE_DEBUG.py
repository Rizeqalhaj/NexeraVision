#!/usr/bin/env python3
"""
Download ALL Violence Detection Datasets from Kaggle - DEBUG VERSION
Shows full error messages to diagnose issues
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
    exit(1)

shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
os.chmod(kaggle_dir / "kaggle.json", 0o600)
print("✅ Kaggle credentials configured")

# Test Kaggle CLI first
print("\n🔍 Testing Kaggle CLI authentication...")
test_result = subprocess.run(['kaggle', 'datasets', 'list', '--max-size', '1'],
                            capture_output=True, text=True)
if test_result.returncode != 0:
    print(f"❌ Kaggle authentication failed!")
    print(f"STDOUT: {test_result.stdout}")
    print(f"STDERR: {test_result.stderr}")
    exit(1)
else:
    print("✅ Kaggle authentication working!")

output_base = Path("/workspace/violence_datasets_kaggle")
output_base.mkdir(exist_ok=True)

print("\n" + "="*80)
print("DOWNLOADING VIOLENCE DETECTION DATASETS FROM KAGGLE")
print("="*80)

# Start with just a few datasets to test
datasets = [
    ("sayakpaul/rwf-2000", "RWF-2000 (Real World Fights) - 2,000 videos"),
    ("mohamedabdallah/real-life-violence-situations-dataset", "RLVS Dataset - 2,000 videos"),
    ("pelealg/ucf-crime-dataset", "UCF Crime Dataset - 1,900 videos"),
]

results = []

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

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        print(f"\nReturn code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")

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

            print(f"\n✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
            results.append({
                'name': description,
                'dataset': dataset_name,
                'status': 'success',
                'videos': video_count,
                'size_gb': round(size_gb, 2)
            })

        else:
            print(f"\n❌ FAILED")
            results.append({
                'name': description,
                'dataset': dataset_name,
                'status': 'failed',
                'stdout': result.stdout,
                'stderr': result.stderr
            })

    except subprocess.TimeoutExpired:
        print(f"\n⏱️  TIMEOUT")
        results.append({
            'name': description,
            'dataset': dataset_name,
            'status': 'timeout'
        })

    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append({
            'name': description,
            'dataset': dataset_name,
            'status': 'error',
            'exception': str(e)
        })

# Save results
print("\n" + "="*80)
print("RESULTS:")
print("="*80)
for r in results:
    print(f"\n{r['name']}: {r['status']}")
    if r['status'] == 'success':
        print(f"  Videos: {r['videos']}, Size: {r['size_gb']} GB")
    else:
        if 'stdout' in r:
            print(f"  STDOUT: {r['stdout'][:200]}")
        if 'stderr' in r:
            print(f"  STDERR: {r['stderr'][:200]}")
        if 'exception' in r:
            print(f"  Exception: {r['exception']}")

report_file = output_base / "debug_report.json"
with open(report_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nFull report: {report_file}")
