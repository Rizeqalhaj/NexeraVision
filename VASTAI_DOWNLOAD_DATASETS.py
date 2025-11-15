#!/usr/bin/env python3
"""
NexaraVision - Master Dataset Downloader for Vast.ai
Downloads all Tier 1, 2, and 3 datasets for violence detection training

Based on PROGRESS.md research - targeting 50,000+ videos for 93-98% accuracy
"""

import subprocess
import os
from pathlib import Path
import shutil
import json
import sys
from datetime import datetime

# =============================================================================
# TIER 1: Core Violence Detection Datasets (Highest Priority)
# =============================================================================
TIER1_KAGGLE_DATASETS = [
    {
        'name': 'RWF-2000',
        'kaggle_id': 'vulamnguyen/rwf2000',
        'videos': 2000,
        'size_gb': 1.5,
        'description': 'Real-world fight videos from surveillance'
    },
    {
        'name': 'UCF-Crime',
        'kaggle_id': 'odins0n/ucf-crime-dataset',
        'videos': 1900,
        'size_gb': 12.0,
        'description': 'Untrimmed surveillance videos with 13 anomaly types'
    },
    {
        'name': 'SmartCity-CCTV (SCVD)',
        'kaggle_id': 'toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd',
        'videos': 4000,
        'size_gb': 3.5,
        'description': 'Smart city CCTV violence detection dataset'
    },
    {
        'name': 'Real-Life Violence',
        'kaggle_id': 'mohamedmustafa/real-life-violence-situations-dataset',
        'videos': 2000,
        'size_gb': 2.0,
        'description': 'Real-life violence situations'
    },
    {
        'name': 'EAVDD',
        'kaggle_id': 'arnab91/eavdd-violence',
        'videos': 1500,
        'size_gb': 1.8,
        'description': 'Extended abnormal video dataset for violence'
    },
]

# =============================================================================
# TIER 2: Extended Datasets (Secondary Priority)
# =============================================================================
TIER2_KAGGLE_DATASETS = [
    {
        'name': 'XD-Violence',
        'kaggle_id': 'nguhaduong/xd-violence-video-dataset',
        'videos': 4754,
        'size_gb': 45.0,
        'description': 'Large-scale violence detection dataset'
    },
    {
        'name': 'Hockey Fight Detection',
        'kaggle_id': 'dataset/hockey-fight-detection',
        'videos': 1000,
        'size_gb': 0.8,
        'description': 'Hockey fight videos'
    },
]

# =============================================================================
# TIER 3: Non-Violence / Normal Activity Datasets
# =============================================================================
TIER3_DATASETS = [
    {
        'name': 'UCF-101 Normal',
        'kaggle_id': 'dataset/ucf101',
        'videos': 13320,
        'size_gb': 7.0,
        'description': 'Normal human activities (101 classes)'
    },
    {
        'name': 'HMDB-51',
        'kaggle_id': 'dataset/hmdb51',
        'videos': 6766,
        'size_gb': 2.0,
        'description': 'Normal human motion activities'
    },
]

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path("/workspace/datasets")
TIER1_DIR = BASE_DIR / "tier1"
TIER2_DIR = BASE_DIR / "tier2"
TIER3_DIR = BASE_DIR / "tier3"

# Create directories
for dir_path in [TIER1_DIR, TIER2_DIR, TIER3_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Setup Kaggle credentials
def setup_kaggle():
    """Ensure Kaggle credentials are configured"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("❌ ERROR: Kaggle credentials not found!")
        print("Run VASTAI_SETUP_COMPLETE.sh first")
        sys.exit(1)

    print("✅ Kaggle credentials configured\n")

# Download function with progress
def download_dataset(dataset_info, output_dir, tier):
    """Download a single dataset from Kaggle"""

    name = dataset_info['name']
    kaggle_id = dataset_info['kaggle_id']

    print("\n" + "="*80)
    print(f"📥 {tier} - {name}")
    print(f"Dataset ID: {kaggle_id}")
    print(f"Expected: {dataset_info['videos']} videos (~{dataset_info['size_gb']} GB)")
    print(f"Description: {dataset_info['description']}")
    print("="*80)

    dataset_dir = output_dir / name.replace(' ', '_').replace('-', '_')
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download with kaggle CLI
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', kaggle_id,
            '-p', str(dataset_dir),
            '--unzip'
        ]

        print(f"\n⏳ Downloading {name}...")

        # Run with live output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()

        process.wait()

        if process.returncode == 0:
            # Count videos
            video_count = 0
            for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI', '*.MKV']:
                video_count += len(list(dataset_dir.rglob(ext)))

            # Calculate size
            size_gb = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file()) / (1024**3)

            print(f"\n✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")

            return {
                'name': name,
                'tier': tier,
                'status': 'success',
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'path': str(dataset_dir)
            }
        else:
            print(f"\n❌ FAILED: Return code {process.returncode}")
            return {
                'name': name,
                'tier': tier,
                'status': 'failed',
                'error': f'Return code {process.returncode}'
            }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return {
            'name': name,
            'tier': tier,
            'status': 'error',
            'error': str(e)
        }

# Main download function
def main():
    print("="*80)
    print("NexaraVision - Dataset Download Manager")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

    # Setup
    setup_kaggle()

    results = []

    # Download TIER 1 (Core Violence Datasets)
    print("\n" + "🔴"*40)
    print("TIER 1: Core Violence Detection Datasets")
    print("🔴"*40)

    for dataset in TIER1_KAGGLE_DATASETS:
        result = download_dataset(dataset, TIER1_DIR, "TIER 1")
        results.append(result)

    # Download TIER 2 (Extended Datasets)
    print("\n" + "🟡"*40)
    print("TIER 2: Extended Violence Datasets")
    print("🟡"*40)

    for dataset in TIER2_KAGGLE_DATASETS:
        result = download_dataset(dataset, TIER2_DIR, "TIER 2")
        results.append(result)

    # Download TIER 3 (Non-Violence Normal)
    print("\n" + "🟢"*40)
    print("TIER 3: Non-Violence / Normal Activity Datasets")
    print("🟢"*40)

    for dataset in TIER3_DATASETS:
        result = download_dataset(dataset, TIER3_DIR, "TIER 3")
        results.append(result)

    # Save results
    results_file = BASE_DIR / "download_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, indent=2, fp=f)

    # Summary
    print("\n" + "="*80)
    print("📊 DOWNLOAD SUMMARY")
    print("="*80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    total_videos = sum(r.get('videos', 0) for r in successful)
    total_size = sum(r.get('size_gb', 0) for r in successful)

    print(f"\n✅ Successful: {len(successful)}/{len(results)} datasets")
    print(f"📹 Total Videos: {total_videos:,}")
    print(f"💾 Total Size: {total_size:.2f} GB")

    if failed:
        print(f"\n❌ Failed: {len(failed)} datasets")
        for r in failed:
            print(f"   - {r['name']}: {r.get('error', 'Unknown error')}")

    print(f"\n📄 Full results saved to: {results_file}")
    print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
