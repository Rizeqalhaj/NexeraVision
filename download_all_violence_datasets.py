#!/usr/bin/env python3
"""
Download ALL major violence detection datasets for maximum accuracy.
Target: 10,000+ videos for 93-97% accuracy.
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure kaggle is configured
if not Path("~/.kaggle/kaggle.json").expanduser().exists():
    print("⚠️ Kaggle credentials not found!")
    print("Run this first:")
    print("cat > ~/.kaggle/kaggle.json << 'EOF'")
    print('{"username":"issadalu","key":"5aabafacbfdefea1bf4f2171d98cc52b"}')
    print("EOF")
    print("chmod 600 ~/.kaggle/kaggle.json")
    sys.exit(1)

# All major violence detection datasets
DATASETS = {
    "rwf2000": {
        "name": "Real-World Fight (RWF-2000)",
        "kaggle": "vulamnguyen/rwf2000",
        "videos": "~2000",
        "priority": 1
    },
    "hockey": {
        "name": "Hockey Fight Detection",
        "kaggle": "yassershrief/hockey-fight-videos",
        "videos": "~1000",
        "priority": 2
    },
    "violence": {
        "name": "Real-Life Violence Situations",
        "kaggle": "mohamedmustafa/real-life-violence-situations-dataset",
        "videos": "~2000",
        "priority": 3
    },
    "cctv": {
        "name": "CCTV Violence Detection",
        "kaggle": "yassershrief/cctv-violence-dataset",
        "videos": "~500",
        "priority": 4
    },
    "movies": {
        "name": "Movie Violence Dataset",
        "kaggle": "naveenkenz/movies-fight-detection-dataset",
        "videos": "~1000",
        "priority": 5
    },
    "ucf_crime": {
        "name": "UCF Crime Dataset (violence subset)",
        "kaggle": "mission-ai/crimeucfdataset",
        "videos": "~2000",
        "priority": 6
    },
    "surveillance": {
        "name": "Surveillance Violence",
        "kaggle": "toluwaniaremu/violence-detection-dataset",
        "videos": "~800",
        "priority": 7
    }
}

def download_dataset(key, info):
    """Download a single dataset."""
    print(f"\n{'='*80}")
    print(f"📥 DOWNLOADING: {info['name']}")
    print(f"   Expected videos: {info['videos']}")
    print(f"   Priority: {info['priority']}")
    print(f"{'='*80}")

    try:
        # Download using kaggle CLI
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", info['kaggle']],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout per dataset
        )

        if result.returncode == 0:
            print(f"✅ {info['name']} downloaded successfully")
            return True
        else:
            print(f"❌ Failed to download {info['name']}")
            print(f"   Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️ Download timeout for {info['name']}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {info['name']}: {e}")
        return False

def main():
    """Download all datasets in priority order."""
    print("="*80)
    print("COMPREHENSIVE VIOLENCE DATASET COLLECTION")
    print("="*80)
    print(f"Total datasets: {len(DATASETS)}")
    print(f"Expected total videos: 10,000+")
    print(f"Estimated download time: 2-4 hours")
    print(f"Required storage: ~50-100 GB")
    print("="*80)

    # Change to workspace data directory
    os.chdir("/workspace")

    # Download in priority order
    sorted_datasets = sorted(DATASETS.items(), key=lambda x: x[1]['priority'])

    successful = 0
    failed = 0

    for key, info in sorted_datasets:
        if download_dataset(key, info):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"✅ Successful: {successful}/{len(DATASETS)}")
    print(f"❌ Failed: {failed}/{len(DATASETS)}")
    print("\nNext steps:")
    print("1. Extract all downloaded zip files")
    print("2. Run: python combine_all_datasets.py")
    print("3. Run: python runpod_train_ultimate.py  # New ultra-advanced script")
    print("="*80)

if __name__ == "__main__":
    main()
