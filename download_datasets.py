#!/usr/bin/env python3
"""
NexaraVision Dataset Downloader
Downloads all Tier 1 violence detection datasets from Kaggle
"""

import subprocess
import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Kaggle Configuration
KAGGLE_USERNAME = "issadalu"
KAGGLE_KEY = "5aabafacbfdefea1bf4f2171d98cc52b"

# Dataset Configuration - Tier 1 (Core Violence Detection)
DATASETS = [
    ('vulamnguyen/rwf2000', 'tier1/RWF2000', 'RWF-2000', 2000, 1.5),
    ('odins0n/ucf-crime-dataset', 'tier1/UCF_Crime', 'UCF-Crime', 1900, 12.0),
    ('toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd', 'tier1/SCVD', 'SmartCity-CCTV', 4000, 3.5),
    ('mohamedmustafa/real-life-violence-situations-dataset', 'tier1/RealLife', 'Real-Life Violence', 2000, 2.0),
    ('arnab91/eavdd-violence', 'tier1/EAVDD', 'EAVDD', 1500, 1.8),
]

def setup_kaggle():
    """Configure Kaggle API credentials"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(json.dumps({
        "username": KAGGLE_USERNAME,
        "key": KAGGLE_KEY
    }))
    kaggle_json.chmod(0o600)

    print("✅ Kaggle credentials configured")

def setup_workspace():
    """Create workspace directory structure"""
    base_dir = Path("/workspace")

    # Create directories
    dirs = [
        base_dir / "datasets" / "tier1",
        base_dir / "datasets" / "tier2",
        base_dir / "datasets" / "tier3",
        base_dir / "datasets" / "processed",
        base_dir / "models",
        base_dir / "logs",
        base_dir / "checkpoints",
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("✅ Workspace directories created")

def install_kaggle():
    """Install Kaggle CLI"""
    print("📦 Installing Kaggle CLI...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "kaggle"],
            check=True,
            capture_output=True
        )
        print("✅ Kaggle CLI installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Kaggle might already be installed: {e}")

def download_dataset(kaggle_id, path, name, expected_videos, expected_size):
    """Download a single dataset from Kaggle"""

    print(f"\n{'='*80}")
    print(f"📥 {name}")
    print(f"{'='*80}")
    print(f"Dataset ID: {kaggle_id}")
    print(f"Expected: {expected_videos:,} videos (~{expected_size}GB)")

    out_dir = Path(f"/workspace/datasets/{path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        start = datetime.now()
        print(f"\n⏳ Downloading from Kaggle...")
        sys.stdout.flush()

        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', kaggle_id, '-p', str(out_dir), '--unzip'],
            capture_output=True,
            text=True,
            timeout=3600
        )

        if result.returncode == 0:
            # Count videos
            video_count = 0
            for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI', '*.MKV']:
                video_count += len(list(out_dir.rglob(ext)))

            # Calculate size
            size_gb = sum(f.stat().st_size for f in out_dir.rglob('*') if f.is_file()) / (1024**3)

            elapsed = (datetime.now() - start).total_seconds()

            print(f"\n✅ SUCCESS!")
            print(f"   Videos: {video_count:,}")
            print(f"   Size: {size_gb:.2f} GB")
            print(f"   Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            sys.stdout.flush()

            return {
                'name': name,
                'status': 'success',
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'time_s': round(elapsed, 1),
                'path': str(out_dir)
            }
        else:
            print(f"\n❌ FAILED: {result.stderr[:300]}")
            sys.stdout.flush()
            return {
                'name': name,
                'status': 'failed',
                'error': result.stderr[:200]
            }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)[:300]}")
        sys.stdout.flush()
        return {
            'name': name,
            'status': 'error',
            'error': str(e)[:200]
        }

def main():
    """Main download function"""

    print("="*80)
    print("NexaraVision Dataset Downloader")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

    # Setup
    print("🔧 Setting up environment...")
    install_kaggle()
    setup_kaggle()
    setup_workspace()

    print("\n" + "="*80)
    print("📥 Starting Dataset Downloads")
    print("="*80)

    # Download all datasets
    results = []
    for i, (kaggle_id, path, name, vids, size) in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Processing: {name}")
        result = download_dataset(kaggle_id, path, name, vids, size)
        results.append(result)

    # Save results
    results_file = Path("/workspace/datasets/download_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    # Print summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print("\n" + "="*80)
    print("📊 DOWNLOAD SUMMARY")
    print("="*80)
    print(f"\n✅ Successful: {len(successful)}/{len(results)} datasets")
    print(f"📹 Total Videos: {sum(r.get('videos', 0) for r in successful):,}")
    print(f"💾 Total Size: {sum(r.get('size_gb', 0) for r in successful):.2f} GB")
    print(f"⏱️  Total Time: {sum(r.get('time_s', 0) for r in successful)/60:.1f} minutes")

    if failed:
        print(f"\n❌ Failed: {len(failed)} datasets")
        for r in failed:
            print(f"   - {r['name']}: {r.get('error', 'Unknown')[:100]}")

    print(f"\n📄 Results saved: {results_file}")
    print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Disk usage
    print(f"\n💾 Final Disk Usage:")
    os.system("du -sh /workspace/datasets/* 2>/dev/null")
    os.system("df -h /workspace | tail -1")

    print("\n✅ ALL DOWNLOADS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
