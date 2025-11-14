#!/bin/bash
# Complete setup and download script for Vast.ai instance
# This will be executed on the remote instance

set -e

echo "========================================"
echo "NexaraVision Complete Setup & Download"
echo "========================================"
echo ""

# Navigate to workspace
cd /workspace

# Install Kaggle if not present
pip3 install --quiet kaggle 2>/dev/null || echo "Kaggle already installed"

# Create directory structure
mkdir -p datasets/{tier1,tier2,tier3,processed} models logs checkpoints

# Configure Kaggle API
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"issadalu","key":"5aabafacbfdefea1bf4f2171d98cc52b"}
EOF
chmod 600 ~/.kaggle/kaggle.json

echo "✅ Setup complete!"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create download script
cat > /workspace/download_all_datasets.py <<'PYTHON_EOF'
#!/usr/bin/env python3
"""
NexaraVision - Complete Dataset Downloader
Downloads Tier 1, 2, and 3 datasets for violence detection
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime
import json

# Tier 1: Core Violence Detection (Highest Priority)
TIER1_DATASETS = [
    ('vulamnguyen/rwf2000', 'tier1/RWF2000', 'RWF-2000', 2000, 1.5),
    ('odins0n/ucf-crime-dataset', 'tier1/UCF_Crime', 'UCF-Crime', 1900, 12.0),
    ('toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd', 'tier1/SCVD', 'SmartCity-CCTV', 4000, 3.5),
    ('mohamedmustafa/real-life-violence-situations-dataset', 'tier1/RealLife', 'Real-Life Violence', 2000, 2.0),
    ('arnab91/eavdd-violence', 'tier1/EAVDD', 'EAVDD', 1500, 1.8),
]

# Tier 2: Extended Violence Datasets
TIER2_DATASETS = [
    ('nguhaduong/xd-violence-video-dataset', 'tier2/XDViolence', 'XD-Violence', 4754, 45.0),
]

# Tier 3: Non-Violence Normal Activity
TIER3_DATASETS = [
    # Add later if needed
]

def download_dataset(kaggle_id, path, name, expected_videos, expected_gb, tier):
    """Download a single dataset from Kaggle"""

    print(f"\n{'='*80}")
    print(f"📥 {tier} - {name}")
    print(f"{'='*80}")
    print(f"Dataset ID: {kaggle_id}")
    print(f"Expected: {expected_videos:,} videos (~{expected_gb} GB)")

    out_dir = Path(f"/workspace/datasets/{path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ['kaggle', 'datasets', 'download', '-d', kaggle_id, '-p', str(out_dir), '--unzip']

    start_time = datetime.now()

    try:
        print(f"\n⏳ Downloading...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            # Count videos
            video_count = 0
            for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI']:
                video_count += len(list(out_dir.rglob(ext)))

            # Calculate size
            size_gb = sum(f.stat().st_size for f in out_dir.rglob('*') if f.is_file()) / (1024**3)

            elapsed = (datetime.now() - start_time).total_seconds()

            print(f"\n✅ SUCCESS!")
            print(f"   Videos: {video_count:,}")
            print(f"   Size: {size_gb:.2f} GB")
            print(f"   Time: {elapsed:.1f}s")

            return {
                'name': name,
                'tier': tier,
                'status': 'success',
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'time_seconds': round(elapsed, 1),
                'path': str(out_dir)
            }
        else:
            print(f"\n❌ FAILED: {result.stderr}")
            return {
                'name': name,
                'tier': tier,
                'status': 'failed',
                'error': result.stderr[:200]
            }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return {
            'name': name,
            'tier': tier,
            'status': 'error',
            'error': str(e)[:200]
        }

def main():
    print("="*80)
    print("NexaraVision - Complete Dataset Download")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

    results = []

    # Download TIER 1 (Core Violence)
    print("\n" + "🔴"*40)
    print("TIER 1: Core Violence Detection Datasets")
    print("🔴"*40)

    for kaggle_id, path, name, videos, size in TIER1_DATASETS:
        result = download_dataset(kaggle_id, path, name, videos, size, "TIER 1")
        results.append(result)

    # Download TIER 2 (Extended - Optional)
    print("\n" + "🟡"*40)
    print("TIER 2: Extended Violence Datasets (Optional)")
    print("🟡"*40)
    print("Skipping for now - start with Tier 1 first")

    # Save results
    results_file = Path("/workspace/datasets/download_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("📊 DOWNLOAD SUMMARY")
    print("="*80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    total_videos = sum(r.get('videos', 0) for r in successful)
    total_size = sum(r.get('size_gb', 0) for r in successful)
    total_time = sum(r.get('time_seconds', 0) for r in successful)

    print(f"\n✅ Successful: {len(successful)}/{len(results)} datasets")
    print(f"📹 Total Videos: {total_videos:,}")
    print(f"💾 Total Size: {total_size:.2f} GB")
    print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if failed:
        print(f"\n❌ Failed: {len(failed)} datasets")
        for r in failed:
            print(f"   - {r['name']}: {r.get('error', 'Unknown error')[:100]}")

    print(f"\n📄 Results saved to: {results_file}")
    print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Final verification
    print("\n📂 Final Directory Structure:")
    os.system("tree -L 3 /workspace/datasets/ 2>/dev/null || find /workspace/datasets -type d | head -20")

    print("\n💾 Disk Usage:")
    os.system("du -sh /workspace/datasets/*")

if __name__ == "__main__":
    main()

PYTHON_EOF

chmod +x /workspace/download_all_datasets.py

echo ""
echo "========================================"
echo "Starting Dataset Download..."
echo "========================================"
echo ""

# Run the download script
python3 /workspace/download_all_datasets.py

echo ""
echo "========================================"
echo "Download Complete!"
echo "========================================"
