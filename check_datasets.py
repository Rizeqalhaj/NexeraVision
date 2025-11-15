#!/usr/bin/env python3
"""
Check Downloaded Datasets Status
Shows what was successfully downloaded and what needs fixing
"""

import json
from pathlib import Path

def check_datasets():
    """Check dataset download status"""

    print("=" * 80)
    print("NexaraVision Dataset Status Check")
    print("=" * 80)
    print()

    # Check results file
    results_file = Path("/workspace/datasets/download_results.json")

    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)

        print("📊 Download Results Summary:")
        print()

        successful = [r for r in data['results'] if r['status'] == 'success']
        failed = [r for r in data['results'] if r['status'] != 'success']

        print(f"✅ Successful: {len(successful)}/5 datasets")
        print(f"❌ Failed: {len(failed)}/5 datasets")
        print()

        if successful:
            print("=" * 80)
            print("✅ SUCCESSFUL DOWNLOADS:")
            print("=" * 80)
            for r in successful:
                print(f"\n📦 {r['name']}")
                print(f"   Videos: {r.get('videos', 'N/A'):,}")
                print(f"   Size: {r.get('size_gb', 'N/A')} GB")
                print(f"   Time: {r.get('time_s', 0)/60:.1f} min")
                print(f"   Path: {r.get('path', 'N/A')}")

        if failed:
            print()
            print("=" * 80)
            print("❌ FAILED DOWNLOADS:")
            print("=" * 80)
            for r in failed:
                print(f"\n📦 {r['name']}")
                print(f"   Status: {r['status']}")
                print(f"   Error: {r.get('error', 'Unknown')[:200]}")

    # Check actual directories
    print()
    print("=" * 80)
    print("📁 Directory Structure:")
    print("=" * 80)

    tier1_dir = Path("/workspace/datasets/tier1")
    if tier1_dir.exists():
        for item in sorted(tier1_dir.iterdir()):
            if item.is_dir():
                # Count videos
                video_count = 0
                for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI', '*.MKV']:
                    video_count += len(list(item.rglob(ext)))

                # Get size
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                size_gb = size / (1024**3)

                print(f"\n📂 {item.name}/")
                print(f"   Videos: {video_count:,}")
                print(f"   Size: {size_gb:.2f} GB")

    print()
    print("=" * 80)

if __name__ == "__main__":
    check_datasets()
