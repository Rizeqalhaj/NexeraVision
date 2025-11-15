#!/usr/bin/env python3
"""
Try Alternative Kaggle Sources for UCF-Crime Videos
Tests multiple Kaggle datasets to find the real video version
"""

import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Alternative Kaggle datasets that might have UCF-Crime videos
ALTERNATIVES = [
    ('mission-ai/crimeucfdataset', 'Alternative 1: mission-ai'),
    ('mateohervas/ucf-crime-dataset', 'Alternative 2: mateohervas'),
    ('datasets/ucfcrime', 'Alternative 3: datasets'),
]

def try_kaggle_alternative(kaggle_id, name):
    """Try downloading from alternative Kaggle source"""

    print("=" * 80)
    print(f"Trying: {name}")
    print(f"Kaggle ID: {kaggle_id}")
    print("=" * 80)
    print()

    tier1_dir = Path("/workspace/datasets/tier1")
    temp_dir = tier1_dir / "UCF_Crime_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"⏳ Downloading...")

        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', kaggle_id, '-p', str(temp_dir), '--unzip'],
            capture_output=True,
            text=True,
            timeout=1800
        )

        if result.returncode == 0:
            print("✅ Download successful!")

            # Check for video files
            video_count = 0
            video_exts = ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']

            for ext in video_exts:
                video_count += len(list(temp_dir.rglob(ext)))

            # Check for image files
            image_count = 0
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                image_count += len(list(temp_dir.rglob(ext)))

            print(f"\n📊 Content check:")
            print(f"   Videos: {video_count:,}")
            print(f"   Images: {image_count:,}")

            if video_count > 100:
                print(f"\n✅ FOUND VIDEOS! This source has {video_count:,} video files")

                # Calculate size
                total_size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
                size_gb = total_size / (1024**3)
                print(f"   Total size: {size_gb:.2f} GB")

                # Move to final location
                ucf_crime_dir = tier1_dir / "UCF_Crime"
                if ucf_crime_dir.exists():
                    shutil.rmtree(ucf_crime_dir)

                temp_dir.rename(ucf_crime_dir)

                print(f"\n📂 Moved to: {ucf_crime_dir}")
                print("\n" + "=" * 80)
                print("✅ SUCCESS! UCF-Crime videos downloaded")
                print("=" * 80)

                return True
            else:
                print(f"\n❌ This source only has images/PNGs, not videos")
                shutil.rmtree(temp_dir)
                return False
        else:
            print(f"❌ Download failed: {result.stderr[:200]}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return False

    except subprocess.TimeoutExpired:
        print("❌ Download timed out")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False

def main():
    """Try all alternative sources"""

    print("=" * 80)
    print("UCF-Crime Alternative Kaggle Sources")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Testing {len(ALTERNATIVES)} alternative sources...\n")

    for kaggle_id, name in ALTERNATIVES:
        success = try_kaggle_alternative(kaggle_id, name)

        if success:
            print(f"\n🎉 Found working source: {name}")
            print("   No need to try other sources")
            return

        print("\n")

    print("=" * 80)
    print("⚠️  No video sources found on Kaggle")
    print("=" * 80)
    print("\nRecommendation: Download from official UCF source")
    print("Run: python3 download_real_ucf_crime.py")
    print("\nOr download manually:")
    print("  http://www.crcv.ucf.edu/projects/real-world/")

if __name__ == "__main__":
    main()
