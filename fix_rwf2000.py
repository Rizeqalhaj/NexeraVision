#!/usr/bin/env python3
"""
Fix RWF-2000 Dataset Download and Extraction
Handles naming issues with manual extraction to pre-named directory
"""

import subprocess
import os
import zipfile
from pathlib import Path
from datetime import datetime

def fix_rwf2000():
    """Download and manually extract RWF-2000 to /workspace/datasets/tier1/RWF2000"""

    print("=" * 80)
    print("RWF-2000 Dataset Fix - Manual Extraction")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup paths
    tier1_dir = Path("/workspace/datasets/tier1")
    rwf2000_dir = tier1_dir / "RWF2000"
    temp_zip = tier1_dir / "rwf2000_temp.zip"

    # Create target directory
    rwf2000_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Target directory created: {rwf2000_dir}\n")

    # Download without unzip
    print("📥 Downloading RWF-2000 from Kaggle...")
    print("   (downloading to temp file, will extract manually)\n")

    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'vulamnguyen/rwf2000', '-p', str(tier1_dir)],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )

        if result.returncode != 0:
            print(f"❌ Download failed: {result.stderr}")
            return False

        print("✅ Download complete!\n")

        # Find the downloaded zip file
        zip_files = list(tier1_dir.glob("*.zip"))
        if not zip_files:
            print("❌ No zip file found after download")
            return False

        zip_file = zip_files[0]
        print(f"📦 Found zip file: {zip_file.name}")
        print(f"   Size: {zip_file.stat().st_size / (1024**3):.2f} GB\n")

        # Extract manually to pre-named directory
        print(f"📂 Extracting to: {rwf2000_dir}")
        print("   (this may take a few minutes...)\n")

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get total files
            total_files = len(zip_ref.namelist())
            print(f"   Total files in archive: {total_files:,}")

            # Extract all
            zip_ref.extractall(rwf2000_dir)

        print("\n✅ Extraction complete!\n")

        # Count videos
        print("📹 Counting video files...")
        video_count = 0
        video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI', '*.MKV']

        for ext in video_extensions:
            video_count += len(list(rwf2000_dir.rglob(ext)))

        # Calculate size
        total_size = sum(f.stat().st_size for f in rwf2000_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)

        print(f"✅ Videos found: {video_count:,}")
        print(f"✅ Total size: {size_gb:.2f} GB\n")

        # Clean up zip file
        print("🧹 Cleaning up temporary zip file...")
        zip_file.unlink()
        print("✅ Cleanup complete!\n")

        print("=" * 80)
        print("✅ RWF-2000 SUCCESSFULLY FIXED!")
        print("=" * 80)
        print(f"📂 Location: {rwf2000_dir}")
        print(f"📹 Videos: {video_count:,}")
        print(f"💾 Size: {size_gb:.2f} GB")
        print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = fix_rwf2000()
    exit(0 if success else 1)
