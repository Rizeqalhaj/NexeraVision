#!/usr/bin/env python3
"""
Download Official UCF-Crime Dataset
Real surveillance videos from UCF Center for Research in Computer Vision
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime
import shutil

def download_ucf_crime():
    """Download official UCF-Crime dataset from UCF CRCV"""

    print("=" * 80)
    print("Official UCF-Crime Dataset Download")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    tier1_dir = Path("/workspace/datasets/tier1")
    ucf_crime_dir = tier1_dir / "UCF_Crime"

    # Remove wrong dataset
    if ucf_crime_dir.exists():
        print(f"🧹 Removing incorrect UCF_Crime dataset (PNG files)...")
        shutil.rmtree(ucf_crime_dir)
        print("✅ Removed\n")

    ucf_crime_dir.mkdir(parents=True, exist_ok=True)

    # Official UCF-Crime dataset sources
    # The official dataset is hosted at UCF CRCV

    print("📥 Downloading Official UCF-Crime Dataset")
    print("   Source: UCF Center for Research in Computer Vision")
    print("   Videos: 1,900 untrimmed surveillance videos")
    print("   Size: ~12 GB\n")

    # Try official UCF source
    official_url = "http://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"

    print(f"⏳ Downloading from: {official_url}")
    print("   (this may take 15-30 minutes for 12GB)\n")

    try:
        # Download with wget (more reliable for large files)
        result = subprocess.run(
            ['wget', '-c', '--progress=bar:force', '-O',
             str(ucf_crime_dir / 'UCF_Crimes.zip'),
             official_url],
            timeout=3600,  # 1 hour timeout
            capture_output=False  # Show progress
        )

        if result.returncode != 0:
            print(f"\n❌ Download failed")
            print("\n📋 Alternative download options:")
            print("   1. Manual download: http://www.crcv.ucf.edu/projects/real-world/")
            print("   2. Google Drive mirror (check research paper)")
            print("   3. Contact UCF CRCV for dataset access")
            return False

        print("\n✅ Download complete!")

        # Extract
        zip_file = ucf_crime_dir / 'UCF_Crimes.zip'

        if zip_file.exists():
            print(f"\n📦 Extracting videos...")
            print(f"   From: {zip_file.name}")
            print(f"   To: {ucf_crime_dir}/\n")

            result = subprocess.run(
                ['unzip', '-q', str(zip_file), '-d', str(ucf_crime_dir)],
                timeout=600
            )

            if result.returncode == 0:
                print("✅ Extraction complete!")

                # Count videos
                video_count = 0
                video_exts = ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI']

                for ext in video_exts:
                    video_count += len(list(ucf_crime_dir.rglob(ext)))

                # Calculate size
                total_size = sum(f.stat().st_size for f in ucf_crime_dir.rglob('*') if f.is_file() and not f.name.endswith('.zip'))
                size_gb = total_size / (1024**3)

                print(f"\n📹 Videos found: {video_count:,}")
                print(f"💾 Total size: {size_gb:.2f} GB")

                # Clean up zip
                print(f"\n🧹 Cleaning up zip file...")
                zip_file.unlink()
                print("✅ Cleanup complete!")

                print("\n" + "=" * 80)
                print("✅ UCF-CRIME DATASET READY!")
                print("=" * 80)
                print(f"📂 Location: {ucf_crime_dir}")
                print(f"📹 Videos: {video_count:,}")
                print(f"💾 Size: {size_gb:.2f} GB")
                print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)

                return True
            else:
                print("❌ Extraction failed")
                return False
        else:
            print("❌ Zip file not found after download")
            return False

    except subprocess.TimeoutExpired:
        print("\n❌ Download timed out (>1 hour)")
        print("   Network may be too slow. Try:")
        print("   1. Download manually and upload to instance")
        print("   2. Use faster network connection")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

def main():
    """Main function"""

    # Check if wget is available
    result = subprocess.run(['which', 'wget'], capture_output=True)
    if result.returncode != 0:
        print("Installing wget...")
        subprocess.run(['apt-get', 'update', '-qq'], check=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'wget'], check=True)

    # Check if unzip is available
    result = subprocess.run(['which', 'unzip'], capture_output=True)
    if result.returncode != 0:
        print("Installing unzip...")
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'unzip'], check=True)

    success = download_ucf_crime()

    if not success:
        print("\n" + "=" * 80)
        print("⚠️  MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 80)
        print("\nIf automatic download failed, download manually:")
        print("\n1. Visit: http://www.crcv.ucf.edu/projects/real-world/")
        print("2. Download: UCF_Crimes.zip (12 GB)")
        print("3. Upload to: /workspace/datasets/tier1/UCF_Crime/")
        print("4. Extract: unzip UCF_Crimes.zip")
        print("\nAlternative Kaggle datasets to try:")
        print("  - kaggle datasets download -d mission-ai/crimeucfdataset")
        print("  - kaggle datasets download -d mateohervas/ucf-crime-dataset")
        print("=" * 80)

if __name__ == "__main__":
    main()
