#!/usr/bin/env python3
"""
Download UCF Crime Dataset from official university source
https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset
"""

import subprocess
import requests
from pathlib import Path
import shutil
from bs4 import BeautifulSoup

output_base = Path("/workspace/ucf_crime_official")
output_base.mkdir(exist_ok=True)

print("="*80)
print("UCF CRIME DATASET - OFFICIAL DOWNLOAD")
print("Source: University of North Carolina Charlotte")
print("="*80)
print()

# Known download links for UCF Crime dataset
download_links = [
    # Main dataset
    "http://www.crcv.ucf.edu/projects/real-world/",
    # Direct Google Drive links (common mirrors)
    "https://drive.google.com/file/d/1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1/view",  # Full dataset
]

print("Installing download tools...")
subprocess.run(['pip', 'install', '-q', 'gdown', 'requests', 'beautifulsoup4'], check=False)

# ============================================================================
# METHOD 1: Try Google Drive direct download
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: Google Drive Download")
print("="*80)
print()

gdrive_ids = [
    '1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1',  # UCF Crime full
    '1t0xR0SRZC_-eXyMRW_Mll_4s6IbGNH4I',  # Alternative mirror
]

success = False

for gid in gdrive_ids:
    print(f"Trying Google Drive ID: {gid}")

    try:
        cmd = ['gdown', '--id', gid, '-O', str(output_base / 'ucf_crime.zip')]
        result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout

        if result.returncode == 0:
            print("✅ Download successful!")
            success = True
            break
        else:
            print("⚠️  Download failed, trying next mirror...")

    except subprocess.TimeoutExpired:
        print("⏱️  Timeout, trying next method...")
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")

# ============================================================================
# METHOD 2: Try wget direct download
# ============================================================================
if not success:
    print("\n" + "="*80)
    print("METHOD 2: Direct wget Download")
    print("="*80)
    print()

    wget_urls = [
        "http://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset/download",
        "http://www.crcv.ucf.edu/data/UCF_Crimes.zip",
    ]

    for url in wget_urls:
        print(f"Trying: {url}")

        try:
            cmd = ['wget', '-c', '-O', str(output_base / 'ucf_crime.zip'), url]
            result = subprocess.run(cmd, timeout=7200)

            if result.returncode == 0:
                print("✅ Download successful!")
                success = True
                break

        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}")

# ============================================================================
# EXTRACT
# ============================================================================
if success or (output_base / 'ucf_crime.zip').exists():
    print("\n" + "="*80)
    print("EXTRACTING DATASET")
    print("="*80)
    print()

    zipfile = output_base / 'ucf_crime.zip'

    if zipfile.exists():
        print(f"Extracting {zipfile.name}...")

        try:
            # Try unzip
            subprocess.run(['unzip', '-q', str(zipfile), '-d', str(output_base)])
            print("✅ Extracted with unzip")

        except:
            # Try Python zipfile
            import zipfile as zf
            with zf.ZipFile(zipfile, 'r') as zip_ref:
                zip_ref.extractall(output_base)
            print("✅ Extracted with Python")

        # Delete zip
        zipfile.unlink()
        print(f"✓ Deleted {zipfile.name}")

        # ====================================================================
        # FILTER FOR SPECIFIC CATEGORIES
        # ====================================================================
        print("\n" + "="*80)
        print("FILTERING CATEGORIES")
        print("Keeping: Abuse, Fighting, Assault, NormalVideos")
        print("="*80)
        print()

        filtered_dir = output_base / "filtered"
        filtered_dir.mkdir(exist_ok=True)

        keep_categories = ['abuse', 'fighting', 'assault', 'normal']

        # Search for matching directories
        for item in output_base.rglob('*'):
            if item.is_dir() and item != filtered_dir:
                dir_name_lower = item.name.lower()

                if any(cat in dir_name_lower for cat in keep_categories):
                    print(f"  Found: {item.name}")

                    output_cat = filtered_dir / item.name
                    output_cat.mkdir(exist_ok=True)

                    # Copy videos
                    for video in item.rglob('*'):
                        if video.is_file() and video.suffix.lower() in ['.mp4', '.avi', '.mkv', '.webm', '.mov']:
                            dest = output_cat / video.name
                            shutil.copy(str(video), str(dest))

        # Count videos
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print()

        total_videos = 0
        for subdir in filtered_dir.iterdir():
            if subdir.is_dir():
                count = sum(1 for _ in subdir.rglob('*.mp4')) + sum(1 for _ in subdir.rglob('*.avi'))
                if count > 0:
                    total_videos += count
                    print(f"  {subdir.name}: {count} videos")

        total_size = sum(f.stat().st_size for f in filtered_dir.rglob('*') if f.is_file()) / (1024**3)

        print()
        print(f"✅ Total: {total_videos} videos ({total_size:.2f} GB)")
        print(f"📁 Saved to: {filtered_dir}")

    else:
        print("❌ Zip file not found!")
else:
    print("\n❌ DOWNLOAD FAILED")
    print()
    print("Manual download instructions:")
    print("1. Go to: https://www.crcv.ucf.edu/projects/real-world/")
    print("2. Download the dataset manually")
    print("3. Upload to /workspace/ucf_crime_official/ucf_crime.zip")
    print("4. Run this script again to extract")

print("\n" + "="*80)
