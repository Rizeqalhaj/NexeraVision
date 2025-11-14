#!/usr/bin/env python3
"""
Download UCF Crime Dataset - NORMAL VIDEOS ONLY
Filter for non-violence activities
"""

import subprocess
from pathlib import Path
import shutil

output_base = Path("/workspace/ucf_normal_videos")
output_base.mkdir(exist_ok=True)

print("="*80)
print("UCF CRIME - NORMAL VIDEOS ONLY")
print("Downloading NON-VIOLENCE activities only")
print("="*80)
print()

subprocess.run(['pip', 'install', '-q', 'gdown'], check=False)

# Download from Google Drive
print("Downloading UCF Crime dataset...")
print("(This is ~13GB, may take 30+ minutes)")
print()

gdrive_ids = [
    '1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1',
    '1t0xR0SRZC_-eXyMRW_Mll_4s6IbGNH4I',
]

success = False
for gid in gdrive_ids:
    print(f"Trying Google Drive ID: {gid}")

    try:
        cmd = ['gdown', '--id', gid, '-O', str(output_base / 'ucf_crime.zip')]
        result = subprocess.run(cmd, timeout=7200)

        if result.returncode == 0:
            print("✅ Download complete!")
            success = True
            break
    except:
        print("Failed, trying next mirror...")

if success or (output_base / 'ucf_crime.zip').exists():
    print("\nExtracting...")

    zipfile = output_base / 'ucf_crime.zip'

    if zipfile.exists():
        subprocess.run(['unzip', '-q', str(zipfile), '-d', str(output_base)])
        zipfile.unlink()

        print("\n" + "="*80)
        print("FILTERING - NORMAL VIDEOS ONLY")
        print("="*80)
        print()

        normal_dir = output_base / "normal_only"
        normal_dir.mkdir(exist_ok=True)

        # Find Normal/Non-violence directories
        for item in output_base.rglob('*'):
            if item.is_dir():
                name_lower = item.name.lower()

                # Keep ONLY normal/non-violence
                if 'normal' in name_lower or 'non' in name_lower:
                    print(f"  Found: {item.name}")

                    output_cat = normal_dir / item.name
                    output_cat.mkdir(exist_ok=True)

                    for video in item.rglob('*'):
                        if video.is_file() and video.suffix.lower() in ['.mp4', '.avi']:
                            dest = output_cat / video.name
                            shutil.copy(str(video), str(dest))

        # Count
        total = sum(1 for _ in normal_dir.rglob('*.mp4')) + sum(1 for _ in normal_dir.rglob('*.avi'))
        size = sum(f.stat().st_size for f in normal_dir.rglob('*') if f.is_file()) / (1024**3)

        print()
        print(f"✅ Total normal videos: {total} ({size:.2f} GB)")
        print(f"📁 Saved to: {normal_dir}")

        # Delete violence videos
        for item in output_base.iterdir():
            if item.is_dir() and item != normal_dir:
                shutil.rmtree(item, ignore_errors=True)

        print("\n✓ Deleted violence categories")

else:
    print("\n❌ Download failed")

print("\n" + "="*80)
