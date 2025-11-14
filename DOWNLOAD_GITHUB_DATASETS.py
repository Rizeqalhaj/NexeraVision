#!/usr/bin/env python3
"""
Download Violence Detection Datasets from GitHub
Finds and downloads actual video data, not just code
"""

import subprocess
import os
from pathlib import Path
import re

output_base = Path("/workspace/violence_datasets_github")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING GITHUB VIOLENCE DETECTION DATASETS")
print("Finding actual video data locations...")
print("="*80)
print()

datasets = []

# ============================================================================
# DATASET 1: RWF-2000
# ============================================================================
print("\n" + "="*80)
print("1. RWF-2000 (Real World Fight Dataset)")
print("="*80)
print("Source: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection")
print()

rwf_dir = output_base / "RWF-2000"
rwf_dir.mkdir(exist_ok=True)

print("📥 Downloading RWF-2000 videos from Google Drive...")
print("   Video data hosted at: Google Drive ID 1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1")
print()

# Install gdown if needed
subprocess.run(['pip', 'install', '--break-system-packages', '-q', 'gdown'], check=False)

# Download from Google Drive
try:
    os.chdir(rwf_dir)
    # Main dataset
    print("   Downloading main dataset (compressed)...")
    subprocess.run(['gdown', '--id', '1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1'], timeout=3600)

    # Try alternative link
    if not list(rwf_dir.glob('*.zip')):
        print("   Trying alternative link...")
        subprocess.run(['gdown', '--id', '1-1gvPCjmMf4TH4PjLM9xTf3tqFKVBL8W'], timeout=3600)

    # Extract
    for zipfile in rwf_dir.glob('*.zip'):
        print(f"   Extracting {zipfile.name}...")
        subprocess.run(['unzip', '-q', str(zipfile)])
        zipfile.unlink()

    # Count videos
    video_count = len(list(rwf_dir.rglob('*.avi'))) + len(list(rwf_dir.rglob('*.mp4')))
    print(f"✅ RWF-2000: {video_count} videos downloaded")
    datasets.append({'name': 'RWF-2000', 'videos': video_count, 'path': str(rwf_dir)})

except Exception as e:
    print(f"❌ RWF-2000 failed: {e}")
    datasets.append({'name': 'RWF-2000', 'videos': 0, 'error': str(e)})

# ============================================================================
# DATASET 2: Hockey Fight Detection
# ============================================================================
print("\n" + "="*80)
print("2. Hockey Fight Detection Dataset")
print("="*80)
print("Source: http://visilab.etsii.uclm.es/personas/oscar/FightDetection/")
print()

hockey_dir = output_base / "Hockey_Fight"
hockey_dir.mkdir(exist_ok=True)

print("📥 Downloading Hockey Fight videos...")

try:
    os.chdir(hockey_dir)
    # Download videos.zip
    subprocess.run(['wget', '-c', 'http://visilab.etsii.uclm.es/personas/oscar/FightDetection/videos.zip'], timeout=1800)

    # Extract
    subprocess.run(['unzip', '-q', 'videos.zip'])
    Path('videos.zip').unlink()

    video_count = len(list(hockey_dir.rglob('*.avi'))) + len(list(hockey_dir.rglob('*.mp4')))
    print(f"✅ Hockey Fight: {video_count} videos downloaded")
    datasets.append({'name': 'Hockey Fight', 'videos': video_count, 'path': str(hockey_dir)})

except Exception as e:
    print(f"❌ Hockey Fight failed: {e}")
    datasets.append({'name': 'Hockey Fight', 'videos': 0, 'error': str(e)})

# ============================================================================
# DATASET 3: Violent Flows
# ============================================================================
print("\n" + "="*80)
print("3. Violent Flows Dataset")
print("="*80)
print("Source: https://www.openu.ac.il/home/hassner/data/violentflows/")
print()

vf_dir = output_base / "Violent_Flows"
vf_dir.mkdir(exist_ok=True)

print("📥 Downloading Violent Flows videos...")

try:
    os.chdir(vf_dir)
    # Download tar.gz
    subprocess.run(['wget', '-c', 'https://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.tar.gz'], timeout=1800)

    # Extract
    subprocess.run(['tar', '-xzf', 'violent_flows.tar.gz'])
    Path('violent_flows.tar.gz').unlink()

    video_count = len(list(vf_dir.rglob('*.avi'))) + len(list(vf_dir.rglob('*.mp4')))
    print(f"✅ Violent Flows: {video_count} videos downloaded")
    datasets.append({'name': 'Violent Flows', 'videos': video_count, 'path': str(vf_dir)})

except Exception as e:
    print(f"❌ Violent Flows failed: {e}")
    datasets.append({'name': 'Violent Flows', 'videos': 0, 'error': str(e)})

# ============================================================================
# DATASET 4: CAVIAR Dataset
# ============================================================================
print("\n" + "="*80)
print("4. CAVIAR Dataset (Surveillance)")
print("="*80)
print("Source: https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/")
print()

caviar_dir = output_base / "CAVIAR"
caviar_dir.mkdir(exist_ok=True)

print("📥 Downloading CAVIAR videos...")

try:
    os.chdir(caviar_dir)

    # CAVIAR has multiple video sets
    video_sets = [
        'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/CAVIAR_VIDEOS/EnterExitCrossingPaths1nt.mpg',
        'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/CAVIAR_VIDEOS/EnterExitCrossingPaths2nt.mpg',
        'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/CAVIAR_VIDEOS/WalkByShop1nt.mpg',
        'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/CAVIAR_VIDEOS/Browse1.mpg',
        'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/CAVIAR_VIDEOS/Fight_OneManDown.mpg',
        'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/CAVIAR_VIDEOS/Fight_RunAway1.mpg',
        'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/CAVIAR_VIDEOS/Fight_RunAway2.mpg',
    ]

    for url in video_sets:
        try:
            subprocess.run(['wget', '-c', url], timeout=300)
        except:
            continue

    video_count = len(list(caviar_dir.rglob('*.mpg'))) + len(list(caviar_dir.rglob('*.avi')))
    print(f"✅ CAVIAR: {video_count} videos downloaded")
    datasets.append({'name': 'CAVIAR', 'videos': video_count, 'path': str(caviar_dir)})

except Exception as e:
    print(f"❌ CAVIAR failed: {e}")
    datasets.append({'name': 'CAVIAR', 'videos': 0, 'error': str(e)})

# ============================================================================
# DATASET 5: UT-Interaction Dataset
# ============================================================================
print("\n" + "="*80)
print("5. UT-Interaction Dataset")
print("="*80)
print("Source: http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html")
print()

ut_dir = output_base / "UT_Interaction"
ut_dir.mkdir(exist_ok=True)

print("📥 Downloading UT-Interaction videos...")

try:
    os.chdir(ut_dir)

    # Download video sets
    subprocess.run(['wget', '-c', 'http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1.avi'], timeout=600)
    subprocess.run(['wget', '-c', 'http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_2.avi'], timeout=600)

    video_count = len(list(ut_dir.rglob('*.avi')))
    print(f"✅ UT-Interaction: {video_count} videos downloaded")
    datasets.append({'name': 'UT-Interaction', 'videos': video_count, 'path': str(ut_dir)})

except Exception as e:
    print(f"❌ UT-Interaction failed: {e}")
    datasets.append({'name': 'UT-Interaction', 'videos': 0, 'error': str(e)})

# ============================================================================
# DATASET 6: UCF Crime (Google Drive)
# ============================================================================
print("\n" + "="*80)
print("6. UCF Crime Dataset")
print("="*80)
print("Source: Multiple mirrors (Google Drive)")
print()

ucf_dir = output_base / "UCF_Crime"
ucf_dir.mkdir(exist_ok=True)

print("📥 Downloading UCF Crime videos...")
print("   This is a large dataset (~13 GB) - may take a while...")

try:
    os.chdir(ucf_dir)

    # Try Google Drive links
    drive_ids = [
        '1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1',  # Main dataset
        '1t0xR0SRZC_-eXyMRW_Mll_4s6IbGNH4I',  # Alternative
    ]

    for drive_id in drive_ids:
        try:
            print(f"   Trying Google Drive ID: {drive_id}")
            subprocess.run(['gdown', '--id', drive_id], timeout=3600)

            # Check if we got a zip file
            if list(ucf_dir.glob('*.zip')):
                break
        except:
            continue

    # Extract any zip files
    for zipfile in ucf_dir.glob('*.zip'):
        print(f"   Extracting {zipfile.name}...")
        subprocess.run(['unzip', '-q', str(zipfile)])
        zipfile.unlink()

    video_count = len(list(ucf_dir.rglob('*.mp4'))) + len(list(ucf_dir.rglob('*.avi')))
    print(f"✅ UCF Crime: {video_count} videos downloaded")
    datasets.append({'name': 'UCF Crime', 'videos': video_count, 'path': str(ucf_dir)})

except Exception as e:
    print(f"❌ UCF Crime failed: {e}")
    datasets.append({'name': 'UCF Crime', 'videos': 0, 'error': str(e)})

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)
print()

total_videos = sum(d['videos'] for d in datasets)
successful = sum(1 for d in datasets if d['videos'] > 0)

print(f"✅ Successful: {successful}/{len(datasets)} datasets")
print(f"📹 Total videos: {total_videos}")
print()

for dataset in datasets:
    if dataset['videos'] > 0:
        print(f"✅ {dataset['name']:20} {dataset['videos']:6} videos")
        print(f"   Path: {dataset['path']}")
    else:
        print(f"❌ {dataset['name']:20} Failed: {dataset.get('error', 'unknown')}")

print()
print(f"📁 All datasets saved to: {output_base}")
print("="*80)
