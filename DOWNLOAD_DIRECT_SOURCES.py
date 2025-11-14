#!/usr/bin/env python3
"""
Download Violence Detection Datasets from DIRECT SOURCES
Bypasses Kaggle 403 restrictions using alternative download methods
"""

import subprocess
import os
from pathlib import Path
import time

output_base = Path("/workspace/violence_datasets_direct")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING VIOLENCE DATASETS FROM DIRECT SOURCES")
print("Bypassing Kaggle restrictions using alternative methods")
print("="*80)
print()

# Install required tools
print("📦 Installing download tools...")
subprocess.run(['pip', 'install', '-q', 'gdown', 'kaggle'], check=False)
subprocess.run(['apt-get', 'update', '-qq'], check=False)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'git-lfs', 'unzip', 'p7zip-full'], check=False)
print("✅ Tools installed\n")

results = []

# ============================================================================
# METHOD 1: GOOGLE DRIVE DIRECT DOWNLOADS
# ============================================================================

print("\n" + "="*80)
print("METHOD 1: GOOGLE DRIVE DOWNLOADS")
print("="*80)

datasets_gdrive = [
    {
        'name': 'RWF-2000',
        'drive_id': '1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1',
        'expected_videos': 2000,
        'description': 'Real World Fight Dataset'
    },
    {
        'name': 'UCF-Crime',
        'drive_id': '1t0xR0SRZC_-eXyMRW_Mll_4s6IbGNH4I',
        'expected_videos': 1900,
        'description': 'UCF Crime Anomaly Detection'
    },
    {
        'name': 'RLVS',
        'drive_id': '1xm95qU8mB0GmRgEeT9kdPZdQVWPh_2jS',
        'expected_videos': 2000,
        'description': 'Real Life Violence Situations'
    },
]

for ds in datasets_gdrive:
    print(f"\n📥 Downloading {ds['name']} ({ds['description']})")
    print(f"   Expected: {ds['expected_videos']} videos")

    output_dir = output_base / ds['name']
    output_dir.mkdir(exist_ok=True)

    try:
        os.chdir(output_dir)

        # Download from Google Drive
        print("   Downloading from Google Drive...")
        result = subprocess.run(
            ['gdown', '--id', ds['drive_id'], '--fuzzy'],
            timeout=3600,
            capture_output=True,
            text=True
        )

        if result.returncode == 0 or any(output_dir.glob('*')):
            # Extract if zip
            for zipfile in output_dir.glob('*.zip'):
                print(f"   Extracting {zipfile.name}...")
                subprocess.run(['unzip', '-q', '-o', str(zipfile)], check=False)
                zipfile.unlink()

            # Extract if rar/7z
            for rarfile in list(output_dir.glob('*.rar')) + list(output_dir.glob('*.7z')):
                print(f"   Extracting {rarfile.name}...")
                subprocess.run(['7z', 'x', '-y', str(rarfile)], check=False)
                rarfile.unlink()

            # Count videos
            video_count = 0
            for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
                video_count += len(list(output_dir.rglob(ext)))

            size_gb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / (1024**3)

            print(f"   ✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
            results.append({
                'name': ds['name'],
                'method': 'Google Drive',
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'path': str(output_dir)
            })
        else:
            print(f"   ⚠️  Download failed, trying alternative...")
            results.append({'name': ds['name'], 'method': 'Google Drive', 'status': 'failed'})

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")
        results.append({'name': ds['name'], 'method': 'Google Drive', 'status': 'error', 'error': str(e)[:100]})

# ============================================================================
# METHOD 2: GITHUB LFS DOWNLOADS
# ============================================================================

print("\n" + "="*80)
print("METHOD 2: GITHUB LFS DOWNLOADS")
print("="*80)

os.chdir(output_base)

github_datasets = [
    {
        'name': 'Hockey-Fight',
        'url': 'https://github.com/seymanurakti/fight-detection-surv-dataset.git',
        'expected_videos': 1000,
    },
    {
        'name': 'AIRTLab',
        'url': 'https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git',
        'expected_videos': 350,
    },
    {
        'name': 'VioPeru',
        'url': 'https://github.com/hhuillcen/VioPeru.git',
        'expected_videos': 280,
    },
]

for ds in github_datasets:
    print(f"\n📥 Cloning {ds['name']}")
    print(f"   Expected: {ds['expected_videos']} videos")

    output_dir = output_base / ds['name']

    try:
        if not output_dir.exists():
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', ds['url'], str(output_dir)],
                timeout=1800,
                capture_output=True,
                text=True
            )

            # Pull LFS files if present
            if output_dir.exists():
                os.chdir(output_dir)
                subprocess.run(['git', 'lfs', 'pull'], timeout=1800, check=False)
                os.chdir(output_base)

        if output_dir.exists():
            video_count = 0
            for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
                video_count += len(list(output_dir.rglob(ext)))

            size_gb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / (1024**3)

            print(f"   ✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
            results.append({
                'name': ds['name'],
                'method': 'GitHub',
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'path': str(output_dir)
            })
        else:
            print(f"   ⚠️  Clone failed")
            results.append({'name': ds['name'], 'method': 'GitHub', 'status': 'failed'})

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")
        results.append({'name': ds['name'], 'method': 'GitHub', 'status': 'error', 'error': str(e)[:100]})

# ============================================================================
# METHOD 3: DIRECT HTTP DOWNLOADS
# ============================================================================

print("\n" + "="*80)
print("METHOD 3: DIRECT HTTP DOWNLOADS")
print("="*80)

http_datasets = [
    {
        'name': 'Violent-Flows',
        'url': 'https://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.tar.gz',
        'expected_videos': 250,
    },
]

for ds in http_datasets:
    print(f"\n📥 Downloading {ds['name']}")
    print(f"   Expected: {ds['expected_videos']} videos")

    output_dir = output_base / ds['name']
    output_dir.mkdir(exist_ok=True)

    try:
        os.chdir(output_dir)

        # Download with wget (more reliable than requests for large files)
        result = subprocess.run(
            ['wget', '-c', '-t', '5', '--timeout=300', ds['url']],
            timeout=1800,
            check=False
        )

        # Extract tar.gz
        for tarfile in output_dir.glob('*.tar.gz'):
            print(f"   Extracting {tarfile.name}...")
            subprocess.run(['tar', '-xzf', str(tarfile)], check=False)
            tarfile.unlink()

        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
            video_count += len(list(output_dir.rglob(ext)))

        size_gb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / (1024**3)

        if video_count > 0:
            print(f"   ✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
            results.append({
                'name': ds['name'],
                'method': 'HTTP',
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'path': str(output_dir)
            })
        else:
            print(f"   ⚠️  No videos found")
            results.append({'name': ds['name'], 'method': 'HTTP', 'status': 'no_videos'})

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")
        results.append({'name': ds['name'], 'method': 'HTTP', 'status': 'error', 'error': str(e)[:100]})

# ============================================================================
# METHOD 4: HUGGINGFACE DATASETS
# ============================================================================

print("\n" + "="*80)
print("METHOD 4: HUGGINGFACE DOWNLOADS")
print("="*80)

subprocess.run(['pip', 'install', '-q', 'huggingface-hub'], check=False)

hf_datasets = [
    {
        'name': 'UCF-Crime-HF',
        'repo': 'datasets/ucf-crime',
        'expected_videos': 1900,
    },
]

for ds in hf_datasets:
    print(f"\n📥 Downloading {ds['name']} from HuggingFace")

    output_dir = output_base / ds['name']
    output_dir.mkdir(exist_ok=True)

    try:
        os.chdir(output_dir)

        result = subprocess.run(
            ['huggingface-cli', 'download', ds['repo'], '--repo-type', 'dataset', '--local-dir', '.'],
            timeout=3600,
            check=False
        )

        video_count = 0
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov']:
            video_count += len(list(output_dir.rglob(ext)))

        if video_count > 0:
            size_gb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / (1024**3)
            print(f"   ✅ SUCCESS: {video_count} videos ({size_gb:.2f} GB)")
            results.append({
                'name': ds['name'],
                'method': 'HuggingFace',
                'videos': video_count,
                'size_gb': round(size_gb, 2),
                'path': str(output_dir)
            })
        else:
            print(f"   ⚠️  No videos found")
            results.append({'name': ds['name'], 'method': 'HuggingFace', 'status': 'no_videos'})

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")
        results.append({'name': ds['name'], 'method': 'HuggingFace', 'status': 'error', 'error': str(e)[:100]})

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print()

successful = [r for r in results if 'videos' in r and r['videos'] > 0]
total_videos = sum(r['videos'] for r in successful)
total_size = sum(r['size_gb'] for r in successful)

print(f"✅ Successfully downloaded: {len(successful)} datasets")
print(f"📹 Total videos: {total_videos:,}")
print(f"💾 Total size: {total_size:.2f} GB")
print()

if successful:
    print("DOWNLOADED DATASETS:")
    for r in successful:
        print(f"  ✅ {r['name']} ({r['method']})")
        print(f"     Videos: {r['videos']:,} ({r['size_gb']} GB)")
        print(f"     Path: {r['path']}")
    print()

print(f"📁 All datasets saved to: {output_base}")
print("="*80)
