#!/usr/bin/env python3
"""
Download VERIFIED WORKING Kaggle Violence Datasets
Based on web search - these are confirmed to be public and accessible
"""

import subprocess
import os
from pathlib import Path
import shutil
import json

# Setup Kaggle credentials
kaggle_json = Path("kaggle.json")
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

if kaggle_json.exists():
    shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    print("✅ Kaggle credentials configured\n")

output_base = Path("/workspace/violence_datasets_verified")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING VERIFIED WORKING KAGGLE DATASETS")
print("From web search - confirmed public access")
print("="*80)
print()

# VERIFIED WORKING DATASETS FROM WEB SEARCH
datasets = [
    # 2024 Datasets (Most Recent)
    {
        'name': 'arnab91/eavdd-violence',
        'description': 'EAVDD Violence Dataset (July 2024)',
        'expected': 'video files',
    },
    {
        'name': 'yash07yadav/project-data',
        'description': 'Violence Detection - Combined (Sept 2024)',
        'expected': 'video files',
    },
    {
        'name': 'toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd',
        'description': 'Smart-City CCTV Violence (Dec 2023)',
        'expected': 'CCTV video files',
    },

    # Recent Public Datasets
    {
        'name': 'naveenk903/movies-fight-detection-dataset',
        'description': 'Video Fight Detection Dataset (2021)',
        'expected': 'fight videos',
    },
    {
        'name': 'shreyj1729/cctv-fights-dataset',
        'description': 'Video Fights Dataset - CCTV (2020)',
        'expected': 'CCTV fight footage',
    },
    {
        'name': 'mohamedmustafa/real-life-violence-situations-dataset',
        'description': 'Real Life Violence Situations (2019)',
        'expected': '2000 videos',
    },
    {
        'name': 'karandeep98/real-life-violence-and-nonviolence-data',
        'description': 'Real Life Violence and Non-Violence (2021)',
        'expected': 'video data',
    },
    {
        'name': 'khushhalreddy/violence-detection-dataset',
        'description': 'Violence Detection Dataset',
        'expected': 'video files',
    },

    # Additional potential datasets
    {
        'name': 'anbumalar1991/fight-dataset',
        'description': 'Fight Dataset (2019)',
        'expected': 'fight videos',
    },
]

results = []

for i, ds in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(datasets)}] {ds['description']}")
    print(f"Dataset: {ds['name']}")
    print(f"{'='*80}")

    output_dir = output_base / ds['name'].replace('/', '_')
    output_dir.mkdir(exist_ok=True)

    try:
        # Download
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', ds['name'],
            '-p', str(output_dir),
            '--unzip'
        ]

        print("Downloading...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            # Count videos
            video_count = 0
            for ext in ['*.mp4', '*.avi', '*.mkv', '*.webm', '*.mov', '*.MP4', '*.AVI']:
                video_count += len(list(output_dir.rglob(ext)))

            # Count images (some datasets might be frames)
            image_count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
                image_count += len(list(output_dir.rglob(ext)))

            # Calculate size
            total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)

            if video_count > 0 or image_count > 0:
                print(f"✅ SUCCESS!")
                print(f"   Videos: {video_count}")
                print(f"   Images: {image_count}")
                print(f"   Size: {size_gb:.2f} GB")

                results.append({
                    'dataset': ds['name'],
                    'description': ds['description'],
                    'status': 'success',
                    'videos': video_count,
                    'images': image_count,
                    'size_gb': round(size_gb, 2),
                    'path': str(output_dir)
                })
            else:
                print(f"⚠️  Downloaded but no videos/images found")
                # List what we got
                files = list(output_dir.rglob('*'))[:10]
                if files:
                    print(f"   Found files: {[f.name for f in files if f.is_file()]}")

                results.append({
                    'dataset': ds['name'],
                    'description': ds['description'],
                    'status': 'no_media',
                    'size_gb': round(size_gb, 2)
                })

        elif '403' in result.stdout or 'Forbidden' in result.stdout:
            print(f"❌ 403 Forbidden - needs manual acceptance")
            results.append({
                'dataset': ds['name'],
                'description': ds['description'],
                'status': 'forbidden'
            })
            shutil.rmtree(output_dir, ignore_errors=True)

        elif '404' in result.stdout or 'not found' in result.stdout.lower():
            print(f"❌ 404 Not Found - dataset doesn't exist")
            results.append({
                'dataset': ds['name'],
                'description': ds['description'],
                'status': 'not_found'
            })
            shutil.rmtree(output_dir, ignore_errors=True)

        else:
            error = result.stdout[:200] if result.stdout else result.stderr[:200]
            print(f"❌ Error: {error}")
            results.append({
                'dataset': ds['name'],
                'description': ds['description'],
                'status': 'error',
                'error': error
            })
            shutil.rmtree(output_dir, ignore_errors=True)

    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout (>30 minutes)")
        results.append({
            'dataset': ds['name'],
            'description': ds['description'],
            'status': 'timeout'
        })
        shutil.rmtree(output_dir, ignore_errors=True)

    except Exception as e:
        print(f"❌ Exception: {str(e)[:100]}")
        results.append({
            'dataset': ds['name'],
            'description': ds['description'],
            'status': 'error',
            'error': str(e)[:100]
        })
        shutil.rmtree(output_dir, ignore_errors=True)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print()

successful = [r for r in results if r['status'] == 'success']
forbidden = [r for r in results if r['status'] == 'forbidden']
not_found = [r for r in results if r['status'] == 'not_found']
errors = [r for r in results if r['status'] in ['error', 'timeout', 'no_media']]

total_videos = sum(r.get('videos', 0) for r in successful)
total_images = sum(r.get('images', 0) for r in successful)
total_size = sum(r.get('size_gb', 0) for r in successful)

print(f"✅ Successful: {len(successful)}/{len(datasets)}")
print(f"❌ Forbidden: {len(forbidden)}/{len(datasets)}")
print(f"❌ Not Found: {len(not_found)}/{len(datasets)}")
print(f"⚠️  Errors: {len(errors)}/{len(datasets)}")
print()

print(f"📹 Total Videos: {total_videos:,}")
print(f"🖼️  Total Images: {total_images:,}")
print(f"💾 Total Size: {total_size:.2f} GB")
print()

if successful:
    print("SUCCESSFUL DOWNLOADS:")
    for r in successful:
        print(f"  ✅ {r['description']}")
        print(f"     Dataset: {r['dataset']}")
        print(f"     Videos: {r['videos']:,}, Images: {r['images']:,}")
        print(f"     Size: {r['size_gb']} GB")
        print(f"     Path: {r['path']}")
        print()

if forbidden:
    print("DATASETS REQUIRING MANUAL ACCEPTANCE:")
    for r in forbidden:
        print(f"  ❌ {r['description']}")
        print(f"     Visit: https://www.kaggle.com/datasets/{r['dataset']}")

if not_found:
    print("\nDATASETS NOT FOUND (wrong paths):")
    for r in not_found:
        print(f"  ❌ {r['description']}")
        print(f"     Tried: {r['dataset']}")

# Save results
report_file = output_base / "download_results.json"
with open(report_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f"📄 Full report: {report_file}")
print(f"📁 Downloaded datasets: {output_base}")
print("="*80)
