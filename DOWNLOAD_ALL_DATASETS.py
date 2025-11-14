#!/usr/bin/env python3
"""
Download ALL Violence Detection Datasets
For Vast.ai - Uses /workspace/kaggle.json
"""

import os
import subprocess
import json
from pathlib import Path

# Setup Kaggle
kaggle_json = Path("/workspace/kaggle.json")
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

if kaggle_json.exists():
    import shutil
    shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    print("✅ Kaggle configured")
else:
    print("⚠️  No kaggle.json found - Kaggle datasets will be skipped")

# Output directory
output_base = Path("/workspace/violence_datasets")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING ALL VIOLENCE DETECTION DATASETS")
print("="*80)
print()

datasets = []

# ============================================================================
# KAGGLE DATASETS
# ============================================================================

kaggle_datasets = [
    # Violence Detection Datasets
    ("mohamedabdallah/real-life-violence-situations-dataset", "RLVS - Real Life Violence Situations"),
    ("naveenk903/rlvs-real-life-violence-situations-dataset", "RLVS Alternative Source"),
    ("mateohervas/dcsass-dataset", "DCSASS - Detecting Carrying Suspicious Actions"),
    ("yassershrief/hockey-fight-detection-dataset", "Hockey Fight Detection"),
    ("pelealg/ucf-crime-dataset", "UCF Crime Dataset"),
    ("mateohervas/surveillance-fighting-dataset", "Surveillance Fighting Dataset"),
    ("sujaykapadnis/fight-detection", "Fight Detection Dataset"),
    ("nishantrahate/fight-dataset", "Fight Dataset"),
    ("sayakpaul/rwf-2000", "RWF-2000 Real World Fights"),
    ("toluwaniaremu/violence-detection-videos", "Violence Detection Videos"),
    ("seifmahmoud9/fighting-videos", "Fighting Videos Dataset"),
    ("puneetmalhotra/violence-detection-dataset", "Violence Detection Dataset"),
    ("nikhilbhange/video-violence-detection", "Video Violence Detection"),

    # Anomaly/Surveillance Datasets
    ("ihelon/ucf-crime", "UCF Crime Alternative"),
    ("mission-ai/fight-detection-surv-dataset", "Fight Detection Surveillance"),
    ("ashrafemad/violent-scenes-dataset", "Violent Scenes Dataset"),
    ("gregorywinter/violence-detection-in-videos", "Violence Detection Videos"),

    # Additional Action Recognition (may contain violence)
    ("pevogam/ucf101", "UCF-101 Action Recognition"),
    ("matthewjansen/ucf101-action-recognition", "UCF-101 Alternative"),
]

# ============================================================================
# DIRECT DOWNLOAD SOURCES (academic datasets)
# ============================================================================

direct_downloads = [
    {
        "name": "RWF-2000 (Official)",
        "url": "https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection/archive/refs/heads/master.zip",
        "extract": True
    },
    {
        "name": "Hockey Fight Dataset (Official)",
        "url": "http://visilab.etsii.uclm.es/personas/oscar/FightDetection/videos.zip",
        "extract": True
    },
    {
        "name": "Violent Flows Dataset",
        "url": "https://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.tar.gz",
        "extract": True
    },
]

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_kaggle_dataset(dataset_name, description):
    """Download from Kaggle"""
    print(f"\n{'='*80}")
    print(f"📥 {description}")
    print(f"   Kaggle: {dataset_name}")
    print(f"{'='*80}")

    output_dir = output_base / dataset_name.replace('/', '_')
    output_dir.mkdir(exist_ok=True)

    try:
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(output_dir), '--unzip']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            # Count videos
            video_count = len(list(output_dir.rglob('*.mp4'))) + len(list(output_dir.rglob('*.avi')))
            print(f"✅ Downloaded: {video_count} videos")
            return {'name': description, 'status': 'success', 'videos': video_count, 'path': str(output_dir)}
        else:
            error = result.stderr[:200] if result.stderr else 'Unknown error'
            print(f"❌ Failed: {error}")
            return {'name': description, 'status': 'failed', 'error': error}
    except Exception as e:
        print(f"❌ Error: {str(e)[:200]}")
        return {'name': description, 'status': 'error', 'error': str(e)[:200]}

def download_direct(info):
    """Download from direct URL"""
    print(f"\n{'='*80}")
    print(f"📥 {info['name']}")
    print(f"   URL: {info['url']}")
    print(f"{'='*80}")

    output_dir = output_base / info['name'].replace(' ', '_').replace('(', '').replace(')', '')
    output_dir.mkdir(exist_ok=True)

    try:
        filename = info['url'].split('/')[-1]
        output_file = output_dir / filename

        # Download with wget
        cmd = ['wget', '-c', info['url'], '-O', str(output_file), '--timeout=30', '--tries=3']
        subprocess.run(cmd, timeout=1800)

        # Extract if needed
        if info.get('extract') and output_file.exists():
            if filename.endswith('.zip'):
                subprocess.run(['unzip', '-q', str(output_file), '-d', str(output_dir)])
                output_file.unlink()
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                subprocess.run(['tar', '-xzf', str(output_file), '-C', str(output_dir)])
                output_file.unlink()

        # Count videos
        video_count = len(list(output_dir.rglob('*.mp4'))) + len(list(output_dir.rglob('*.avi')))
        print(f"✅ Downloaded: {video_count} videos")
        return {'name': info['name'], 'status': 'success', 'videos': video_count, 'path': str(output_dir)}

    except Exception as e:
        print(f"❌ Error: {str(e)[:200]}")
        return {'name': info['name'], 'status': 'error', 'error': str(e)[:200]}

# ============================================================================
# MAIN DOWNLOAD PROCESS
# ============================================================================

results = []

print("\n" + "="*80)
print("PHASE 1: KAGGLE DATASETS")
print("="*80)

if kaggle_json.exists():
    for dataset_name, description in kaggle_datasets:
        result = download_kaggle_dataset(dataset_name, description)
        results.append(result)
else:
    print("⚠️  Skipping Kaggle datasets (no kaggle.json)")

print("\n" + "="*80)
print("PHASE 2: DIRECT DOWNLOADS")
print("="*80)

for dataset_info in direct_downloads:
    result = download_direct(dataset_info)
    results.append(result)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)
print()

success_count = sum(1 for r in results if r['status'] == 'success')
failed_count = sum(1 for r in results if r['status'] in ['failed', 'error'])
total_videos = sum(r.get('videos', 0) for r in results if r['status'] == 'success')

print(f"✅ Successful: {success_count}")
print(f"❌ Failed: {failed_count}")
print(f"📹 Total videos: {total_videos}")
print()

print("Successful downloads:")
for r in results:
    if r['status'] == 'success':
        print(f"  ✅ {r['name']}: {r.get('videos', 0)} videos")
        print(f"     Path: {r.get('path', 'unknown')}")

if failed_count > 0:
    print()
    print("Failed downloads:")
    for r in results:
        if r['status'] in ['failed', 'error']:
            print(f"  ❌ {r['name']}: {r.get('error', 'unknown error')}")

# Save results
results_file = output_base / "download_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f"📄 Results saved to: {results_file}")
print(f"📁 All datasets in: {output_base}")
print("="*80)
