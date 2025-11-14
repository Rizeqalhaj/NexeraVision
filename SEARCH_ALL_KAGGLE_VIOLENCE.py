#!/usr/bin/env python3
"""
Search and download ALL violence-related datasets on Kaggle
"""

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
    import os
    os.chmod(kaggle_dir / "kaggle.json", 0o600)

print("="*80)
print("SEARCHING ALL KAGGLE VIOLENCE DATASETS")
print("="*80)
print()

# Search terms
search_terms = [
    "violence detection",
    "fight detection",
    "violence",
    "fight",
    "aggressive behavior",
    "anomaly detection video",
    "surveillance violence",
    "cctv violence",
    "real world violence",
    "hockey fight",
    "ucf crime",
    "rwf",
    "violent scenes",
    "fighting videos",
    "assault detection"
]

all_datasets = set()

for term in search_terms:
    print(f"🔍 Searching: '{term}'")
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'list', '-s', term],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) > 0:
                        dataset_name = parts[0]
                        all_datasets.add(dataset_name)

    except Exception as e:
        print(f"  ❌ Error: {e}")

print()
print("="*80)
print(f"FOUND {len(all_datasets)} UNIQUE DATASETS")
print("="*80)
print()

# Save list
datasets_file = Path("/workspace/all_violence_datasets.txt")
with open(datasets_file, 'w') as f:
    for dataset in sorted(all_datasets):
        f.write(f"{dataset}\n")
        print(f"  {dataset}")

print()
print(f"📄 Full list saved to: {datasets_file}")
print()
print("To download all, run:")
print(f"  cat {datasets_file} | while read dataset; do")
print(f"    kaggle datasets download -d $dataset -p /workspace/violence_datasets/$dataset --unzip")
print(f"  done")
print("="*80)
