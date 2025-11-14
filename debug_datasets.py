#!/usr/bin/env python3
"""
Debug Dataset Structure
Check actual directory structure to fix scanning
"""

from pathlib import Path

def check_dataset(dataset_name):
    """Check dataset structure"""

    print(f"\n{'='*80}")
    print(f"Checking: {dataset_name}")
    print('='*80)

    dataset_path = Path(f"/workspace/datasets/tier1/{dataset_name}")

    if not dataset_path.exists():
        print(f"❌ Directory not found: {dataset_path}")
        return

    print(f"✅ Directory exists: {dataset_path}\n")

    # List top-level contents
    print("Top-level contents:")
    items = list(dataset_path.iterdir())
    for item in sorted(items)[:20]:
        if item.is_dir():
            # Count items in subdirectory
            sub_count = len(list(item.iterdir()))
            print(f"  📁 {item.name}/ ({sub_count} items)")
        else:
            print(f"  📄 {item.name}")

    # Count video files
    print(f"\nVideo files:")
    video_exts = ['.mp4', '.avi', '.mkv', '.webm', '.mov']
    for ext in video_exts:
        files = list(dataset_path.rglob(f'*{ext}'))
        if files:
            print(f"  {ext}: {len(files)} files")
            # Show sample paths
            for f in files[:3]:
                print(f"    - {f.relative_to(dataset_path)}")

    # Show directory tree (2 levels)
    print(f"\nDirectory tree (2 levels):")
    for item in sorted(dataset_path.iterdir())[:10]:
        if item.is_dir():
            print(f"  {item.name}/")
            for subitem in sorted(item.iterdir())[:10]:
                if subitem.is_dir():
                    print(f"    {subitem.name}/")
                else:
                    print(f"    {subitem.name}")

# Check all datasets
check_dataset("RWF2000")
check_dataset("UCF_Crime")
check_dataset("SCVD")
check_dataset("RealLife")
