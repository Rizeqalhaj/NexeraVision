#!/usr/bin/env python3
"""
Fix RWF-2000 Dataset - Download and Extract with Filename Renaming
Handles excessively long Unicode filenames by renaming to short format
"""

import subprocess
import os
import zipfile
import re
from pathlib import Path
from datetime import datetime

def safe_filename(original_name, index, ext=".avi"):
    """Generate safe short filename from long original name"""
    # Extract directory structure (e.g., train/Fight/)
    parts = Path(original_name).parts

    # Get category (Fight or NonFight) and split (train or val)
    category = None
    split = None

    for part in parts:
        if part in ['Fight', 'NonFight']:
            category = part.lower()
        if part in ['train', 'val', 'test']:
            split = part

    # Generate short filename
    if category and split:
        safe_name = f"{split}_{category}_{index:04d}{ext}"
    else:
        safe_name = f"video_{index:04d}{ext}"

    return safe_name

def fix_rwf2000_with_rename():
    """Download RWF-2000 and extract with filename renaming"""

    print("=" * 80)
    print("RWF-2000 Dataset Fix - Smart Extraction with Renaming")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup paths
    tier1_dir = Path("/workspace/datasets/tier1")
    rwf2000_dir = tier1_dir / "RWF2000"

    # Remove existing directory if it exists
    if rwf2000_dir.exists():
        print(f"🧹 Removing existing directory: {rwf2000_dir}")
        import shutil
        shutil.rmtree(rwf2000_dir)

    # Create clean directory structure
    rwf2000_dir.mkdir(parents=True, exist_ok=True)
    (rwf2000_dir / "train" / "Fight").mkdir(parents=True, exist_ok=True)
    (rwf2000_dir / "train" / "NonFight").mkdir(parents=True, exist_ok=True)
    (rwf2000_dir / "val" / "Fight").mkdir(parents=True, exist_ok=True)
    (rwf2000_dir / "val" / "NonFight").mkdir(parents=True, exist_ok=True)

    print(f"✅ Clean directory structure created: {rwf2000_dir}\n")

    # Download without unzip
    print("📥 Downloading RWF-2000 from Kaggle...")
    print("   (this may take a few minutes for 11.5GB)\n")

    try:
        # Check if zip already exists
        zip_files = list(tier1_dir.glob("rwf2000*.zip"))

        if not zip_files:
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', 'vulamnguyen/rwf2000', '-p', str(tier1_dir)],
                capture_output=True,
                text=True,
                timeout=1800
            )

            if result.returncode != 0:
                print(f"❌ Download failed: {result.stderr}")
                return False

            print("✅ Download complete!\n")
            zip_files = list(tier1_dir.glob("*.zip"))
        else:
            print("✅ Using existing zip file\n")

        if not zip_files:
            print("❌ No zip file found")
            return False

        zip_file = zip_files[0]
        print(f"📦 Zip file: {zip_file.name}")
        print(f"   Size: {zip_file.stat().st_size / (1024**3):.2f} GB\n")

        # Extract with smart renaming
        print(f"📂 Extracting with filename renaming...")
        print("   (renaming long Unicode filenames to short format)\n")

        counters = {
            'train_fight': 0,
            'train_nonfight': 0,
            'val_fight': 0,
            'val_nonfight': 0
        }

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len([f for f in file_list if f.lower().endswith(('.avi', '.mp4', '.mkv', '.webm', '.mov'))])

            print(f"   Total video files: {total_files:,}")
            print(f"   Processing...\n")

            extracted_count = 0
            skipped_count = 0

            for file_info in zip_ref.filelist:
                original_name = file_info.filename

                # Skip directories
                if original_name.endswith('/'):
                    continue

                # Only process video files
                if not original_name.lower().endswith(('.avi', '.mp4', '.mkv', '.webm', '.mov')):
                    continue

                # Determine category and split
                parts = Path(original_name).parts
                category = None
                split = None

                for part in parts:
                    if part in ['Fight', 'NonFight']:
                        category = part
                    if part in ['train', 'val', 'test']:
                        split = part

                if not (category and split):
                    print(f"   ⚠️  Skipping unknown structure: {original_name}")
                    skipped_count += 1
                    continue

                # Get file extension
                ext = Path(original_name).suffix

                # Generate counter key and increment
                counter_key = f"{split.lower()}_{category.lower()}"
                counters[counter_key] += 1
                index = counters[counter_key]

                # Generate safe filename
                safe_name = f"{split}_{category.lower()}_{index:04d}{ext}"
                target_path = rwf2000_dir / split / category / safe_name

                # Extract with new name
                try:
                    with zip_ref.open(file_info) as source:
                        with open(target_path, 'wb') as target:
                            target.write(source.read())
                    extracted_count += 1

                    if extracted_count % 100 == 0:
                        print(f"   Progress: {extracted_count}/{total_files} videos extracted...")

                except Exception as e:
                    print(f"   ❌ Failed to extract: {original_name[:50]}... - {str(e)[:100]}")

        print(f"\n✅ Extraction complete!")
        print(f"   Extracted: {extracted_count:,} videos")
        if skipped_count:
            print(f"   Skipped: {skipped_count} files\n")

        # Show category breakdown
        print("\n📊 Dataset Breakdown:")
        print(f"   Train/Fight: {counters['train_fight']:,}")
        print(f"   Train/NonFight: {counters['train_nonfight']:,}")
        print(f"   Val/Fight: {counters['val_fight']:,}")
        print(f"   Val/NonFight: {counters['val_nonfight']:,}")
        print(f"   Total: {sum(counters.values()):,}")

        # Calculate final size
        total_size = sum(f.stat().st_size for f in rwf2000_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)

        # Clean up zip file
        print(f"\n🧹 Cleaning up zip file...")
        zip_file.unlink()
        print("✅ Cleanup complete!\n")

        print("=" * 80)
        print("✅ RWF-2000 SUCCESSFULLY EXTRACTED WITH RENAMED FILES!")
        print("=" * 80)
        print(f"📂 Location: {rwf2000_dir}")
        print(f"📹 Videos: {sum(counters.values()):,}")
        print(f"💾 Size: {size_gb:.2f} GB")
        print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Show renamed file examples
        print("\n📝 Filename Examples:")
        for split in ['train', 'val']:
            for category in ['Fight', 'NonFight']:
                sample_dir = rwf2000_dir / split / category
                if sample_dir.exists():
                    samples = list(sample_dir.glob('*'))[:3]
                    if samples:
                        print(f"\n   {split}/{category}/:")
                        for sample in samples:
                            print(f"      - {sample.name}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_rwf2000_with_rename()
    exit(0 if success else 1)
