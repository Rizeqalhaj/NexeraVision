#!/usr/bin/env python3
"""
Download RWF-2000 dataset with filename length fix
Renames videos with excessively long filenames
"""

import os
import shutil
import hashlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_filename(filename, max_length=200):
    """
    Create a safe filename if original is too long.

    Args:
        filename: Original filename
        max_length: Maximum allowed length

    Returns:
        Safe filename with same extension
    """
    if len(filename) <= max_length:
        return filename

    # Keep extension
    name, ext = os.path.splitext(filename)

    # Create hash of original name to preserve uniqueness
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]

    # Truncate name and add hash
    safe_name = name[:max_length-len(ext)-9] + '_' + hash_suffix + ext

    return safe_name


def fix_long_filenames(dataset_path):
    """
    Rename files with excessively long names.

    Args:
        dataset_path: Path to RWF-2000 dataset directory
    """
    dataset_path = Path(dataset_path)

    logger.info("Checking for files with long names...")

    renamed_count = 0

    # Check all video files
    for video_path in dataset_path.rglob("*.avi"):
        filename = video_path.name

        # Check if filename is too long
        if len(filename.encode('utf-8')) > 200:  # Conservative limit
            safe_name = safe_filename(filename)
            new_path = video_path.parent / safe_name

            logger.info(f"Renaming: {filename[:50]}... -> {safe_name}")

            try:
                video_path.rename(new_path)
                renamed_count += 1
            except Exception as e:
                logger.error(f"Failed to rename {filename}: {e}")

    logger.info(f"✅ Renamed {renamed_count} files with long names")

    return renamed_count


def download_rwf2000_safe():
    """
    Download RWF-2000 dataset with filename safety checks.
    """
    import kagglehub

    logger.info("=" * 80)
    logger.info("DOWNLOADING RWF-2000 DATASET (WITH FILENAME FIX)")
    logger.info("=" * 80)

    try:
        # Download to shorter path
        logger.info("Downloading dataset...")

        # Set custom cache dir to avoid long paths
        os.environ['KAGGLE_CACHE_DIR'] = '/tmp/kaggle_cache'

        path = kagglehub.dataset_download("vulamnguyen/rwf2000")
        logger.info(f"Dataset downloaded to: {path}")

        # Find RWF-2000 directory
        dataset_path = Path(path)
        rwf_path = dataset_path / "RWF-2000"

        if not rwf_path.exists():
            # Try parent directories
            for p in [dataset_path, dataset_path.parent]:
                potential_path = p / "RWF-2000"
                if potential_path.exists():
                    rwf_path = potential_path
                    break

        if not rwf_path.exists():
            logger.error(f"RWF-2000 directory not found in {dataset_path}")
            return None

        logger.info(f"Found dataset at: {rwf_path}")

        # Fix long filenames
        logger.info("\n" + "=" * 80)
        logger.info("FIXING LONG FILENAMES")
        logger.info("=" * 80)

        renamed_count = fix_long_filenames(rwf_path)

        # Verify dataset structure
        logger.info("\n" + "=" * 80)
        logger.info("DATASET VERIFICATION")
        logger.info("=" * 80)

        train_fight = len(list((rwf_path / "train" / "Fight").glob("*.avi")))
        train_nonfight = len(list((rwf_path / "train" / "NonFight").glob("*.avi")))
        val_fight = len(list((rwf_path / "val" / "Fight").glob("*.avi")))
        val_nonfight = len(list((rwf_path / "val" / "NonFight").glob("*.avi")))

        total = train_fight + train_nonfight + val_fight + val_nonfight

        logger.info(f"Train Fight videos: {train_fight}")
        logger.info(f"Train NonFight videos: {train_nonfight}")
        logger.info(f"Val Fight videos: {val_fight}")
        logger.info(f"Val NonFight videos: {val_nonfight}")
        logger.info(f"Total videos: {total}")

        if total < 1500:
            logger.warning(f"⚠️ Dataset incomplete! Expected ~2000 videos, got {total}")
        else:
            logger.info("✅ Dataset appears complete!")

        logger.info("=" * 80)

        return str(rwf_path)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

        logger.info("\nTroubleshooting:")
        logger.info("1. Check Kaggle credentials: ~/.kaggle/kaggle.json")
        logger.info("2. Verify disk space: df -h")
        logger.info("3. Try manual download from Kaggle website")

        return None


def main():
    """Main entry point"""

    # Check if dataset already exists
    existing_paths = [
        "/root/.cache/kagglehub/datasets/vulamnguyen/rwf2000/versions/1/RWF-2000",
        "/tmp/kaggle_cache/datasets/vulamnguyen/rwf2000/versions/1/RWF-2000",
        "/workspace/data/RWF-2000"
    ]

    logger.info("Checking for existing dataset...")

    for path in existing_paths:
        if Path(path).exists():
            logger.info(f"Found existing dataset at: {path}")
            logger.info("Fixing long filenames in existing dataset...")

            renamed_count = fix_long_filenames(path)

            logger.info(f"\n✅ Fixed {renamed_count} filenames")
            logger.info(f"Dataset ready at: {path}")

            return path

    logger.info("No existing dataset found. Downloading...")

    # Download with fixes
    dataset_path = download_rwf2000_safe()

    if dataset_path:
        logger.info("\n" + "=" * 80)
        logger.info("✅ SUCCESS!")
        logger.info("=" * 80)
        logger.info(f"Dataset ready at: {dataset_path}")
        logger.info(f"\nUse this path in training:")
        logger.info(f'DATA_DIR = "{dataset_path}"')
    else:
        logger.error("\n❌ Dataset download failed")


if __name__ == "__main__":
    main()
