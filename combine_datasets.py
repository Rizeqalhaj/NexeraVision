#!/usr/bin/env python3
"""
Combine multiple violence detection datasets
Creates unified train/val structure
"""

import os
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def combine_datasets(datasets_config, output_dir="/workspace/data/combined"):
    """
    Combine multiple datasets into unified structure.

    Args:
        datasets_config: List of dataset configurations
        output_dir: Output directory for combined dataset
    """
    logger.info("=" * 80)
    logger.info("COMBINING MULTIPLE DATASETS")
    logger.info("=" * 80)

    output_path = Path(output_dir)

    # Create output structure
    train_fight = output_path / "train" / "Fight"
    train_nonfight = output_path / "train" / "NonFight"
    val_fight = output_path / "val" / "Fight"
    val_nonfight = output_path / "val" / "NonFight"

    for dir_path in [train_fight, train_nonfight, val_fight, val_nonfight]:
        dir_path.mkdir(parents=True, exist_ok=True)

    total_copied = 0

    # Process each dataset
    for dataset_name, dataset_path, split_ratio in datasets_config:
        logger.info(f"\nProcessing: {dataset_name}")
        logger.info(f"  Path: {dataset_path}")

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            logger.warning(f"  ⚠️ Dataset not found: {dataset_path}")
            continue

        # Find fight and non-fight videos
        fight_videos = list(dataset_path.rglob("*Fight*/*.avi")) + \
                      list(dataset_path.rglob("*fight*/*.avi")) + \
                      list(dataset_path.rglob("*violence*/*.avi"))

        nonfight_videos = list(dataset_path.rglob("*NonFight*/*.avi")) + \
                         list(dataset_path.rglob("*nonfight*/*.avi")) + \
                         list(dataset_path.rglob("*normal*/*.avi"))

        logger.info(f"  Found: {len(fight_videos)} fight, {len(nonfight_videos)} non-fight")

        # Split into train/val
        train_split = int(len(fight_videos) * split_ratio)

        # Copy fight videos
        for i, video in enumerate(tqdm(fight_videos, desc=f"  Copying {dataset_name} fights")):
            dest_dir = train_fight if i < train_split else val_fight
            dest_file = dest_dir / f"{dataset_name}_{video.name}"
            shutil.copy2(video, dest_file)
            total_copied += 1

        # Copy non-fight videos
        train_split = int(len(nonfight_videos) * split_ratio)
        for i, video in enumerate(tqdm(nonfight_videos, desc=f"  Copying {dataset_name} non-fights")):
            dest_dir = train_nonfight if i < train_split else val_nonfight
            dest_file = dest_dir / f"{dataset_name}_{video.name}"
            shutil.copy2(video, dest_file)
            total_copied += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("COMBINED DATASET SUMMARY")
    logger.info("=" * 80)

    train_fight_count = len(list(train_fight.glob("*.avi")))
    train_nonfight_count = len(list(train_nonfight.glob("*.avi")))
    val_fight_count = len(list(val_fight.glob("*.avi")))
    val_nonfight_count = len(list(val_nonfight.glob("*.avi")))

    logger.info(f"Train Fight: {train_fight_count}")
    logger.info(f"Train NonFight: {train_nonfight_count}")
    logger.info(f"Val Fight: {val_fight_count}")
    logger.info(f"Val NonFight: {val_nonfight_count}")
    logger.info(f"Total videos: {train_fight_count + train_nonfight_count + val_fight_count + val_nonfight_count}")
    logger.info("=" * 80)

    return str(output_path)


def main():
    """Main entry point"""

    # Configure datasets to combine
    # Format: (name, path, train_split_ratio)
    datasets = [
        ("RWF2000", "/workspace/data/RWF-2000/RWF-2000", 0.8),
        ("Hockey", "/root/.cache/kagglehub/datasets/yassershrief/hockey-fight-videos/versions/1", 0.8),
        ("CCTV", "/root/.cache/kagglehub/datasets/naveenk903/cctv-fight-dataset/versions/1", 0.8),
    ]

    # Combine datasets
    combined_path = combine_datasets(datasets, "/workspace/data/combined")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info(f"1. Update training script DATA_DIR to: {combined_path}")
    logger.info("2. Run training with larger dataset:")
    logger.info("   python runpod_train_multi_gpu.py")
    logger.info("3. Expected improvement: 80-85% accuracy with more data")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
