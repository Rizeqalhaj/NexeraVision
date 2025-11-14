#!/usr/bin/env python3
"""
Download and prepare additional violence detection datasets
Combines multiple datasets to improve model performance
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def download_hockey_fights_dataset():
    """
    Hockey Fights Dataset
    - 1,000 hockey fight videos (500 fight, 500 non-fight)
    - Similar to RWF-2000 but different context
    - Kaggle: https://www.kaggle.com/datasets/yassershrief/hockey-fight-videos
    """
    import kagglehub

    logger.info("=" * 80)
    logger.info("DOWNLOADING HOCKEY FIGHTS DATASET")
    logger.info("=" * 80)

    try:
        path = kagglehub.dataset_download("yassershrief/hockey-fight-videos")
        logger.info(f"✅ Hockey Fights downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"❌ Error downloading Hockey Fights: {e}")
        return None


def download_violent_flows_dataset():
    """
    Violent Flows Dataset
    - Larger dataset with more variety
    - Academic dataset for violence detection research
    """
    logger.info("=" * 80)
    logger.info("VIOLENT FLOWS DATASET")
    logger.info("=" * 80)
    logger.info("Manual download required:")
    logger.info("1. Visit: http://www.openu.ac.il/home/hassner/data/violentflows/")
    logger.info("2. Download the dataset")
    logger.info("3. Extract to: /workspace/data/violent-flows/")
    logger.info("=" * 80)


def download_ucf_crime_dataset():
    """
    UCF Crime Dataset (Anomaly Detection)
    - 1,900 long untrimmed surveillance videos
    - 13 realistic anomalies including fighting, assault, etc.
    - Very large dataset (128 hours of video)
    """
    logger.info("=" * 80)
    logger.info("UCF CRIME DATASET")
    logger.info("=" * 80)
    logger.info("Large dataset - Manual download:")
    logger.info("1. Visit: https://www.crcv.ucf.edu/projects/real-world/")
    logger.info("2. Request access and download")
    logger.info("3. Extract to: /workspace/data/ucf-crime/")
    logger.info("Note: Very large (128 hours of video)")
    logger.info("=" * 80)


def download_movies_fights_dataset():
    """
    Movies Fights Dataset
    - Fight scenes from movies
    - Higher quality video
    - Different context from surveillance
    """
    import kagglehub

    logger.info("=" * 80)
    logger.info("DOWNLOADING MOVIES FIGHTS DATASET")
    logger.info("=" * 80)

    try:
        # Try to find movies fight dataset on Kaggle
        logger.info("Searching for movies fight dataset...")
        logger.info("Manual alternative: Search Kaggle for 'movie fight scenes' or 'action recognition'")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ Error: {e}")


def download_cctv_fights_dataset():
    """
    CCTV Fights Dataset
    - Real-world CCTV footage
    - More realistic than movies
    - Kaggle: https://www.kaggle.com/datasets/naveenk903/cctv-fight-dataset
    """
    import kagglehub

    logger.info("=" * 80)
    logger.info("DOWNLOADING CCTV FIGHTS DATASET")
    logger.info("=" * 80)

    try:
        path = kagglehub.dataset_download("naveenk903/cctv-fight-dataset")
        logger.info(f"✅ CCTV Fights downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"❌ Error downloading CCTV Fights: {e}")
        logger.info("Manual alternative: Visit https://www.kaggle.com/datasets/naveenk903/cctv-fight-dataset")
        return None


def main():
    """Download all available datasets"""

    logger.info("=" * 80)
    logger.info("ADDITIONAL VIOLENCE DETECTION DATASETS")
    logger.info("=" * 80)

    datasets = []

    # Dataset 1: Hockey Fights (Easy to download)
    logger.info("\n1. Hockey Fights Dataset (1,000 videos)")
    hockey_path = download_hockey_fights_dataset()
    if hockey_path:
        datasets.append(("Hockey Fights", hockey_path))

    # Dataset 2: CCTV Fights (Easy to download)
    logger.info("\n2. CCTV Fights Dataset")
    cctv_path = download_cctv_fights_dataset()
    if cctv_path:
        datasets.append(("CCTV Fights", cctv_path))

    # Dataset 3: Violent Flows (Manual)
    logger.info("\n3. Violent Flows Dataset (Manual)")
    download_violent_flows_dataset()

    # Dataset 4: UCF Crime (Manual - Very large)
    logger.info("\n4. UCF Crime Dataset (Manual - 128 hours)")
    download_ucf_crime_dataset()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successfully downloaded: {len(datasets)} datasets")
    for name, path in datasets:
        logger.info(f"  ✅ {name}: {path}")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("1. Organize all datasets into unified structure:")
    logger.info("   /workspace/data/combined/")
    logger.info("   ├── train/")
    logger.info("   │   ├── Fight/")
    logger.info("   │   └── NonFight/")
    logger.info("   └── val/")
    logger.info("       ├── Fight/")
    logger.info("       └── NonFight/")
    logger.info("")
    logger.info("2. Run training with combined dataset:")
    logger.info("   python runpod_train_multi_gpu.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
