#!/usr/bin/env python3
"""
Find all downloaded violence detection datasets and show their structure
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def find_video_files(path, max_depth=5):
    """Find all video files recursively with depth limit."""
    path = Path(path)
    videos = []

    try:
        for root, dirs, files in os.walk(path):
            # Calculate current depth
            depth = len(Path(root).relative_to(path).parts)
            if depth > max_depth:
                continue

            for file in files:
                if file.endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    videos.append(Path(root) / file)
    except Exception as e:
        logger.error(f"Error scanning {path}: {e}")

    return videos


def analyze_dataset_structure(base_path):
    """Analyze dataset structure and classify videos."""
    base_path = Path(base_path)

    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYZING: {base_path}")
    logger.info(f"{'='*80}")

    if not base_path.exists():
        logger.warning(f"Path does not exist")
        return None

    # Find all videos
    videos = find_video_files(base_path)

    if not videos:
        logger.warning(f"No videos found")
        return None

    logger.info(f"Total videos found: {len(videos)}")

    # Try to classify
    fight_videos = []
    nonfight_videos = []
    unknown_videos = []

    for video in videos:
        video_str = str(video).lower()
        parent_str = str(video.parent).lower()

        # Check for fight indicators
        if any(keyword in video_str for keyword in ['fight', 'violence', 'violent', 'vi', 'fi']):
            fight_videos.append(video)
        elif any(keyword in parent_str for keyword in ['fight', 'violence', 'violent']):
            fight_videos.append(video)
        # Check for non-fight indicators
        elif any(keyword in video_str for keyword in ['nonfight', 'nonviolence', 'normal', 'nofi', 'nv']):
            nonfight_videos.append(video)
        elif any(keyword in parent_str for keyword in ['nonfight', 'nonviolence', 'normal']):
            nonfight_videos.append(video)
        else:
            unknown_videos.append(video)

    logger.info(f"  Fight videos: {len(fight_videos)}")
    logger.info(f"  Non-fight videos: {len(nonfight_videos)}")
    logger.info(f"  Unknown/unclassified: {len(unknown_videos)}")

    # Show directory structure
    logger.info(f"\nDirectory structure:")
    dirs = set()
    for video in videos[:20]:  # Show first 20
        dirs.add(str(video.parent.relative_to(base_path)))

    for d in sorted(dirs):
        logger.info(f"  {d}/")

    if len(videos) > 20:
        logger.info(f"  ... (showing first 20 directories)")

    # Show sample filenames
    logger.info(f"\nSample filenames:")
    for video in videos[:5]:
        logger.info(f"  {video.name}")

    return {
        'base_path': base_path,
        'total': len(videos),
        'fight': fight_videos,
        'nonfight': nonfight_videos,
        'unknown': unknown_videos
    }


def main():
    """Find and analyze all datasets."""

    logger.info("="*80)
    logger.info("SEARCHING FOR ALL VIOLENCE DETECTION DATASETS")
    logger.info("="*80)

    # Search locations
    search_paths = [
        "/workspace/data",
        "/root/.cache/kagglehub/datasets",
        "/tmp/kaggle_cache",
    ]

    found_datasets = []

    for search_path in search_paths:
        search_path = Path(search_path)
        if not search_path.exists():
            continue

        logger.info(f"\nSearching in: {search_path}")

        # Find all subdirectories that might contain datasets
        try:
            for item in search_path.rglob("*"):
                if item.is_dir():
                    # Check if directory name suggests a dataset
                    dir_name = item.name.lower()
                    if any(keyword in dir_name for keyword in [
                        'rwf', 'hockey', 'fight', 'violence', 'cctv',
                        'dataset', 'violence', 'real-life', 'ucf'
                    ]):
                        # Check if it has videos
                        videos = list(item.glob("*.avi")) + list(item.glob("*.mp4"))
                        if videos or any((item / subdir).exists() for subdir in ['train', 'val', 'Fight', 'NonFight']):
                            result = analyze_dataset_structure(item)
                            if result and result['total'] > 0:
                                found_datasets.append(result)
        except Exception as e:
            logger.error(f"Error searching {search_path}: {e}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("DATASET SUMMARY")
    logger.info("="*80)

    if not found_datasets:
        logger.warning("No datasets found!")
        logger.info("\nTry downloading datasets first:")
        logger.info("  python download_hockey.py")
        return

    total_videos = 0
    total_fight = 0
    total_nonfight = 0

    for i, dataset in enumerate(found_datasets, 1):
        logger.info(f"\n{i}. {dataset['base_path']}")
        logger.info(f"   Total: {dataset['total']} videos")
        logger.info(f"   Fight: {len(dataset['fight'])}")
        logger.info(f"   Non-fight: {len(dataset['nonfight'])}")
        logger.info(f"   Unknown: {len(dataset['unknown'])}")

        total_videos += dataset['total']
        total_fight += len(dataset['fight'])
        total_nonfight += len(dataset['nonfight'])

    logger.info("\n" + "="*80)
    logger.info(f"GRAND TOTAL: {total_videos} videos")
    logger.info(f"  Fight: {total_fight}")
    logger.info(f"  Non-fight: {total_nonfight}")
    logger.info(f"  Unknown: {total_videos - total_fight - total_nonfight}")
    logger.info("="*80)

    # Save results for combination script
    import json
    output_file = "/workspace/found_datasets.json"
    with open(output_file, 'w') as f:
        json.dump({
            'datasets': [{
                'path': str(d['base_path']),
                'total': d['total'],
                'fight_count': len(d['fight']),
                'nonfight_count': len(d['nonfight']),
            } for d in found_datasets]
        }, f, indent=2)

    logger.info(f"\n✅ Dataset info saved to: {output_file}")
    logger.info("\nNext step: Run combine_all_datasets.py to merge everything")


if __name__ == "__main__":
    main()
