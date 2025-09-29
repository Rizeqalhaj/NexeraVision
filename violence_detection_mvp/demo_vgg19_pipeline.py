#!/usr/bin/env python3
"""
Demo script for VGG19 Feature Extraction Pipeline
Violence Detection MVP Project

This script demonstrates the complete VGG19 feature extraction pipeline
including data preprocessing, feature extraction, caching, and validation.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.feature_extraction import (
    VGG19FeatureExtractor,
    FeaturePipeline,
    print_vgg19_info,
    validate_feature_extraction
)
from src.data_preprocessing import (
    VideoFrameExtractor,
    DataPreprocessor,
    analyze_dataset_videos
)
from src.validation_utils import VGG19ValidationSuite, run_quick_validation
from src.logging_config import setup_logging, log_system_info, log_config_info


def demo_model_info():
    """Demonstrate VGG19 model information."""
    print("\n" + "="*60)
    print("VGG19 MODEL INFORMATION")
    print("="*60)

    print_vgg19_info()


def demo_feature_extraction(video_path: Path = None):
    """Demonstrate feature extraction from a single video."""
    print("\n" + "="*60)
    print("VGG19 FEATURE EXTRACTION DEMO")
    print("="*60)

    config = Config()
    logger = logging.getLogger(__name__)

    if video_path and video_path.exists():
        logger.info(f"Testing feature extraction with video: {video_path}")

        # Validate video
        from src.data_preprocessing import validate_video_file
        validation = validate_video_file(video_path)

        if validation['valid']:
            logger.info(f"Video validation passed: {validation}")

            # Extract features
            result = validate_feature_extraction(video_path, config)

            if result.get('success'):
                logger.info("Feature extraction successful!")
                logger.info(f"Frames shape: {result['frames_shape']}")
                logger.info(f"Features shape: {result['features_shape']}")
                logger.info(f"Features mean: {result['features_mean']:.4f}")
                logger.info(f"Features std: {result['features_std']:.4f}")
            else:
                logger.error(f"Feature extraction failed: {result.get('error', 'Unknown error')}")
        else:
            logger.error(f"Video validation failed: {validation['error']}")
    else:
        logger.info("Testing feature extraction with synthetic data...")

        # Test with synthetic frames
        import numpy as np
        extractor = VGG19FeatureExtractor(config)

        test_frames = np.random.rand(
            config.FRAMES_PER_VIDEO,
            config.IMG_SIZE,
            config.IMG_SIZE,
            3
        ).astype(np.float32)

        logger.info(f"Created synthetic frames: {test_frames.shape}")

        features = extractor.extract_features_from_frames(test_frames)

        if features is not None:
            logger.info("Synthetic feature extraction successful!")
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Features dtype: {features.dtype}")
            logger.info(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
        else:
            logger.error("Synthetic feature extraction failed!")


def demo_data_preprocessing():
    """Demonstrate data preprocessing capabilities."""
    print("\n" + "="*60)
    print("DATA PREPROCESSING DEMO")
    print("="*60)

    config = Config()
    logger = logging.getLogger(__name__)

    # Test frame extractor
    extractor = VideoFrameExtractor(config)

    # Test evenly spaced frame calculation
    test_cases = [10, 50, 100, 200, 500]

    for total_frames in test_cases:
        indices = extractor._calculate_evenly_spaced_indices(total_frames)
        logger.info(f"Total frames: {total_frames}, Selected indices: {indices[:5]}...{indices[-5:]}")

    # Test with very short video
    short_indices = extractor._calculate_evenly_spaced_indices(5)
    logger.info(f"Short video (5 frames) indices: {short_indices}")


def demo_cache_operations():
    """Demonstrate cache operations."""
    print("\n" + "="*60)
    print("CACHE OPERATIONS DEMO")
    print("="*60)

    config = Config()
    logger = logging.getLogger(__name__)

    from src.feature_extraction import FeatureCache
    import numpy as np
    import tempfile

    cache = FeatureCache(config)

    # Create test data
    logger.info("Creating test data...")
    test_features = [
        np.random.rand(config.FRAMES_PER_VIDEO, config.TRANSFER_VALUES_SIZE)
        .astype(getattr(np, config.FEATURE_DTYPE))
        for _ in range(5)
    ]
    test_labels = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]

    # Test cache save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "demo_cache.h5"

        logger.info(f"Saving to cache: {cache_path}")
        cache.save_features_to_cache(test_features, test_labels, cache_path)

        logger.info("Loading from cache...")
        loaded_features, loaded_labels = cache.load_features_from_cache(cache_path)

        logger.info(f"Loaded features shape: {loaded_features.shape}")
        logger.info(f"Loaded labels shape: {loaded_labels.shape}")

        # Check data integrity
        original_features = np.concatenate(test_features, axis=0)
        if np.allclose(loaded_features, original_features, rtol=1e-3):
            logger.info("✅ Data integrity check passed!")
        else:
            logger.error("❌ Data integrity check failed!")


def demo_pipeline_integration(data_dir: Path = None):
    """Demonstrate complete pipeline integration."""
    print("\n" + "="*60)
    print("PIPELINE INTEGRATION DEMO")
    print("="*60)

    config = Config()
    logger = logging.getLogger(__name__)

    pipeline = FeaturePipeline(config)

    if data_dir and data_dir.exists():
        logger.info(f"Testing pipeline with dataset: {data_dir}")

        # Analyze dataset
        analysis = analyze_dataset_videos(data_dir, config)

        if 'error' not in analysis:
            logger.info(f"Dataset analysis:")
            for key, value in analysis.items():
                if key != 'warnings':
                    logger.info(f"  {key}: {value}")

            if analysis['warnings']:
                logger.warning(f"Found {len(analysis['warnings'])} warnings")
                for warning in analysis['warnings'][:3]:
                    logger.warning(f"  - {warning}")
        else:
            logger.error(f"Dataset analysis failed: {analysis['error']}")
    else:
        logger.info("Testing pipeline with synthetic data...")

        # Create mock data
        from src.feature_extraction import FeatureCache
        import numpy as np
        import tempfile

        test_features = [
            np.random.rand(config.FRAMES_PER_VIDEO, config.TRANSFER_VALUES_SIZE)
            .astype(getattr(np, config.FEATURE_DTYPE))
            for _ in range(3)
        ]
        test_labels = [[1, 0], [0, 1], [1, 0]]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "pipeline_test.h5"

            # Save test data
            pipeline.cache.save_features_to_cache(test_features, test_labels, cache_path)

            # Load and process
            data, targets = pipeline.load_processed_features(cache_path)

            logger.info(f"Pipeline processing results:")
            logger.info(f"  Videos processed: {len(data)}")
            logger.info(f"  Video shape: {data[0].shape if data else 'N/A'}")
            logger.info(f"  Target shape: {targets[0].shape if targets else 'N/A'}")

            # Get statistics
            stats = pipeline.get_feature_statistics(cache_path)
            logger.info(f"  Feature statistics: {stats}")


def demo_validation_suite(data_dir: Path = None):
    """Demonstrate validation suite."""
    print("\n" + "="*60)
    print("VALIDATION SUITE DEMO")
    print("="*60)

    # Run quick validation first
    logger = logging.getLogger(__name__)
    logger.info("Running quick validation...")

    if run_quick_validation():
        logger.info("✅ Quick validation passed!")
    else:
        logger.error("❌ Quick validation failed!")
        return

    # Run full validation suite
    logger.info("Running comprehensive validation suite...")
    validator = VGG19ValidationSuite()
    results = validator.run_all_tests(data_dir)

    # Print results
    validator.print_results()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="VGG19 Feature Extraction Pipeline Demo")
    parser.add_argument("--data-dir", type=Path, help="Directory containing video files for testing")
    parser.add_argument("--video", type=Path, help="Single video file for testing")
    parser.add_argument("--mode", choices=[
        "all", "model-info", "feature-extraction", "preprocessing",
        "cache", "pipeline", "validation"
    ], default="all", help="Demo mode to run")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting VGG19 Feature Extraction Pipeline Demo")

    # Log system information
    log_system_info(logger)
    log_config_info(Config, logger)

    try:
        if args.mode in ["all", "model-info"]:
            demo_model_info()

        if args.mode in ["all", "feature-extraction"]:
            demo_feature_extraction(args.video)

        if args.mode in ["all", "preprocessing"]:
            demo_data_preprocessing()

        if args.mode in ["all", "cache"]:
            demo_cache_operations()

        if args.mode in ["all", "pipeline"]:
            demo_pipeline_integration(args.data_dir)

        if args.mode in ["all", "validation"]:
            demo_validation_suite(args.data_dir)

        logger.info("Demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()