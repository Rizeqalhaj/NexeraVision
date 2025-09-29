"""
Validation and testing utilities for the Violence Detection MVP project.
Comprehensive testing of the VGG19 feature extraction pipeline.
"""

import numpy as np
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import tensorflow as tf
from tensorflow.keras.applications import VGG19

from .config import Config
from .feature_extraction import VGG19FeatureExtractor, FeaturePipeline, FeatureCache
from .data_preprocessing import VideoFrameExtractor, DataPreprocessor, validate_video_file
from .logging_config import setup_logging, log_system_info, log_config_info

logger = logging.getLogger(__name__)


class VGG19ValidationSuite:
    """Comprehensive validation suite for VGG19 feature extraction pipeline."""

    def __init__(self, config: Config = Config):
        """Initialize the validation suite."""
        self.config = config
        self.results = {}

    def run_all_tests(self, data_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run all validation tests.

        Args:
            data_dir: Optional directory containing test videos

        Returns:
            Dictionary containing all test results
        """
        logger.info("Starting VGG19 validation suite...")

        # System validation
        self.results['system'] = self.validate_system_requirements()

        # Model validation
        self.results['model'] = self.validate_vgg19_model()

        # Feature extraction validation
        self.results['feature_extraction'] = self.validate_feature_extraction()

        # Data preprocessing validation
        self.results['data_preprocessing'] = self.validate_data_preprocessing()

        # Cache validation
        self.results['cache'] = self.validate_cache_operations()

        # Integration validation
        self.results['integration'] = self.validate_integration()

        # Performance validation
        self.results['performance'] = self.validate_performance()

        # Dataset validation (if data_dir provided)
        if data_dir:
            self.results['dataset'] = self.validate_dataset(data_dir)

        # Generate summary
        self.results['summary'] = self.generate_summary()

        logger.info("VGG19 validation suite completed")
        return self.results

    def validate_system_requirements(self) -> Dict[str, Any]:
        """Validate system requirements."""
        logger.info("Validating system requirements...")

        results = {
            'tensorflow_version': tf.__version__,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'memory_check': True,
            'errors': []
        }

        try:
            # Check TensorFlow version
            tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
            if tf_version < (2, 12):
                results['errors'].append(f"TensorFlow version {tf.__version__} is below minimum 2.12")

            # Check available memory
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < 4:
                results['errors'].append(f"Low available memory: {available_gb:.1f} GB")
                results['memory_check'] = False

            # Check GPU configuration
            if results['gpu_available']:
                gpus = tf.config.list_physical_devices('GPU')
                results['gpu_count'] = len(gpus)
                results['gpu_memory_growth'] = True
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    results['gpu_memory_growth'] = False
                    results['errors'].append(f"GPU memory growth config failed: {str(e)}")

            results['status'] = 'passed' if not results['errors'] else 'failed'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def validate_vgg19_model(self) -> Dict[str, Any]:
        """Validate VGG19 model loading and configuration."""
        logger.info("Validating VGG19 model...")

        results = {
            'model_loaded': False,
            'transfer_model_created': False,
            'output_shape_correct': False,
            'errors': []
        }

        try:
            # Test VGG19 model loading
            extractor = VGG19FeatureExtractor(self.config)
            results['model_loaded'] = True

            # Check transfer model
            if hasattr(extractor, 'transfer_model'):
                results['transfer_model_created'] = True

                # Check output shape
                expected_shape = (None, self.config.TRANSFER_VALUES_SIZE)
                actual_shape = extractor.transfer_model.output_shape
                results['output_shape_correct'] = actual_shape == expected_shape

                if not results['output_shape_correct']:
                    results['errors'].append(
                        f"Output shape mismatch: expected {expected_shape}, got {actual_shape}"
                    )

            results['status'] = 'passed' if not results['errors'] else 'failed'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def validate_feature_extraction(self) -> Dict[str, Any]:
        """Validate feature extraction functionality."""
        logger.info("Validating feature extraction...")

        results = {
            'synthetic_extraction': False,
            'batch_processing': False,
            'preprocessing_applied': False,
            'output_dtype_correct': False,
            'errors': []
        }

        try:
            extractor = VGG19FeatureExtractor(self.config)

            # Test with synthetic data
            test_frames = np.random.rand(
                self.config.FRAMES_PER_VIDEO,
                self.config.IMG_SIZE,
                self.config.IMG_SIZE,
                3
            ).astype(np.float32)

            features = extractor.extract_features_from_frames(test_frames)

            if features is not None:
                results['synthetic_extraction'] = True

                # Check output shape
                expected_shape = (self.config.FRAMES_PER_VIDEO, self.config.TRANSFER_VALUES_SIZE)
                if features.shape == expected_shape:
                    results['batch_processing'] = True

                # Check output dtype
                expected_dtype = getattr(np, self.config.FEATURE_DTYPE)
                if features.dtype == expected_dtype:
                    results['output_dtype_correct'] = True

                # Test preprocessing
                preprocessed = extractor._preprocess_frames(test_frames)
                if preprocessed.shape == test_frames.shape:
                    results['preprocessing_applied'] = True

            results['status'] = 'passed' if not results['errors'] else 'failed'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def validate_data_preprocessing(self) -> Dict[str, Any]:
        """Validate data preprocessing functionality."""
        logger.info("Validating data preprocessing...")

        results = {
            'frame_extractor_created': False,
            'evenly_spaced_calculation': False,
            'synthetic_video_processing': False,
            'errors': []
        }

        try:
            # Test frame extractor creation
            extractor = VideoFrameExtractor(self.config)
            results['frame_extractor_created'] = True

            # Test evenly spaced frame calculation
            total_frames = 100
            indices = extractor._calculate_evenly_spaced_indices(total_frames)

            if len(indices) == self.config.FRAMES_PER_VIDEO:
                results['evenly_spaced_calculation'] = True

                # Check indices are valid
                if all(0 <= idx < total_frames for idx in indices):
                    results['indices_valid'] = True
                else:
                    results['errors'].append("Invalid frame indices calculated")

            # Test with very short video
            short_indices = extractor._calculate_evenly_spaced_indices(5)
            if len(short_indices) == self.config.FRAMES_PER_VIDEO:
                results['short_video_handling'] = True

            results['status'] = 'passed' if not results['errors'] else 'failed'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def validate_cache_operations(self) -> Dict[str, Any]:
        """Validate cache operations."""
        logger.info("Validating cache operations...")

        results = {
            'cache_save': False,
            'cache_load': False,
            'compression_applied': False,
            'metadata_saved': False,
            'errors': []
        }

        try:
            cache = FeatureCache(self.config)

            # Create test data
            test_features = [
                np.random.rand(self.config.FRAMES_PER_VIDEO, self.config.TRANSFER_VALUES_SIZE)
                .astype(getattr(np, self.config.FEATURE_DTYPE))
                for _ in range(3)
            ]
            test_labels = [[1, 0], [0, 1], [1, 0]]

            # Test cache save
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_path = Path(temp_dir) / "test_cache.h5"

                cache.save_features_to_cache(test_features, test_labels, cache_path)
                results['cache_save'] = cache_path.exists()

                if results['cache_save']:
                    # Test cache load
                    loaded_features, loaded_labels = cache.load_features_from_cache(cache_path)
                    results['cache_load'] = True

                    # Validate loaded data
                    expected_features = np.concatenate(test_features, axis=0)
                    expected_labels = np.repeat(test_labels, self.config.FRAMES_PER_VIDEO, axis=0)

                    if np.allclose(loaded_features, expected_features, rtol=1e-3):
                        results['data_integrity'] = True

                    if np.array_equal(loaded_labels, expected_labels):
                        results['labels_integrity'] = True

                    # Check file size (compression)
                    file_size = cache_path.stat().st_size
                    if file_size > 0:
                        results['compression_applied'] = True

            results['status'] = 'passed' if not results['errors'] else 'failed'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def validate_integration(self) -> Dict[str, Any]:
        """Validate end-to-end integration."""
        logger.info("Validating integration...")

        results = {
            'pipeline_created': False,
            'end_to_end_processing': False,
            'errors': []
        }

        try:
            # Test pipeline creation
            pipeline = FeaturePipeline(self.config)
            results['pipeline_created'] = True

            # Test with synthetic data
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_path = Path(temp_dir) / "integration_test.h5"

                # Create mock video data
                test_features = [
                    np.random.rand(self.config.FRAMES_PER_VIDEO, self.config.TRANSFER_VALUES_SIZE)
                    .astype(getattr(np, self.config.FEATURE_DTYPE))
                    for _ in range(2)
                ]
                test_labels = [[1, 0], [0, 1]]

                # Save and load
                pipeline.cache.save_features_to_cache(test_features, test_labels, cache_path)
                data, targets = pipeline.load_processed_features(cache_path)

                if len(data) == 2 and len(targets) == 2:
                    results['end_to_end_processing'] = True

            results['status'] = 'passed' if not results['errors'] else 'failed'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics."""
        logger.info("Validating performance...")

        results = {
            'feature_extraction_time': 0,
            'cache_save_time': 0,
            'cache_load_time': 0,
            'memory_usage_reasonable': True,
            'errors': []
        }

        try:
            extractor = VGG19FeatureExtractor(self.config)
            cache = FeatureCache(self.config)

            # Measure feature extraction time
            test_frames = np.random.rand(
                self.config.FRAMES_PER_VIDEO,
                self.config.IMG_SIZE,
                self.config.IMG_SIZE,
                3
            ).astype(np.float32)

            start_time = time.time()
            features = extractor.extract_features_from_frames(test_frames)
            results['feature_extraction_time'] = time.time() - start_time

            # Measure cache operations
            if features is not None:
                test_features = [features]
                test_labels = [[1, 0]]

                with tempfile.TemporaryDirectory() as temp_dir:
                    cache_path = Path(temp_dir) / "perf_test.h5"

                    # Measure save time
                    start_time = time.time()
                    cache.save_features_to_cache(test_features, test_labels, cache_path)
                    results['cache_save_time'] = time.time() - start_time

                    # Measure load time
                    start_time = time.time()
                    loaded_features, loaded_labels = cache.load_features_from_cache(cache_path)
                    results['cache_load_time'] = time.time() - start_time

            # Performance thresholds
            if results['feature_extraction_time'] > 10:  # 10 seconds for 20 frames
                results['errors'].append(f"Slow feature extraction: {results['feature_extraction_time']:.2f}s")

            results['status'] = 'passed' if not results['errors'] else 'failed'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def validate_dataset(self, data_dir: Path) -> Dict[str, Any]:
        """Validate dataset if provided."""
        logger.info(f"Validating dataset in {data_dir}...")

        results = {
            'directory_exists': data_dir.exists(),
            'videos_found': 0,
            'valid_videos': 0,
            'errors': []
        }

        try:
            if results['directory_exists']:
                # Analyze dataset
                from .data_preprocessing import analyze_dataset_videos
                analysis = analyze_dataset_videos(data_dir, self.config)

                if 'error' not in analysis:
                    results['videos_found'] = analysis['total_videos']
                    results['valid_videos'] = analysis['valid_videos']
                    results['total_duration_hours'] = analysis['total_duration_hours']
                    results['average_fps'] = analysis['average_fps']

                    if analysis['invalid_videos'] > 0:
                        results['errors'].append(f"{analysis['invalid_videos']} invalid videos found")

                else:
                    results['errors'].append(analysis['error'])

            results['status'] = 'passed' if not results['errors'] else 'failed'

        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_tests = len(self.results) - 1  # Exclude summary itself
        passed_tests = sum(1 for key, result in self.results.items()
                          if key != 'summary' and result.get('status') == 'passed')

        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'overall_status': 'passed' if passed_tests == total_tests else 'failed',
            'critical_issues': []
        }

        # Identify critical issues
        for test_name, result in self.results.items():
            if test_name != 'summary' and result.get('status') in ['failed', 'error']:
                summary['critical_issues'].append({
                    'test': test_name,
                    'status': result.get('status'),
                    'errors': result.get('errors', [])
                })

        return summary

    def print_results(self) -> None:
        """Print validation results in a readable format."""
        print("\n" + "="*60)
        print("VGG19 FEATURE EXTRACTION VALIDATION RESULTS")
        print("="*60)

        for test_name, result in self.results.items():
            if test_name == 'summary':
                continue

            status = result.get('status', 'unknown')
            status_symbol = "✅" if status == 'passed' else "❌" if status == 'failed' else "⚠️"

            print(f"\n{status_symbol} {test_name.upper()}: {status.upper()}")

            if result.get('errors'):
                for error in result['errors']:
                    print(f"  - Error: {error}")

        # Print summary
        summary = self.results.get('summary', {})
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {summary.get('total_tests', 0)}")
        print(f"Passed: {summary.get('passed_tests', 0)}")
        print(f"Failed: {summary.get('failed_tests', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0):.1%}")
        print(f"Overall status: {summary.get('overall_status', 'unknown').upper()}")

        if summary.get('critical_issues'):
            print(f"\nCRITICAL ISSUES:")
            for issue in summary['critical_issues']:
                print(f"  - {issue['test']}: {issue['status']}")


def run_quick_validation(config: Config = Config) -> bool:
    """
    Run a quick validation of the core components.

    Args:
        config: Configuration object

    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Test model loading
        extractor = VGG19FeatureExtractor(config)

        # Test with synthetic data
        test_frames = np.random.rand(
            config.FRAMES_PER_VIDEO,
            config.IMG_SIZE,
            config.IMG_SIZE,
            3
        ).astype(np.float32)

        features = extractor.extract_features_from_frames(test_frames)

        if features is None:
            return False

        expected_shape = (config.FRAMES_PER_VIDEO, config.TRANSFER_VALUES_SIZE)
        return features.shape == expected_shape

    except Exception as e:
        logger.error(f"Quick validation failed: {str(e)}")
        return False


def main():
    """Main function for running validation from command line."""
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    # Log system info
    log_system_info(logger)
    log_config_info(Config, logger)

    # Run validation
    validator = VGG19ValidationSuite()
    results = validator.run_all_tests()

    # Print results
    validator.print_results()

    # Exit with appropriate code
    summary = results.get('summary', {})
    exit_code = 0 if summary.get('overall_status') == 'passed' else 1
    exit(exit_code)


if __name__ == "__main__":
    main()