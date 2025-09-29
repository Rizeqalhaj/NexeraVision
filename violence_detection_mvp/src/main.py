#!/usr/bin/env python3
"""
Main entry point for Violence Detection MVP.
Provides comprehensive CLI interface for all operations.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Local imports
from .config import Config
from .training import TrainingPipeline, ExperimentManager
from .evaluation import ModelEvaluator
from .inference import ViolencePredictor, RealTimeVideoProcessor, InferenceAPI
from .visualization import TrainingVisualizer, EvaluationVisualizer
from .utils import SystemInfo, ConfigManager, Logger, validate_project_setup
from .data_preprocessing import VideoFrameExtractor
from .feature_extraction import VGG19FeatureExtractor

# Set up logging
logger = logging.getLogger(__name__)


class ViolenceDetectionCLI:
    """Main CLI interface for Violence Detection MVP."""

    def __init__(self):
        """Initialize CLI interface."""
        self.config = Config()
        self.setup_logging()

    def setup_logging(self) -> None:
        """Setup enhanced logging for CLI."""
        log_file = Path("violence_detection.log")
        self.logger = Logger.setup_logging(
            log_file=log_file,
            level="INFO"
        )

    def run_training(self, args: argparse.Namespace) -> bool:
        """
        Run training pipeline.

        Args:
            args: Command line arguments

        Returns:
            True if successful, False otherwise
        """
        try:
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                return False

            logger.info("Starting training pipeline...")

            # Initialize training pipeline
            trainer = TrainingPipeline(self.config)

            # Prepare data
            logger.info("Preparing training data...")
            train_data, train_targets, test_data, test_targets = trainer.prepare_data(data_dir)

            if train_data is None:
                logger.error("Failed to prepare training data")
                return False

            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")

            # Train model
            logger.info("Starting model training...")
            history = trainer.train_model(
                train_data, train_targets,
                test_data, test_targets,
                save_best=True
            )

            if history is None:
                logger.error("Training failed")
                return False

            # Save training history
            history_path = Path("training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history.history, f, indent=2, default=str)

            logger.info(f"Training completed successfully!")
            logger.info(f"Model saved to: {self.config.MODELS_DIR}")
            logger.info(f"Training history saved to: {history_path}")

            # Generate training visualizations if requested
            if args.visualize:
                logger.info("Generating training visualizations...")
                viz = TrainingVisualizer()
                viz.plot_training_history(
                    history.history,
                    save_path="training_curves.png"
                )
                logger.info("Training visualizations saved to training_curves.png")

            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

    def run_evaluation(self, args: argparse.Namespace) -> bool:
        """
        Run model evaluation.

        Args:
            args: Command line arguments

        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(args.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            logger.info("Starting model evaluation...")

            # Initialize evaluator
            evaluator = ModelEvaluator()
            evaluator.load_model(model_path)

            # Load test data
            data_dir = Path(args.data_dir)
            trainer = TrainingPipeline(self.config)
            _, _, test_data, test_targets = trainer.prepare_data(data_dir)

            if test_data is None:
                logger.error("Failed to load test data")
                return False

            # Run comprehensive evaluation
            logger.info("Running comprehensive evaluation...")
            results = evaluator.evaluate_model_comprehensive(test_data, test_targets)

            # Print results
            metrics = results['metrics']
            logger.info("Evaluation Results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision_macro']:.4f}")
            logger.info(f"  Recall: {metrics['recall_macro']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1_macro']:.4f}")

            # Save detailed results
            results_path = Path("evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Detailed results saved to: {results_path}")

            # Generate evaluation visualizations if requested
            if args.visualize:
                logger.info("Generating evaluation visualizations...")
                viz = EvaluationVisualizer()

                # Confusion matrix
                viz.plot_confusion_matrix(
                    results['confusion_matrix'],
                    save_path="confusion_matrix.png"
                )

                # ROC curves
                if 'roc_data' in results:
                    viz.plot_roc_curves(
                        results['roc_data'],
                        save_path="roc_curves.png"
                    )

                logger.info("Evaluation visualizations saved")

            return True

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return False

    def run_inference(self, args: argparse.Namespace) -> bool:
        """
        Run inference on video files.

        Args:
            args: Command line arguments

        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(args.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            logger.info("Initializing predictor...")
            predictor = ViolencePredictor(model_path, self.config)

            # Set confidence threshold if provided
            if hasattr(args, 'confidence') and args.confidence:
                predictor.set_confidence_threshold(args.confidence)
                logger.info(f"Confidence threshold set to: {args.confidence}")

            results = []

            if args.batch_mode:
                # Batch processing
                video_paths = []

                if args.video_dir:
                    # Process all videos in directory
                    video_dir = Path(args.video_dir)
                    if not video_dir.exists():
                        logger.error(f"Video directory not found: {video_dir}")
                        return False

                    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
                    for ext in video_extensions:
                        video_paths.extend(video_dir.glob(f"*{ext}"))

                elif args.video_list:
                    # Process videos from list file
                    list_file = Path(args.video_list)
                    with open(list_file, 'r') as f:
                        video_paths = [Path(line.strip()) for line in f if line.strip()]

                else:
                    logger.error("Batch mode requires either --video-dir or --video-list")
                    return False

                if not video_paths:
                    logger.error("No video files found for processing")
                    return False

                logger.info(f"Processing {len(video_paths)} videos in batch mode...")
                results = predictor.predict_video_batch(video_paths, return_probabilities=True)

            else:
                # Single video processing
                if not args.video_path:
                    logger.error("Single mode requires --video-path")
                    return False

                video_path = Path(args.video_path)
                if not video_path.exists():
                    logger.error(f"Video file not found: {video_path}")
                    return False

                logger.info(f"Processing single video: {video_path}")
                result = predictor.predict_video(video_path, return_probabilities=True)
                results = [result]

            # Process and display results
            successful_predictions = 0
            for i, result in enumerate(results):
                if "error" not in result:
                    successful_predictions += 1
                    violence_detected = result['violence_detected']
                    confidence = result['confidence']
                    video_path = result['video_path']

                    status = "VIOLENCE DETECTED" if violence_detected else "NO VIOLENCE"
                    logger.info(f"Video {i+1}: {video_path} -> {status} (confidence: {confidence:.3f})")

                    if args.verbose:
                        probs = result.get('probabilities', {})
                        logger.info(f"  Probabilities: Violence={probs.get('Violence', 0):.3f}, "
                                  f"No Violence={probs.get('No Violence', 0):.3f}")
                        logger.info(f"  Processing time: {result.get('prediction_time_seconds', 0):.3f}s")
                else:
                    logger.error(f"Video {i+1}: {result.get('video_path', 'Unknown')} -> ERROR: {result['error']}")

            # Save results if requested
            if args.output_file:
                output_path = Path(args.output_file)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to: {output_path}")

            logger.info(f"Inference completed: {successful_predictions}/{len(results)} videos processed successfully")
            return True

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return False

    def run_realtime(self, args: argparse.Namespace) -> bool:
        """
        Run real-time video processing.

        Args:
            args: Command line arguments

        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(args.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            logger.info("Initializing real-time processor...")
            processor = RealTimeVideoProcessor(model_path, self.config)

            # Determine video source
            if args.camera:
                video_source = args.camera  # Camera index
                logger.info(f"Using camera {video_source}")
            elif args.video_file:
                video_source = args.video_file
                logger.info(f"Processing video file: {video_source}")
            else:
                video_source = 0  # Default camera
                logger.info("Using default camera (index 0)")

            # Setup output callback if needed
            output_callback = None
            if args.output_file:
                results_log = []

                def callback(result):
                    results_log.append({
                        "timestamp": time.time(),
                        "violence_detected": result.get('violence_detected', False),
                        "confidence": result.get('confidence', 0),
                        "predicted_class": result.get('predicted_class', 'Unknown')
                    })

                output_callback = callback

            logger.info("Starting real-time processing (Press 'q' to quit)...")

            # Run real-time processing
            processor.process_video_stream(
                video_source=video_source,
                output_callback=output_callback,
                display=args.display
            )

            # Save results if callback was used
            if output_callback and args.output_file:
                output_path = Path(args.output_file)
                with open(output_path, 'w') as f:
                    json.dump(results_log, f, indent=2, default=str)
                logger.info(f"Real-time results saved to: {output_path}")

            logger.info("Real-time processing completed")
            return True

        except Exception as e:
            logger.error(f"Real-time processing failed: {str(e)}")
            return False

    def run_demo(self, args: argparse.Namespace) -> bool:
        """
        Run end-to-end demo.

        Args:
            args: Command line arguments

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting Violence Detection MVP Demo...")

            # Check system requirements
            logger.info("Checking system requirements...")
            deps = SystemInfo.check_dependencies()
            if not deps['all_available']:
                logger.warning("Some dependencies are missing:")
                for dep, available in deps['dependencies'].items():
                    if not available:
                        logger.warning(f"  - {dep}: {deps['versions'][dep]}")

            # Validate project setup
            project_root = Path().absolute()
            validation = validate_project_setup(project_root)

            if not validation['setup_complete']:
                logger.error("Project setup is incomplete")
                return False

            # Check if model exists
            model_path = Path(args.model_path) if args.model_path else Path("models/violence_detection_model.h5")

            if not model_path.exists():
                logger.info("Trained model not found. Running training first...")

                # Create mock args for training
                train_args = argparse.Namespace(
                    data_dir=args.data_dir or "data/raw",
                    visualize=True
                )

                if not self.run_training(train_args):
                    logger.error("Training failed during demo")
                    return False

            # Run evaluation
            logger.info("Running model evaluation...")
            eval_args = argparse.Namespace(
                model_path=str(model_path),
                data_dir=args.data_dir or "data/raw",
                visualize=True
            )

            if not self.run_evaluation(eval_args):
                logger.warning("Evaluation failed, continuing demo...")

            # Run sample inference
            logger.info("Running sample inference...")
            data_dir = Path(args.data_dir or "data/raw")

            # Find sample videos
            video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
            sample_videos = []
            for ext in video_extensions:
                sample_videos.extend(list(data_dir.glob(f"*{ext}"))[:3])  # Max 3 samples

            if sample_videos:
                inference_args = argparse.Namespace(
                    model_path=str(model_path),
                    batch_mode=True,
                    video_dir=str(data_dir),
                    confidence=0.5,
                    verbose=True,
                    output_file="demo_results.json"
                )

                self.run_inference(inference_args)
            else:
                logger.warning("No sample videos found for inference demo")

            logger.info("Demo completed successfully!")
            logger.info("Generated files:")
            logger.info("  - training_curves.png (training visualizations)")
            logger.info("  - confusion_matrix.png (evaluation results)")
            logger.info("  - demo_results.json (inference results)")
            logger.info("  - violence_detection.log (detailed logs)")

            return True

        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            return False

    def show_info(self, args: argparse.Namespace) -> bool:
        """
        Show system and project information.

        Args:
            args: Command line arguments

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Violence Detection MVP - System Information")
            logger.info("=" * 50)

            # System info
            sys_info = SystemInfo.get_system_info()
            logger.info("System Information:")
            logger.info(f"  Platform: {sys_info.get('platform', 'Unknown')}")
            logger.info(f"  Python Version: {sys_info.get('python_version', 'Unknown')}")
            logger.info(f"  CPU Count: {sys_info.get('cpu_count', 'Unknown')}")
            logger.info(f"  Memory Total: {sys_info.get('memory_total_gb', 0):.1f} GB")
            logger.info(f"  Memory Available: {sys_info.get('memory_available_gb', 0):.1f} GB")

            # Dependencies
            deps = SystemInfo.check_dependencies()
            logger.info("\nDependency Status:")
            for dep, available in deps['dependencies'].items():
                status = "✓" if available else "✗"
                version = deps['versions'][dep]
                logger.info(f"  {status} {dep}: {version}")

            # Project structure validation
            project_root = Path().absolute()
            validation = validate_project_setup(project_root)

            logger.info("\nProject Structure:")
            for dir_path, exists in validation['required_directories'].items():
                status = "✓" if exists else "✗"
                logger.info(f"  {status} {dir_path}/")

            logger.info("\nRequired Files:")
            for file_path, exists in validation['required_files'].items():
                status = "✓" if exists else "✗"
                logger.info(f"  {status} {file_path}")

            # Model information if available
            model_path = Path("models/violence_detection_model.h5")
            if model_path.exists():
                logger.info(f"\nModel Information:")
                logger.info(f"  Model Path: {model_path}")
                logger.info(f"  Model Size: {model_path.stat().st_size / (1024*1024):.1f} MB")

                try:
                    api = InferenceAPI(model_path)
                    model_info = api.get_model_info()
                    logger.info(f"  Model Type: {model_info['model_type']}")
                    logger.info(f"  Input Shape: {model_info['input_shape']}")
                    logger.info(f"  Classes: {model_info['class_names']}")
                except Exception as e:
                    logger.warning(f"  Could not load model info: {str(e)}")
            else:
                logger.info(f"\nModel Status: No trained model found")

            # Configuration
            logger.info(f"\nConfiguration:")
            logger.info(f"  Image Size: {self.config.IMG_SIZE}x{self.config.IMG_SIZE}")
            logger.info(f"  Frames per Video: {self.config.FRAMES_PER_VIDEO}")
            logger.info(f"  RNN Size: {self.config.RNN_SIZE}")
            logger.info(f"  Batch Size: {self.config.BATCH_SIZE}")
            logger.info(f"  Learning Rate: {self.config.LEARNING_RATE}")

            return True

        except Exception as e:
            logger.error(f"Failed to show info: {str(e)}")
            return False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI interface."""
    parser = argparse.ArgumentParser(
        description="Violence Detection MVP - Comprehensive CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python -m src.main train --data-dir data/raw --visualize

  # Evaluation
  python -m src.main evaluate --model-path models/model.h5 --data-dir data/raw --visualize

  # Single video inference
  python -m src.main infer --model-path models/model.h5 --video-path video.avi --confidence 0.7

  # Batch inference
  python -m src.main infer --model-path models/model.h5 --batch-mode --video-dir data/test --output-file results.json

  # Real-time processing
  python -m src.main realtime --model-path models/model.h5 --camera 0 --display

  # Complete demo
  python -m src.main demo --data-dir data/raw

  # System information
  python -m src.main info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train violence detection model')
    train_parser.add_argument('--data-dir', type=str, default='data/raw',
                             help='Directory containing training videos')
    train_parser.add_argument('--visualize', action='store_true',
                             help='Generate training visualizations')

    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model file')
    eval_parser.add_argument('--data-dir', type=str, default='data/raw',
                            help='Directory containing test videos')
    eval_parser.add_argument('--visualize', action='store_true',
                            help='Generate evaluation visualizations')

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on videos')
    infer_parser.add_argument('--model-path', type=str, required=True,
                             help='Path to trained model file')
    infer_parser.add_argument('--video-path', type=str,
                             help='Path to single video file')
    infer_parser.add_argument('--batch-mode', action='store_true',
                             help='Process multiple videos')
    infer_parser.add_argument('--video-dir', type=str,
                             help='Directory containing videos (batch mode)')
    infer_parser.add_argument('--video-list', type=str,
                             help='Text file with video paths (batch mode)')
    infer_parser.add_argument('--confidence', type=float, default=0.5,
                             help='Confidence threshold for predictions')
    infer_parser.add_argument('--output-file', type=str,
                             help='Save results to JSON file')
    infer_parser.add_argument('--verbose', action='store_true',
                             help='Show detailed prediction information')

    # Real-time command
    realtime_parser = subparsers.add_parser('realtime', help='Real-time video processing')
    realtime_parser.add_argument('--model-path', type=str, required=True,
                                help='Path to trained model file')
    realtime_parser.add_argument('--camera', type=int,
                                help='Camera index (default: 0)')
    realtime_parser.add_argument('--video-file', type=str,
                                help='Process video file instead of camera')
    realtime_parser.add_argument('--display', action='store_true',
                                help='Display video with predictions')
    realtime_parser.add_argument('--output-file', type=str,
                                help='Save real-time results to JSON file')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run complete end-to-end demo')
    demo_parser.add_argument('--data-dir', type=str, default='data/raw',
                            help='Directory containing demo videos')
    demo_parser.add_argument('--model-path', type=str,
                            help='Path to existing model (optional)')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system and project information')

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize CLI
    cli = ViolenceDetectionCLI()

    # Route to appropriate command
    success = False

    try:
        if args.command == 'train':
            success = cli.run_training(args)
        elif args.command == 'evaluate':
            success = cli.run_evaluation(args)
        elif args.command == 'infer':
            success = cli.run_inference(args)
        elif args.command == 'realtime':
            success = cli.run_realtime(args)
        elif args.command == 'demo':
            success = cli.run_demo(args)
        elif args.command == 'info':
            success = cli.show_info(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())