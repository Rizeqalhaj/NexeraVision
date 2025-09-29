"""
Violence Detection MVP - A comprehensive video violence detection system.

This package provides a complete pipeline for detecting violence in videos using:
- VGG19 for feature extraction from video frames
- LSTM with Attention mechanism for sequence classification
- Transfer learning approach for efficient training

Main modules:
- config: Configuration management
- data_preprocessing: Video frame extraction and labeling
- feature_extraction: VGG19-based feature extraction
- model_architecture: LSTM-Attention model definition
- training: Training pipeline and experiment management
- evaluation: Model evaluation and metrics
- inference: Real-time prediction and API
- visualization: Plotting and visualization utilities
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "NEXARA Team"
__description__ = "Violence Detection MVP using Deep Learning"

# Import main classes for easy access
from .config import Config
from .data_preprocessing import DataPreprocessor, VideoFrameExtractor
from .feature_extraction import FeaturePipeline, VGG19FeatureExtractor
from .model_architecture import ViolenceDetectionModel, AttentionLayer
from .training import TrainingPipeline, ExperimentManager
from .evaluation import ModelEvaluator, ModelComparator
from .inference import ViolencePredictor, InferenceAPI, RealTimeVideoProcessor
from .visualization import TrainingVisualizer, EvaluationVisualizer, DataVisualizer
from .utils import FileManager, DataSaver, SystemInfo

__all__ = [
    # Core classes
    'Config',
    'DataPreprocessor',
    'VideoFrameExtractor',
    'FeaturePipeline',
    'VGG19FeatureExtractor',
    'ViolenceDetectionModel',
    'AttentionLayer',
    'TrainingPipeline',
    'ExperimentManager',
    'ModelEvaluator',
    'ModelComparator',
    'ViolencePredictor',
    'InferenceAPI',
    'RealTimeVideoProcessor',

    # Visualization
    'TrainingVisualizer',
    'EvaluationVisualizer',
    'DataVisualizer',

    # Utilities
    'FileManager',
    'DataSaver',
    'SystemInfo',
]