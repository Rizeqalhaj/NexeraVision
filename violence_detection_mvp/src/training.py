"""
Training module for Violence Detection MVP.
Handles the complete training pipeline with callbacks and validation.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging
import json
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History

from .config import Config
from .model_architecture import ViolenceDetectionModel, create_callbacks
from .feature_extraction import FeaturePipeline
from .data_preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for violence detection model."""

    def __init__(self, config: Config = Config):
        """Initialize the training pipeline."""
        self.config = config
        self.model_builder = ViolenceDetectionModel(config)
        self.feature_pipeline = FeaturePipeline(config)
        self.data_preprocessor = DataPreprocessor(config)

        self.model: Optional[Model] = None
        self.training_history: Optional[History] = None
        self.training_stats: Dict[str, Any] = {}

    def prepare_data(self, data_dir: Path) -> Tuple[
        List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
    ]:
        """
        Prepare training and validation data.

        Args:
            data_dir: Directory containing video files

        Returns:
            Tuple of (train_data, train_targets, val_data, val_targets)
        """
        logger.info("Preparing data for training...")

        # Get video names and labels
        train_names, test_names, train_labels, test_labels = (
            self.data_preprocessor.preprocess_dataset(data_dir)
        )

        # Extract and cache features
        train_cache_path = self.config.get_cache_path("train")
        test_cache_path = self.config.get_cache_path("test")

        # Extract training features
        if not self.feature_pipeline.cache.cache_exists(train_cache_path):
            logger.info("Extracting training features...")
            self.feature_pipeline.extract_and_cache_features(
                train_names, train_labels, data_dir, train_cache_path
            )

        # Extract test features
        if not self.feature_pipeline.cache.cache_exists(test_cache_path):
            logger.info("Extracting test features...")
            self.feature_pipeline.extract_and_cache_features(
                test_names, test_labels, data_dir, test_cache_path
            )

        # Load processed features
        train_data, train_targets = self.feature_pipeline.load_processed_features(train_cache_path)
        test_data, test_targets = self.feature_pipeline.load_processed_features(test_cache_path)

        logger.info(f"Training data: {len(train_data)} videos")
        logger.info(f"Test data: {len(test_data)} videos")

        return train_data, train_targets, test_data, test_targets

    def create_validation_split(
        self,
        train_data: List[np.ndarray],
        train_targets: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Create validation split from training data.

        Args:
            train_data: Training data
            train_targets: Training targets

        Returns:
            Tuple of (train_data, train_targets, val_data, val_targets)
        """
        split_index = int(len(train_data) * (1 - self.config.VALIDATION_SPLIT))

        actual_train_data = train_data[:split_index]
        actual_train_targets = train_targets[:split_index]
        val_data = train_data[split_index:]
        val_targets = train_targets[split_index:]

        logger.info(f"Training split: {len(actual_train_data)} videos")
        logger.info(f"Validation split: {len(val_data)} videos")

        return actual_train_data, actual_train_targets, val_data, val_targets

    def train_model(
        self,
        train_data: List[np.ndarray],
        train_targets: List[np.ndarray],
        val_data: Optional[List[np.ndarray]] = None,
        val_targets: Optional[List[np.ndarray]] = None,
        model_name: str = "violence_detection_model"
    ) -> History:
        """
        Train the violence detection model.

        Args:
            train_data: Training data
            train_targets: Training targets
            val_data: Validation data (optional)
            val_targets: Validation targets (optional)
            model_name: Name for saving the model

        Returns:
            Training history object
        """
        logger.info("Starting model training...")

        # Create model
        self.model = self.model_builder.create_model()

        # Create callbacks
        callbacks = create_callbacks(self.config)

        # Convert to numpy arrays
        X_train = np.array(train_data)
        y_train = np.array(train_targets)

        validation_data = None
        if val_data is not None and val_targets is not None:
            X_val = np.array(val_data)
            y_val = np.array(val_targets)
            validation_data = (X_val, y_val)

        # Record training start time
        training_start_time = time.time()

        # Train the model
        self.training_history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        # Record training end time
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time

        # Save training statistics
        self.training_stats = {
            'training_duration_seconds': training_duration,
            'training_duration_minutes': training_duration / 60,
            'epochs_completed': len(self.training_history.history['loss']),
            'final_train_loss': float(self.training_history.history['loss'][-1]),
            'final_train_accuracy': float(self.training_history.history['accuracy'][-1]),
            'best_val_loss': float(min(self.training_history.history.get('val_loss', [0]))),
            'best_val_accuracy': float(max(self.training_history.history.get('val_accuracy', [0]))),
            'model_name': model_name,
            'training_timestamp': datetime.now().isoformat()
        }

        # Save the model
        model_path = self.config.get_model_path(model_name)
        self.model.save(str(model_path))
        logger.info(f"Model saved to: {model_path}")

        # Save training history
        history_path = self.config.MODELS_DIR / f"{model_name}_history.json"
        self.save_training_history(history_path)

        logger.info(f"Training completed in {training_duration:.2f} seconds")
        return self.training_history

    def save_training_history(self, filepath: Path) -> None:
        """
        Save training history to JSON file.

        Args:
            filepath: Path to save the history
        """
        if self.training_history is None:
            logger.warning("No training history to save")
            return

        # Convert history to serializable format
        history_dict = {
            'history': {key: [float(val) for val in values]
                       for key, values in self.training_history.history.items()},
            'stats': self.training_stats
        }

        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"Training history saved to: {filepath}")

    def load_training_history(self, filepath: Path) -> Dict[str, Any]:
        """
        Load training history from JSON file.

        Args:
            filepath: Path to the history file

        Returns:
            Dictionary containing training history and stats
        """
        with open(filepath, 'r') as f:
            history_dict = json.load(f)

        logger.info(f"Training history loaded from: {filepath}")
        return history_dict

    def resume_training(
        self,
        model_path: Path,
        train_data: List[np.ndarray],
        train_targets: List[np.ndarray],
        val_data: Optional[List[np.ndarray]] = None,
        val_targets: Optional[List[np.ndarray]] = None,
        additional_epochs: int = 50
    ) -> History:
        """
        Resume training from a saved model.

        Args:
            model_path: Path to the saved model
            train_data: Training data
            train_targets: Training targets
            val_data: Validation data (optional)
            val_targets: Validation targets (optional)
            additional_epochs: Number of additional epochs to train

        Returns:
            Training history object
        """
        logger.info(f"Resuming training from: {model_path}")

        # Load the model
        self.model = self.model_builder.load_model(str(model_path))

        # Update epochs in config
        original_epochs = self.config.EPOCHS
        self.config.EPOCHS = additional_epochs

        # Train for additional epochs
        history = self.train_model(
            train_data, train_targets, val_data, val_targets,
            model_name=f"{model_path.stem}_resumed"
        )

        # Restore original epochs
        self.config.EPOCHS = original_epochs

        return history

    def evaluate_model(
        self,
        test_data: List[np.ndarray],
        test_targets: List[np.ndarray],
        model_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.

        Args:
            test_data: Test data
            test_targets: Test targets
            model_path: Optional path to load model from

        Returns:
            Dictionary containing evaluation metrics
        """
        if model_path is not None:
            self.model = self.model_builder.load_model(str(model_path))

        if self.model is None:
            raise ValueError("No model available for evaluation")

        logger.info("Evaluating model on test data...")

        X_test = np.array(test_data)
        y_test = np.array(test_targets)

        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_samples': len(test_data)
        }

        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")

        return evaluation_results


class ExperimentManager:
    """Manage multiple training experiments."""

    def __init__(self, config: Config = Config):
        """Initialize the experiment manager."""
        self.config = config
        self.experiments: List[Dict[str, Any]] = []

    def run_experiment(
        self,
        experiment_name: str,
        data_dir: Path,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete training experiment.

        Args:
            experiment_name: Name of the experiment
            data_dir: Directory containing video files
            hyperparameters: Optional hyperparameter overrides

        Returns:
            Dictionary containing experiment results
        """
        logger.info(f"Starting experiment: {experiment_name}")

        # Create experiment config
        experiment_config = Config()
        if hyperparameters:
            for key, value in hyperparameters.items():
                if hasattr(experiment_config, key):
                    setattr(experiment_config, key, value)

        # Initialize training pipeline with experiment config
        pipeline = TrainingPipeline(experiment_config)

        # Prepare data
        train_data, train_targets, test_data, test_targets = pipeline.prepare_data(data_dir)

        # Create validation split
        train_data, train_targets, val_data, val_targets = pipeline.create_validation_split(
            train_data, train_targets
        )

        # Train model
        history = pipeline.train_model(
            train_data, train_targets, val_data, val_targets,
            model_name=experiment_name
        )

        # Evaluate model
        evaluation_results = pipeline.evaluate_model(test_data, test_targets)

        # Compile experiment results
        experiment_results = {
            'experiment_name': experiment_name,
            'hyperparameters': hyperparameters or {},
            'training_stats': pipeline.training_stats,
            'evaluation_results': evaluation_results,
            'config': {
                'batch_size': experiment_config.BATCH_SIZE,
                'learning_rate': experiment_config.LEARNING_RATE,
                'epochs': experiment_config.EPOCHS,
                'rnn_size': experiment_config.RNN_SIZE,
                'dropout_rate': experiment_config.DROPOUT_RATE
            }
        }

        # Save experiment results
        self.experiments.append(experiment_results)
        self.save_experiment_results(experiment_name)

        logger.info(f"Experiment {experiment_name} completed")
        return experiment_results

    def save_experiment_results(self, experiment_name: str) -> None:
        """
        Save experiment results to JSON file.

        Args:
            experiment_name: Name of the experiment
        """
        results_path = self.config.MODELS_DIR / f"{experiment_name}_results.json"

        experiment_result = next(
            (exp for exp in self.experiments if exp['experiment_name'] == experiment_name),
            None
        )

        if experiment_result:
            with open(results_path, 'w') as f:
                json.dump(experiment_result, f, indent=2)

            logger.info(f"Experiment results saved to: {results_path}")

    def compare_experiments(self) -> Dict[str, Any]:
        """
        Compare results from multiple experiments.

        Returns:
            Dictionary containing comparison results
        """
        if not self.experiments:
            return {"message": "No experiments to compare"}

        comparison = {
            'num_experiments': len(self.experiments),
            'experiments': []
        }

        for exp in self.experiments:
            exp_summary = {
                'name': exp['experiment_name'],
                'test_accuracy': exp['evaluation_results']['test_accuracy'],
                'test_loss': exp['evaluation_results']['test_loss'],
                'training_time_minutes': exp['training_stats']['training_duration_minutes'],
                'epochs_completed': exp['training_stats']['epochs_completed'],
                'hyperparameters': exp['hyperparameters']
            }
            comparison['experiments'].append(exp_summary)

        # Find best experiment
        best_exp = max(comparison['experiments'], key=lambda x: x['test_accuracy'])
        comparison['best_experiment'] = best_exp

        return comparison


def validate_training_setup(config: Config, data_dir: Path) -> Dict[str, Any]:
    """
    Validate that the training setup is correct.

    Args:
        config: Configuration object
        data_dir: Data directory

    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'config_valid': False,
        'data_dir_exists': False,
        'gpu_available': False,
        'tensorflow_version': tf.__version__,
        'errors': []
    }

    try:
        # Check config
        config.validate_config()
        validation_results['config_valid'] = True
    except Exception as e:
        validation_results['errors'].append(f"Config validation failed: {str(e)}")

    # Check data directory
    if data_dir.exists():
        validation_results['data_dir_exists'] = True
        validation_results['data_dir_files'] = len(list(data_dir.iterdir()))
    else:
        validation_results['errors'].append(f"Data directory not found: {data_dir}")

    # Check GPU availability
    validation_results['gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
    validation_results['gpu_devices'] = [device.name for device in tf.config.list_physical_devices('GPU')]

    # Check memory
    try:
        import psutil
        validation_results['system_memory_gb'] = psutil.virtual_memory().total / (1024**3)
        validation_results['available_memory_gb'] = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        validation_results['errors'].append("psutil not available for memory check")

    return validation_results