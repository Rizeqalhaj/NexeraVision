"""
Evaluation module for Violence Detection MVP.
Handles model evaluation, metrics calculation, and performance analysis.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import json

import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc, multilabel_confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config
from .model_architecture import ViolenceDetectionModel
from .feature_extraction import FeaturePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and metrics calculation."""

    def __init__(self, config: Config = Config):
        """Initialize the model evaluator."""
        self.config = config
        self.model: Optional[Model] = None
        self.predictions: Optional[np.ndarray] = None
        self.true_labels: Optional[np.ndarray] = None
        self.class_names = ['Violence', 'No Violence']

    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model for evaluation.

        Args:
            model_path: Path to the saved model
        """
        model_builder = ViolenceDetectionModel(self.config)
        self.model = model_builder.load_model(str(model_path))
        logger.info(f"Model loaded from: {model_path}")

    def predict(
        self,
        test_data: List[np.ndarray],
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test data.

        Args:
            test_data: Test data
            threshold: Classification threshold

        Returns:
            Tuple of (raw_predictions, binary_predictions)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        X_test = np.array(test_data)

        # Get raw predictions
        raw_predictions = self.model.predict(X_test, verbose=0)

        # Convert to binary predictions
        binary_predictions = (raw_predictions > threshold).astype(int)

        self.predictions = binary_predictions
        logger.info(f"Generated predictions for {len(test_data)} samples")

        return raw_predictions, binary_predictions

    def calculate_metrics(
        self,
        true_labels: List[np.ndarray],
        predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            true_labels: True labels
            predictions: Predictions (if None, uses stored predictions)

        Returns:
            Dictionary containing all metrics
        """
        if predictions is None:
            predictions = self.predictions

        if predictions is None:
            raise ValueError("No predictions available. Call predict() first.")

        y_true = np.array(true_labels)
        y_pred = predictions

        # Store for other methods
        self.true_labels = y_true

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics (using average='weighted' for imbalanced datasets)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class detailed metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_per_class': [float(p) for p in precision_per_class],
            'recall_per_class': [float(r) for r in recall_per_class],
            'f1_per_class': [float(f) for f in f1_per_class],
            'class_names': self.class_names,
            'total_samples': len(y_true)
        }

        logger.info(f"Calculated metrics - Accuracy: {accuracy:.4f}, F1 (macro): {f1_macro:.4f}")
        return metrics

    def generate_confusion_matrix(
        self,
        true_labels: Optional[List[np.ndarray]] = None,
        predictions: Optional[np.ndarray] = None,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate confusion matrix.

        Args:
            true_labels: True labels (if None, uses stored labels)
            predictions: Predictions (if None, uses stored predictions)
            normalize: Normalization mode ('true', 'pred', 'all', or None)

        Returns:
            Confusion matrix
        """
        if true_labels is None:
            true_labels = self.true_labels
        if predictions is None:
            predictions = self.predictions

        if true_labels is None or predictions is None:
            raise ValueError("Labels and predictions not available")

        y_true = np.array(true_labels)
        y_pred = predictions

        # Convert to single label format for sklearn
        y_true_single = np.argmax(y_true, axis=1)
        y_pred_single = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true_single, y_pred_single, normalize=normalize)

        logger.info("Generated confusion matrix")
        return cm

    def generate_multilabel_confusion_matrix(
        self,
        true_labels: Optional[List[np.ndarray]] = None,
        predictions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate multilabel confusion matrix as in the original implementation.

        Args:
            true_labels: True labels (if None, uses stored labels)
            predictions: Predictions (if None, uses stored predictions)

        Returns:
            Multilabel confusion matrix
        """
        if true_labels is None:
            true_labels = self.true_labels
        if predictions is None:
            predictions = self.predictions

        if true_labels is None or predictions is None:
            raise ValueError("Labels and predictions not available")

        y_true = np.array(true_labels)
        y_pred = predictions

        mcm = multilabel_confusion_matrix(y_true, y_pred)

        logger.info("Generated multilabel confusion matrix")
        return mcm

    def generate_classification_report(
        self,
        true_labels: Optional[List[np.ndarray]] = None,
        predictions: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            true_labels: True labels (if None, uses stored labels)
            predictions: Predictions (if None, uses stored predictions)

        Returns:
            Classification report as string
        """
        if true_labels is None:
            true_labels = self.true_labels
        if predictions is None:
            predictions = self.predictions

        if true_labels is None or predictions is None:
            raise ValueError("Labels and predictions not available")

        y_true = np.array(true_labels)
        y_pred = predictions

        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=False
        )

        logger.info("Generated classification report")
        return report

    def calculate_roc_auc(
        self,
        true_labels: List[np.ndarray],
        raw_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate ROC-AUC metrics for each class.

        Args:
            true_labels: True labels
            raw_predictions: Raw prediction probabilities

        Returns:
            Dictionary containing ROC-AUC metrics
        """
        y_true = np.array(true_labels)

        roc_auc_scores = {}
        fpr_tpr = {}

        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], raw_predictions[:, i])
            roc_auc = auc(fpr, tpr)

            roc_auc_scores[class_name] = float(roc_auc)
            fpr_tpr[class_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

        # Calculate macro average
        roc_auc_scores['macro_avg'] = float(np.mean(list(roc_auc_scores.values())))

        logger.info(f"Calculated ROC-AUC scores: {roc_auc_scores}")
        return {'scores': roc_auc_scores, 'curves': fpr_tpr}

    def evaluate_model_comprehensive(
        self,
        test_data: List[np.ndarray],
        test_labels: List[np.ndarray],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.

        Args:
            test_data: Test data
            test_labels: Test labels
            threshold: Classification threshold

        Returns:
            Dictionary containing all evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Starting comprehensive model evaluation...")

        # Generate predictions
        raw_predictions, binary_predictions = self.predict(test_data, threshold)

        # Calculate metrics
        metrics = self.calculate_metrics(test_labels, binary_predictions)

        # Generate confusion matrices
        cm = self.generate_confusion_matrix(test_labels, binary_predictions)
        mcm = self.generate_multilabel_confusion_matrix(test_labels, binary_predictions)

        # Generate classification report
        classification_report_str = self.generate_classification_report(test_labels, binary_predictions)

        # Calculate ROC-AUC
        roc_auc_results = self.calculate_roc_auc(test_labels, raw_predictions)

        # Compile results
        evaluation_results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'multilabel_confusion_matrix': mcm.tolist(),
            'classification_report': classification_report_str,
            'roc_auc': roc_auc_results,
            'threshold': threshold,
            'evaluation_timestamp': np.datetime64('now').isoformat()
        }

        logger.info("Comprehensive evaluation completed")
        return evaluation_results

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Evaluation results saved to: {output_path}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


class ModelComparator:
    """Compare multiple models' performance."""

    def __init__(self, config: Config = Config):
        """Initialize the model comparator."""
        self.config = config
        self.model_results: List[Dict[str, Any]] = []

    def add_model_results(
        self,
        model_name: str,
        evaluation_results: Dict[str, Any]
    ) -> None:
        """
        Add evaluation results for a model.

        Args:
            model_name: Name of the model
            evaluation_results: Evaluation results dictionary
        """
        model_result = {
            'model_name': model_name,
            'results': evaluation_results
        }
        self.model_results.append(model_result)
        logger.info(f"Added results for model: {model_name}")

    def compare_models(self) -> Dict[str, Any]:
        """
        Compare all added models.

        Returns:
            Dictionary containing comparison results
        """
        if len(self.model_results) < 2:
            return {"error": "Need at least 2 models for comparison"}

        comparison = {
            'num_models': len(self.model_results),
            'models': []
        }

        for model_result in self.model_results:
            model_name = model_result['model_name']
            metrics = model_result['results']['metrics']

            model_summary = {
                'name': model_name,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'f1_per_class': metrics['f1_per_class']
            }
            comparison['models'].append(model_summary)

        # Find best model by F1 macro score
        best_model = max(comparison['models'], key=lambda x: x['f1_macro'])
        comparison['best_model'] = best_model

        # Calculate ranking
        comparison['models'].sort(key=lambda x: x['f1_macro'], reverse=True)

        logger.info(f"Model comparison completed. Best model: {best_model['name']}")
        return comparison

    def save_comparison_results(
        self,
        comparison_results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Save comparison results to file.

        Args:
            comparison_results: Comparison results dictionary
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)

        logger.info(f"Comparison results saved to: {output_path}")


class PerformanceAnalyzer:
    """Analyze model performance across different scenarios."""

    def __init__(self, config: Config = Config):
        """Initialize the performance analyzer."""
        self.config = config

    def analyze_class_performance(
        self,
        true_labels: List[np.ndarray],
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze performance for each class separately.

        Args:
            true_labels: True labels
            predictions: Model predictions

        Returns:
            Dictionary containing per-class analysis
        """
        y_true = np.array(true_labels)
        y_pred = predictions

        class_analysis = {}

        for i, class_name in enumerate(['Violence', 'No Violence']):
            # Extract samples for this class
            class_mask = y_true[:, i] == 1
            class_true = y_true[class_mask]
            class_pred = y_pred[class_mask]

            if len(class_true) > 0:
                accuracy = accuracy_score(class_true, class_pred)
                class_analysis[class_name] = {
                    'sample_count': int(np.sum(class_mask)),
                    'accuracy': float(accuracy),
                    'percentage_of_dataset': float(np.sum(class_mask) / len(y_true) * 100)
                }

        return class_analysis

    def analyze_prediction_confidence(
        self,
        raw_predictions: np.ndarray,
        true_labels: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze prediction confidence distributions.

        Args:
            raw_predictions: Raw prediction probabilities
            true_labels: True labels

        Returns:
            Dictionary containing confidence analysis
        """
        y_true = np.array(true_labels)

        # Calculate confidence (max probability)
        confidence_scores = np.max(raw_predictions, axis=1)

        # Separate by correct/incorrect predictions
        predicted_labels = np.argmax(raw_predictions, axis=1)
        true_labels_single = np.argmax(y_true, axis=1)
        correct_mask = predicted_labels == true_labels_single

        correct_confidence = confidence_scores[correct_mask]
        incorrect_confidence = confidence_scores[~correct_mask]

        analysis = {
            'overall_confidence': {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores))
            },
            'correct_predictions_confidence': {
                'mean': float(np.mean(correct_confidence)) if len(correct_confidence) > 0 else 0,
                'std': float(np.std(correct_confidence)) if len(correct_confidence) > 0 else 0,
                'count': int(len(correct_confidence))
            },
            'incorrect_predictions_confidence': {
                'mean': float(np.mean(incorrect_confidence)) if len(incorrect_confidence) > 0 else 0,
                'std': float(np.std(incorrect_confidence)) if len(incorrect_confidence) > 0 else 0,
                'count': int(len(incorrect_confidence))
            },
            'high_confidence_threshold_0.9': {
                'total_high_confidence': int(np.sum(confidence_scores > 0.9)),
                'correct_high_confidence': int(np.sum(confidence_scores[correct_mask] > 0.9)),
                'accuracy_high_confidence': float(
                    np.sum(confidence_scores[correct_mask] > 0.9) /
                    max(np.sum(confidence_scores > 0.9), 1)
                )
            }
        }

        return analysis


def evaluate_model_from_cache(
    model_path: Path,
    test_cache_path: Path,
    config: Config = Config,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a model using cached test features.

    Args:
        model_path: Path to the saved model
        test_cache_path: Path to test features cache
        config: Configuration object
        threshold: Classification threshold

    Returns:
        Dictionary containing evaluation results
    """
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    evaluator.load_model(model_path)

    # Load test data
    feature_pipeline = FeaturePipeline(config)
    test_data, test_labels = feature_pipeline.load_processed_features(test_cache_path)

    # Perform evaluation
    results = evaluator.evaluate_model_comprehensive(test_data, test_labels, threshold)

    return results