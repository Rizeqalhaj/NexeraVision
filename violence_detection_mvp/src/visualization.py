"""
Visualization module for Violence Detection MVP.
Handles plotting training curves, confusion matrices, and analysis charts.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualize training metrics and curves."""

    def __init__(self, config: Config = Config):
        """Initialize the training visualizer."""
        self.config = config
        self.figure_size = config.FIGURE_SIZE
        self.dpi = config.PLOT_DPI

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot training and validation curves.

        Args:
            history: Training history dictionary
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        if 'accuracy' in history:
            ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)

        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot loss
        if 'loss' in history:
            ax2.plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)

        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_learning_rate_schedule(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Optional[Figure]:
        """
        Plot learning rate schedule if available.

        Args:
            history: Training history dictionary
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object or None
        """
        if 'lr' not in history:
            logger.warning("Learning rate data not found in history")
            return None

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.plot(history['lr'], linewidth=2, color='red')
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot comparison of different metrics across models or experiments.

        Args:
            metrics_dict: Dictionary of model names to metrics
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        model_names = list(metrics_dict.keys())
        metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metric_names):
            values = [metrics_dict[model].get(metric, 0) for model in model_names]

            bars = axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

            # Rotate x-axis labels if too many models
            if len(model_names) > 3:
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def _save_figure(self, fig: Figure, save_path: Path) -> None:
        """Save figure to file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")


class EvaluationVisualizer:
    """Visualize evaluation results and metrics."""

    def __init__(self, config: Config = Config):
        """Initialize the evaluation visualizer."""
        self.config = config
        self.figure_size = config.FIGURE_SIZE
        self.dpi = config.PLOT_DPI
        self.class_names = ['Violence', 'No Violence']

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        normalize: bool = False,
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array
            normalize: Whether to normalize the matrix
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'

        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_multilabel_confusion_matrices(
        self,
        multilabel_cm: np.ndarray,
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot multilabel confusion matrices as in the original implementation.

        Args:
            multilabel_cm: Multilabel confusion matrix array
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for i, (ax, class_name) in enumerate(zip(axes, self.class_names)):
            cm = multilabel_cm[i]

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'],
                ax=ax
            )

            ax.set_title(f'Confusion Matrix for {class_name}', fontweight='bold')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict[str, List[float]]],
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot ROC curves for each class.

        Args:
            roc_data: ROC curve data with FPR and TPR for each class
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        colors = ['blue', 'red', 'green']

        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if class_name in roc_data:
                fpr = roc_data[class_name]['fpr']
                tpr = roc_data[class_name]['tpr']
                auc_score = np.trapz(tpr, fpr)

                ax.plot(
                    fpr, tpr, color=color, linewidth=2,
                    label=f'{class_name} (AUC = {auc_score:.3f})'
                )

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_precision_recall_curves(
        self,
        pr_data: Dict[str, Dict[str, List[float]]],
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot Precision-Recall curves for each class.

        Args:
            pr_data: Precision-Recall curve data
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        colors = ['blue', 'red', 'green']

        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if class_name in pr_data:
                precision = pr_data[class_name]['precision']
                recall = pr_data[class_name]['recall']
                auc_score = np.trapz(precision, recall)

                ax.plot(
                    recall, precision, color=color, linewidth=2,
                    label=f'{class_name} (AUC = {auc_score:.3f})'
                )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot class distribution.

        Args:
            class_counts: Dictionary of class names to counts
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ['#ff6b6b', '#4ecdc4']

        # Bar plot
        bars = ax1.bar(classes, counts, color=colors, alpha=0.7)
        ax1.set_title('Class Distribution (Bar Chart)', fontweight='bold')
        ax1.set_ylabel('Number of Samples')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                   f'{count}', ha='center', va='bottom')

        # Pie chart
        ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (Pie Chart)', fontweight='bold')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def _save_figure(self, fig: Figure, save_path: Path) -> None:
        """Save figure to file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")


class DataVisualizer:
    """Visualize data and preprocessing results."""

    def __init__(self, config: Config = Config):
        """Initialize the data visualizer."""
        self.config = config
        self.figure_size = config.FIGURE_SIZE
        self.dpi = config.PLOT_DPI

    def plot_sample_frames(
        self,
        frames: np.ndarray,
        title: str = "Sample Video Frames",
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot sample frames from a video.

        Args:
            frames: Array of video frames
            title: Plot title
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        num_frames = min(len(frames), 8)  # Show up to 8 frames
        cols = 4
        rows = (num_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))

        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i in range(num_frames):
            frame = frames[i]

            # Ensure frame is in correct format for display
            if frame.dtype == np.float32 or frame.dtype == np.float16:
                frame = (frame * 255).astype(np.uint8)

            axes[i].imshow(frame)
            axes[i].set_title(f'Frame {i + 1}')
            axes[i].axis('off')

        # Hide remaining subplots
        for i in range(num_frames, len(axes)):
            axes[i].axis('off')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_feature_statistics(
        self,
        features: np.ndarray,
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot feature statistics and distributions.

        Args:
            features: Feature array
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Feature mean histogram
        feature_means = np.mean(features, axis=0)
        axes[0, 0].hist(feature_means, bins=50, alpha=0.7)
        axes[0, 0].set_title('Distribution of Feature Means')
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Frequency')

        # Feature standard deviation histogram
        feature_stds = np.std(features, axis=0)
        axes[0, 1].hist(feature_stds, bins=50, alpha=0.7, color='orange')
        axes[0, 1].set_title('Distribution of Feature Standard Deviations')
        axes[0, 1].set_xlabel('Std Value')
        axes[0, 1].set_ylabel('Frequency')

        # Sample feature vector
        sample_idx = np.random.randint(0, len(features))
        axes[1, 0].plot(features[sample_idx], alpha=0.7)
        axes[1, 0].set_title(f'Sample Feature Vector (Sample {sample_idx})')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Feature Value')

        # Feature correlation heatmap (subset)
        if features.shape[1] > 100:
            # Use subset for correlation
            subset_indices = np.random.choice(features.shape[1], 50, replace=False)
            subset_features = features[:, subset_indices]
        else:
            subset_features = features

        correlation_matrix = np.corrcoef(subset_features.T)
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Feature Correlation Matrix (Subset)')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_data_pipeline_summary(
        self,
        pipeline_stats: Dict[str, Any],
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot summary of data pipeline processing.

        Args:
            pipeline_stats: Dictionary containing pipeline statistics
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Processing time by stage
        if 'processing_times' in pipeline_stats:
            stages = list(pipeline_stats['processing_times'].keys())
            times = list(pipeline_stats['processing_times'].values())

            axes[0, 0].bar(stages, times, alpha=0.7)
            axes[0, 0].set_title('Processing Time by Stage')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # Data size progression
        if 'data_sizes' in pipeline_stats:
            stages = list(pipeline_stats['data_sizes'].keys())
            sizes = list(pipeline_stats['data_sizes'].values())

            axes[0, 1].plot(stages, sizes, marker='o', linewidth=2, markersize=8)
            axes[0, 1].set_title('Data Size Progression')
            axes[0, 1].set_ylabel('Size (MB)')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Error rates
        if 'error_rates' in pipeline_stats:
            stages = list(pipeline_stats['error_rates'].keys())
            error_rates = list(pipeline_stats['error_rates'].values())

            axes[1, 0].bar(stages, error_rates, alpha=0.7, color='red')
            axes[1, 0].set_title('Error Rates by Stage')
            axes[1, 0].set_ylabel('Error Rate (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # Quality metrics
        if 'quality_metrics' in pipeline_stats:
            metrics = list(pipeline_stats['quality_metrics'].keys())
            values = list(pipeline_stats['quality_metrics'].values())

            axes[1, 1].bar(metrics, values, alpha=0.7, color='green')
            axes[1, 1].set_title('Quality Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def _save_figure(self, fig: Figure, save_path: Path) -> None:
        """Save figure to file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")


class ModelVisualizer:
    """Visualize model architecture and predictions."""

    def __init__(self, config: Config = Config):
        """Initialize the model visualizer."""
        self.config = config
        self.figure_size = config.FIGURE_SIZE
        self.dpi = config.PLOT_DPI

    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        frame_indices: Optional[List[int]] = None,
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot attention weights across video frames.

        Args:
            attention_weights: Attention weight array
            frame_indices: Optional frame indices for x-axis
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        if frame_indices is None:
            frame_indices = list(range(len(attention_weights)))

        ax.bar(frame_indices, attention_weights, alpha=0.7)
        ax.set_title('Attention Weights Across Video Frames', fontweight='bold')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Attention Weight')
        ax.grid(True, alpha=0.3)

        # Highlight frames with highest attention
        max_attention_idx = np.argmax(attention_weights)
        ax.bar(frame_indices[max_attention_idx], attention_weights[max_attention_idx],
               color='red', alpha=0.8, label=f'Max Attention (Frame {max_attention_idx})')
        ax.legend()

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_prediction_confidence_distribution(
        self,
        confidences: List[float],
        predictions: List[bool],
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Plot distribution of prediction confidences.

        Args:
            confidences: List of confidence scores
            predictions: List of prediction results (True for violence)
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Overall confidence distribution
        axes[0].hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_title('Overall Prediction Confidence Distribution')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(np.mean(confidences), color='red', linestyle='--',
                       label=f'Mean: {np.mean(confidences):.3f}')
        axes[0].legend()

        # Confidence by prediction class
        violence_confidences = [conf for conf, pred in zip(confidences, predictions) if pred]
        no_violence_confidences = [conf for conf, pred in zip(confidences, predictions) if not pred]

        axes[1].hist(violence_confidences, bins=20, alpha=0.7, label='Violence', color='red')
        axes[1].hist(no_violence_confidences, bins=20, alpha=0.7, label='No Violence', color='blue')
        axes[1].set_title('Confidence Distribution by Predicted Class')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def _save_figure(self, fig: Figure, save_path: Path) -> None:
        """Save figure to file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")


def create_comprehensive_report(
    training_history: Dict[str, List[float]],
    evaluation_results: Dict[str, Any],
    model_config: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Create a comprehensive visualization report.

    Args:
        training_history: Training history data
        evaluation_results: Evaluation results
        model_config: Model configuration
        output_dir: Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizers
    training_viz = TrainingVisualizer()
    eval_viz = EvaluationVisualizer()

    # Plot training curves
    training_viz.plot_training_history(
        training_history,
        save_path=output_dir / "training_curves.png",
        show_plot=False
    )

    # Plot confusion matrix
    if 'confusion_matrix' in evaluation_results:
        cm = np.array(evaluation_results['confusion_matrix'])
        eval_viz.plot_confusion_matrix(
            cm,
            save_path=output_dir / "confusion_matrix.png",
            show_plot=False
        )

    # Plot multilabel confusion matrices
    if 'multilabel_confusion_matrix' in evaluation_results:
        mcm = np.array(evaluation_results['multilabel_confusion_matrix'])
        eval_viz.plot_multilabel_confusion_matrices(
            mcm,
            save_path=output_dir / "multilabel_confusion_matrices.png",
            show_plot=False
        )

    # Plot ROC curves
    if 'roc_auc' in evaluation_results and 'curves' in evaluation_results['roc_auc']:
        eval_viz.plot_roc_curves(
            evaluation_results['roc_auc']['curves'],
            save_path=output_dir / "roc_curves.png",
            show_plot=False
        )

    logger.info(f"Comprehensive report created in: {output_dir}")


def save_all_plots_as_eps(plot_functions: List[callable], output_dir: Path) -> None:
    """
    Save all plots in EPS format for publication.

    Args:
        plot_functions: List of plot function calls
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, plot_func in enumerate(plot_functions):
        try:
            fig = plot_func()
            eps_path = output_dir / f"plot_{i:02d}.eps"
            fig.savefig(eps_path, format='eps', dpi=1000, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved EPS plot: {eps_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {i}: {str(e)}")