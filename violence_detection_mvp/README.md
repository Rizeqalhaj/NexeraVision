# Violence Detection MVP

A comprehensive video violence detection system using deep learning with VGG19 feature extraction and LSTM-Attention architecture.

## Overview

This project implements a robust violence detection system that:
- Extracts features from video frames using pre-trained VGG19
- Uses LSTM with Attention mechanism for sequence classification
- Provides real-time inference capabilities
- Includes comprehensive evaluation and visualization tools

## Architecture

```
Video Input → Frame Extraction → VGG19 Features → LSTM+Attention → Violence/No Violence
```

### Key Components

1. **VGG19 Feature Extractor**: Pre-trained CNN for spatial feature extraction
2. **LSTM with Attention**: Temporal sequence modeling with attention mechanism
3. **Transfer Learning**: Leverages ImageNet pre-trained weights
4. **Real-time Inference**: Optimized for production deployment

## Project Structure

```
violence_detection_mvp/
├── src/                          # Source code
│   ├── config.py                 # Configuration management
│   ├── data_preprocessing.py     # Video frame extraction and labeling
│   ├── feature_extraction.py     # VGG19 feature pipeline
│   ├── model_architecture.py     # LSTM-Attention model
│   ├── training.py              # Training pipeline
│   ├── evaluation.py            # Model evaluation and metrics
│   ├── inference.py             # Real-time prediction
│   ├── visualization.py         # Plotting and charts
│   └── utils.py                 # Helper functions
├── data/                        # Data directories
│   ├── raw/                     # Original video files
│   └── processed/               # Cached features
├── models/                      # Saved model checkpoints
├── notebooks/                   # Jupyter notebooks
│   └── mvp_demo.ipynb          # End-to-end demo
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd violence_detection_mvp
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "from src import Config; print('Installation successful!')"
```

## Quick Start

### 1. Data Preparation

Place video files in the `data/raw/` directory with naming convention:
- Violence videos: `fi_*.avi`, `V_*.avi`
- Non-violence videos: `no_*.avi`, `NV_*.avi`

### 2. CLI Interface

The project includes a comprehensive command-line interface for all operations:

#### Complete Demo
```bash
# Run end-to-end demo (training, evaluation, inference)
python -m src.main demo --data-dir data/raw
```

#### Training
```bash
# Train model with visualizations
python -m src.main train --data-dir data/raw --visualize
```

#### Evaluation
```bash
# Evaluate trained model
python -m src.main evaluate --model-path models/violence_detection_model.h5 --data-dir data/raw --visualize
```

#### Single Video Inference
```bash
# Predict single video
python -m src.main infer --model-path models/violence_detection_model.h5 --video-path video.avi --confidence 0.7 --verbose
```

#### Batch Video Inference
```bash
# Process all videos in directory
python -m src.main infer --model-path models/violence_detection_model.h5 --batch-mode --video-dir data/test --output-file results.json --verbose

# Process videos from list file
python -m src.main infer --model-path models/violence_detection_model.h5 --batch-mode --video-list video_list.txt --output-file results.json
```

#### Real-time Processing
```bash
# Process webcam feed
python -m src.main realtime --model-path models/violence_detection_model.h5 --camera 0 --display

# Process video file in real-time
python -m src.main realtime --model-path models/violence_detection_model.h5 --video-file input.mp4 --display --output-file realtime_results.json
```

#### System Information
```bash
# Check system and project status
python -m src.main info
```

### 3. Python API Usage

#### Training
```python
from src.training import TrainingPipeline
from pathlib import Path

# Initialize training pipeline
trainer = TrainingPipeline()

# Prepare data and train model
data_dir = Path(\"data/raw\")
train_data, train_targets, test_data, test_targets = trainer.prepare_data(data_dir)
history = trainer.train_model(train_data, train_targets, test_data, test_targets)
```

#### Evaluation
```python
from src.evaluation import ModelEvaluator
from pathlib import Path

# Load and evaluate model
evaluator = ModelEvaluator()
evaluator.load_model(Path(\"models/violence_detection_model.h5\"))
results = evaluator.evaluate_model_comprehensive(test_data, test_targets)

print(f\"Accuracy: {results['metrics']['accuracy']:.4f}\")
print(f\"F1 Score: {results['metrics']['f1_macro']:.4f}\")
```

#### Inference
```python
from src.inference import ViolencePredictor
from pathlib import Path

# Initialize predictor
predictor = ViolencePredictor(Path(\"models/violence_detection_model.h5\"))

# Predict single video
result = predictor.predict_video(Path(\"path/to/video.avi\"))
print(f\"Violence detected: {result['violence_detected']}\")
print(f\"Confidence: {result['confidence']:.3f}\")
```

## Configuration

Key configuration parameters in `src/config.py`:

```python
# Model parameters
IMG_SIZE = 224                    # Input image size
FRAMES_PER_VIDEO = 20            # Number of frames per video
RNN_SIZE = 128                   # LSTM hidden size
BATCH_SIZE = 64                  # Training batch size
LEARNING_RATE = 0.0001           # Learning rate
EPOCHS = 250                     # Training epochs

# Data paths
RAW_DATA_DIR = \"data/raw\"        # Original videos
PROCESSED_DATA_DIR = \"data/processed\"  # Cached features
MODELS_DIR = \"models\"            # Saved models
```

## Model Architecture

### VGG19 Feature Extraction
- Pre-trained on ImageNet
- Uses `fc2` layer output (4096 features)
- Processes 20 frames per video

### LSTM with Attention
```
Input: (batch_size, 20, 4096)
├── LSTM Layer 1 (128 units) + BatchNorm + Dropout
├── LSTM Layer 2 (128 units) + BatchNorm + Dropout
├── LSTM Layer 3 (128 units) + BatchNorm + Dropout
├── Attention Mechanism
├── Dense Layer 1 (256 units) + BatchNorm + ReLU + Dropout
├── Dense Layer 2 (128 units) + BatchNorm + ReLU + Dropout
├── Dense Layer 3 (64 units) + ReLU + Dropout
└── Output Layer (2 units, Softmax)
```

## Performance

### Model Metrics
- **Parameters**: ~2.5M trainable parameters
- **Model Size**: ~9.5 MB
- **Inference Speed**: ~10-15ms per video (GPU)

### Expected Results
- **Accuracy**: 85-90% on test set
- **F1 Score**: 0.85-0.90 (macro average)
- **Precision**: 0.85-0.92 per class
- **Recall**: 0.84-0.94 per class

## Advanced Usage

### Experiment Management

```python
from src.training import ExperimentManager
from pathlib import Path

# Run multiple experiments
manager = ExperimentManager()

# Experiment 1: Default parameters
results1 = manager.run_experiment(\"baseline\", Path(\"data/raw\"))

# Experiment 2: Modified hyperparameters
hyperparams = {\"LEARNING_RATE\": 0.0005, \"RNN_SIZE\": 256}
results2 = manager.run_experiment(\"larger_model\", Path(\"data/raw\"), hyperparams)

# Compare results
comparison = manager.compare_experiments()
print(f\"Best model: {comparison['best_experiment']['name']}\")
```

### Real-time Processing

```python
from src.inference import RealTimeVideoProcessor

# Process video stream
processor = RealTimeVideoProcessor(Path(\"models/violence_detection_model.h5\"))

# Process webcam feed
processor.process_video_stream(0, display=True)

# Process video file
processor.process_video_stream(\"path/to/video.mp4\", display=True)
```

### Visualization

```python
from src.visualization import TrainingVisualizer, EvaluationVisualizer

# Plot training curves
training_viz = TrainingVisualizer()
training_viz.plot_training_history(history.history, save_path=\"training_curves.png\")

# Plot evaluation results
eval_viz = EvaluationVisualizer()
eval_viz.plot_confusion_matrix(confusion_matrix, save_path=\"confusion_matrix.png\")
eval_viz.plot_roc_curves(roc_data, save_path=\"roc_curves.png\")
```

## CLI Reference

### Command Overview

The main CLI interface supports the following commands:

| Command | Description | Key Options |
|---------|-------------|-------------|
| `demo` | Complete end-to-end demonstration | `--data-dir`, `--model-path` |
| `train` | Train violence detection model | `--data-dir`, `--visualize` |
| `evaluate` | Evaluate model performance | `--model-path`, `--data-dir`, `--visualize` |
| `infer` | Run inference on videos | `--model-path`, `--video-path`, `--batch-mode` |
| `realtime` | Real-time video processing | `--model-path`, `--camera`, `--display` |
| `info` | System and project information | No options |

### CLI Command Details

#### Demo Command
```bash
python -m src.main demo [options]
```
- `--data-dir PATH`: Directory containing demo videos (default: data/raw)
- `--model-path PATH`: Path to existing model (optional, will train if missing)

#### Train Command
```bash
python -m src.main train [options]
```
- `--data-dir PATH`: Directory containing training videos (default: data/raw)
- `--visualize`: Generate training visualizations

#### Evaluate Command
```bash
python -m src.main evaluate [options]
```
- `--model-path PATH`: Path to trained model file (required)
- `--data-dir PATH`: Directory containing test videos (default: data/raw)
- `--visualize`: Generate evaluation visualizations

#### Infer Command
```bash
python -m src.main infer [options]
```
- `--model-path PATH`: Path to trained model file (required)
- `--video-path PATH`: Path to single video file
- `--batch-mode`: Process multiple videos
- `--video-dir PATH`: Directory containing videos (batch mode)
- `--video-list PATH`: Text file with video paths (batch mode)
- `--confidence FLOAT`: Confidence threshold for predictions (default: 0.5)
- `--output-file PATH`: Save results to JSON file
- `--verbose`: Show detailed prediction information

#### Realtime Command
```bash
python -m src.main realtime [options]
```
- `--model-path PATH`: Path to trained model file (required)
- `--camera INT`: Camera index (default: 0)
- `--video-file PATH`: Process video file instead of camera
- `--display`: Display video with predictions
- `--output-file PATH`: Save real-time results to JSON file

#### Info Command
```bash
python -m src.main info
```
Shows system information, dependencies, project structure, and model status.

### Usage Examples

#### Complete Workflow
```bash
# 1. Check system status
python -m src.main info

# 2. Run complete demo
python -m src.main demo --data-dir data/raw

# 3. Train with custom data
python -m src.main train --data-dir path/to/videos --visualize

# 4. Evaluate model
python -m src.main evaluate --model-path models/violence_detection_model.h5 --data-dir data/test --visualize

# 5. Run batch inference
python -m src.main infer --model-path models/violence_detection_model.h5 --batch-mode --video-dir data/test --output-file results.json --verbose

# 6. Real-time processing
python -m src.main realtime --model-path models/violence_detection_model.h5 --camera 0 --display
```

## API Usage

### Inference API

```python
from src.inference import InferenceAPI

# Initialize API
api = InferenceAPI(Path(\"models/violence_detection_model.h5\"))

# Single video prediction
result = api.predict_single_video(\"video.avi\", confidence_threshold=0.7)

# Batch prediction
video_paths = [\"video1.avi\", \"video2.avi\", \"video3.avi\"]
results = api.predict_multiple_videos(video_paths)

# Get model info
info = api.get_model_info()
```

## Troubleshooting

### Common Issues

1. **Memory Error during training**:
   - Reduce `BATCH_SIZE` in config
   - Use smaller `RNN_SIZE`
   - Process fewer videos at once

2. **CUDA out of memory**:
   - Reduce batch size
   - Use CPU-only mode: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

3. **Video loading errors**:
   - Check video format (supported: .avi, .mp4, .mov, .mkv)
   - Verify OpenCV installation: `pip install opencv-python`

4. **Model loading errors**:
   - Ensure model file exists and is not corrupted
   - Check TensorFlow compatibility

### Performance Optimization

1. **GPU Acceleration**:
```python
import tensorflow as tf
print(\"GPU Available: \", tf.config.list_physical_devices('GPU'))
```

2. **Memory Management**:
```python
# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

3. **Batch Processing**:
- Process multiple videos in batches for better throughput
- Use `predict_video_batch()` instead of individual calls

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{violence_detection_mvp,
  title={Violence Detection MVP: Deep Learning Video Analysis},
  author={NEXARA Team},
  year={2024},
  url={https://github.com/nexara/violence-detection-mvp}
}
```

## Acknowledgments

- VGG19 architecture from [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- Attention mechanism inspired by [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Transfer learning approach based on ImageNet pre-trained models

## Contact

For questions and support:
- Email: support@nexara.com
- Issues: GitHub Issues
- Documentation: Project Wiki