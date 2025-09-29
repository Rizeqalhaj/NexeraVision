# NexaraVision - Violence Detection System

A production-ready deep learning system for real-time video violence detection using VGG19 feature extraction and LSTM-Attention architecture.

## System Overview

NexaraVision implements a two-stage deep learning pipeline:
1. **Feature Extraction**: VGG19 CNN extracts spatial features from video frames
2. **Sequence Classification**: LSTM with Attention mechanism processes temporal patterns

### Architecture

```
Video Input → Frame Extraction (20 frames) → VGG19 Features (4096) → LSTM+Attention → Violence/No Violence
```

## MVP Components

### Core Architecture
- **VGG19 Feature Extractor**: Pre-trained CNN for spatial feature extraction from video frames
- **LSTM-Attention Network**: Temporal sequence modeling with attention mechanism for violence detection
- **Transfer Learning**: Leverages ImageNet pre-trained weights for robust feature extraction
- **Real-time Inference Engine**: Optimized prediction pipeline for production deployment

### Data Flow
1. Video input processing (20 frames per video at 224x224 resolution)
2. VGG19 feature extraction (fc2 layer output: 4096 features per frame)
3. LSTM sequence processing with attention mechanism
4. Binary classification (Violence/No Violence)

## Data Structures and Schemas

### Core Data Types

#### Video Data Structure
```python
class VideoData:
    frames: np.ndarray              # Shape: (20, 224, 224, 3)
    features: np.ndarray            # Shape: (20, 4096) - VGG19 features
    label: np.ndarray               # Shape: (2,) - One-hot encoded [violence, no_violence]
    metadata: Dict[str, Any]        # Video metadata (filename, duration, etc.)
```

#### Model Configuration Schema
```python
class ModelConfig:
    IMG_SIZE: int = 224                    # Input image dimensions
    FRAMES_PER_VIDEO: int = 20             # Fixed frame count per video
    TRANSFER_VALUES_SIZE: int = 4096       # VGG19 fc2 layer output size
    RNN_SIZE: int = 128                    # LSTM hidden units
    NUM_CLASSES: int = 2                   # Binary classification
    BATCH_SIZE: int = 64                   # Training batch size
    LEARNING_RATE: float = 0.0001          # Adam optimizer learning rate
    DROPOUT_RATE: float = 0.5              # Regularization dropout rate
```

#### Training Data Schema
```python
class TrainingData:
    train_features: np.ndarray      # Shape: (N_train, 20, 4096)
    train_labels: np.ndarray        # Shape: (N_train, 2)
    test_features: np.ndarray       # Shape: (N_test, 20, 4096)
    test_labels: np.ndarray         # Shape: (N_test, 2)
    validation_split: float = 0.2   # Validation data percentage
```

#### Prediction Output Schema
```python
class PredictionResult:
    violence_probability: float     # Probability of violence (0-1)
    violence_detected: bool         # Binary classification result
    confidence: float              # Model confidence score
    frame_attention_weights: List[float]  # Per-frame attention weights
    processing_time: float         # Inference time in milliseconds
```

### File Structure Schemas

#### Dataset Organization
```
data/
├── raw/                          # Original video files
│   ├── fi_*.avi                 # Violence videos (prefix: fi, V)
│   ├── V_*.avi                  # Violence videos
│   ├── no_*.avi                 # Non-violence videos (prefix: no, NV)
│   └── NV_*.avi                 # Non-violence videos
├── processed/                    # Cached features
│   ├── train_features.h5        # Training set VGG19 features
│   └── test_features.h5         # Test set VGG19 features
└── cache/                       # Temporary processing files
```

#### Model Artifacts
```
models/
├── violence_detection_model.h5   # Main trained model
├── checkpoint.h5                 # Best training checkpoint
├── vgg19_feature_extractor.h5    # Cached VGG19 model
└── model_metadata.json          # Training history and metrics
```

### Neural Network Architecture Schema

#### VGG19 Feature Extraction Pipeline
```python
VGG19_ARCHITECTURE = {
    'input_shape': (224, 224, 3),
    'transfer_layer': 'fc2',
    'output_features': 4096,
    'weights': 'imagenet',
    'preprocessing': 'tf_normalize'
}
```

#### LSTM-Attention Model Schema
```python
LSTM_ATTENTION_ARCHITECTURE = {
    'input_layer': {
        'shape': (20, 4096),
        'name': 'video_input'
    },
    'lstm_layers': [
        {'units': 128, 'return_sequences': True, 'name': 'lstm_1'},
        {'units': 128, 'return_sequences': True, 'name': 'lstm_2'},
        {'units': 128, 'return_sequences': True, 'name': 'lstm_3'}
    ],
    'attention_layer': {
        'type': 'custom_attention',
        'name': 'attention'
    },
    'dense_layers': [
        {'units': 256, 'activation': 'relu', 'name': 'dense_1'},
        {'units': 128, 'activation': 'relu', 'name': 'dense_2'},
        {'units': 64, 'activation': 'relu', 'name': 'dense_3'}
    ],
    'output_layer': {
        'units': 2,
        'activation': 'softmax',
        'name': 'output'
    },
    'regularization': {
        'dropout_rate': 0.5,
        'batch_normalization': True
    }
}
```

### Performance Metrics Schema

#### Model Performance
```python
class PerformanceMetrics:
    accuracy: float                # Overall classification accuracy
    precision: Dict[str, float]    # Per-class precision scores
    recall: Dict[str, float]       # Per-class recall scores
    f1_score: Dict[str, float]     # Per-class F1 scores
    confusion_matrix: np.ndarray   # 2x2 confusion matrix
    roc_auc: float                # ROC Area Under Curve
    training_time: float          # Model training duration (seconds)
    inference_time: float         # Average prediction time (milliseconds)
```

#### System Specifications
```python
class SystemRequirements:
    python_version: str = "3.8+"
    tensorflow_version: str = "2.x"
    memory_requirements: Dict[str, str] = {
        'training': '8GB RAM',
        'inference': '4GB RAM',
        'gpu_memory': '4GB VRAM (optional)'
    }
    storage_requirements: Dict[str, str] = {
        'model_size': '~10MB',
        'feature_cache': '~1GB per 1000 videos',
        'training_data': 'Variable based on dataset size'
    }
```

## Technical Specifications

### Model Architecture
- **Total Parameters**: ~2.5M trainable parameters
- **Model Size**: ~9.5 MB
- **Input Format**: 20 frames per video at 224x224x3 resolution
- **Output Format**: 2-class softmax probability distribution

### Performance Characteristics
- **Training Time**: ~2-4 hours on GPU for 1000+ videos
- **Inference Speed**: 10-15ms per video on GPU, 50-100ms on CPU
- **Memory Usage**: 4-8GB RAM during training, 2-4GB during inference
- **Expected Accuracy**: 85-90% on balanced test sets

### Data Processing Pipeline
1. **Video Loading**: OpenCV-based frame extraction
2. **Preprocessing**: Resize to 224x224, normalize to [0,1]
3. **Feature Extraction**: VGG19 fc2 layer (4096 features per frame)
4. **Sequence Formation**: Stack 20 frames into temporal sequences
5. **Attention Processing**: Weighted feature aggregation
6. **Classification**: Softmax probability output

## Installation and Usage

### Quick Start
```bash
git clone https://github.com/psdew2ewqws/NexeraVision.git
cd NexeraVision/violence_detection_mvp
pip install -r requirements.txt
python -m src.main demo --data-dir data/raw
```

### API Usage
```python
from src.inference import ViolencePredictor

predictor = ViolencePredictor("models/violence_detection_model.h5")
result = predictor.predict_video("path/to/video.avi")
print(f"Violence detected: {result['violence_detected']}")
```

## Project Structure

```
NexaraVision/
├── violence_detection_mvp/       # Main MVP implementation
│   ├── src/                      # Source code modules
│   │   ├── config.py            # Configuration management
│   │   ├── model_architecture.py # LSTM-Attention model
│   │   ├── feature_extraction.py # VGG19 pipeline
│   │   ├── training.py          # Training pipeline
│   │   ├── inference.py         # Prediction engine
│   │   └── evaluation.py        # Performance metrics
│   ├── data/                    # Dataset and features
│   ├── models/                  # Trained model files
│   └── requirements.txt         # Dependencies
├── docs/                        # Additional documentation
└── README.md                    # This file
```

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{nexaravision2024,
  title={NexaraVision: Deep Learning Violence Detection System},
  author={NEXARA Team},
  year={2024},
  url={https://github.com/psdew2ewqws/NexeraVision}
}
```