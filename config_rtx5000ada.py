"""
Optimized Configuration for 2× RTX 5000 Ada (64GB VRAM)
AMD Threadripper PRO 7945WX 24-core, 257GB RAM
"""

# ============================================
# GPU CONFIGURATION - 2× RTX 5000 Ada
# ============================================
GPU_COUNT = 2
GPU_MEMORY = 32  # GB per GPU
TOTAL_VRAM = 64  # GB total

# Mixed precision for 2× speed boost
MIXED_PRECISION = True
DTYPE = 'float16'

# ============================================
# BATCH SIZE OPTIMIZATION - 64GB VRAM
# ============================================
# RTX 5000 Ada has 32GB per GPU - we can use MUCH larger batches

# Feature Extraction (EfficientNetB4)
FEATURE_BATCH_SIZE = 96  # Per GPU (192 total with 2 GPUs)

# Training batch sizes
BATCH_SIZE_PER_GPU = 128  # Ultimate model
GLOBAL_BATCH_SIZE = 256  # 128 × 2 GPUs

# Ensemble training (can fit 2 models simultaneously)
ENSEMBLE_BATCH_SIZE = 96  # Per GPU per model

# ============================================
# CPU OPTIMIZATION - 24 cores Threadripper
# ============================================
NUM_WORKERS = 20  # Leave 4 cores for system
PREFETCH_BUFFER = 8  # Larger buffer with 257GB RAM

# ============================================
# MEMORY OPTIMIZATION - 257GB RAM
# ============================================
DATASET_CACHE = True  # Cache entire dataset in RAM
MAX_QUEUE_SIZE = 32  # Larger queue with abundant RAM
MAX_MEMORY_MB = 200000  # 200GB for dataset caching

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
# Feature extraction
SEQUENCE_LENGTH = 30  # 30 frames per video
IMG_SIZE = (224, 224)

# Ultimate model training
EPOCHS = 80  # More epochs with faster hardware
LEARNING_RATE = 0.0005  # Optimal for 256 batch size
DROPOUT = 0.4

# ============================================
# PERFORMANCE SETTINGS
# ============================================
# Use XLA compilation for 15-30% speedup
XLA_COMPILE = True

# TensorFlow optimizations
TF_GPU_THREAD_MODE = 'gpu_private'
TF_GPU_THREAD_COUNT = 2  # Per GPU
TF_INTER_OP_THREADS = 24  # Match CPU cores
TF_INTRA_OP_THREADS = 24

# Data augmentation (10× augmentation for better accuracy)
AUGMENTATION_FACTOR = 10
AUGMENT_ROTATION = 10
AUGMENT_ZOOM = 0.1
AUGMENT_BRIGHTNESS = 0.2

# ============================================
# STORAGE PATHS - 1TB NVMe
# ============================================
BASE_DIR = '/workspace'
RAW_DATA_DIR = f'{BASE_DIR}/datasets'
PROCESSED_DATA_DIR = f'{BASE_DIR}/data/combined'
FEATURES_DIR = f'{BASE_DIR}/features'
MODELS_DIR = f'{BASE_DIR}/models'

# ============================================
# ESTIMATED TRAINING TIMES (2× RTX 5000 Ada)
# ============================================
"""
Phase 1: Feature Extraction (95K videos)
- Time: 4-5 hours
- Batch: 192 videos/iteration
- Speed: ~320 videos/minute

Phase 2: Ultimate Model Training
- Time: 10-12 hours
- Epochs: 80
- Accuracy: 88-93%

Phase 3: Ensemble Training (5 models)
- Time: 18-22 hours
- Can train 2 models in parallel
- Final Accuracy: 93-97%

TOTAL: 32-39 hours × $1.07/hr = $34-42
"""

# ============================================
# COST OPTIMIZATION
# ============================================
CHECKPOINT_FREQUENCY = 10  # Save every 10 epochs
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement
AUTO_SHUTDOWN = True  # Shutdown pod when complete

# ============================================
# MONITORING
# ============================================
LOG_FREQUENCY = 100  # Log every 100 batches
TENSORBOARD = True
WANDB_LOGGING = False  # Set to True if using Weights & Biases

# ============================================
# ENSEMBLE CONFIGURATION
# ============================================
ENSEMBLE_MODELS = [
    'deep_lstm',
    'bidirectional_lstm',
    'gru_attention',
    'conv1d_lstm',
    'attention_lstm'
]

# Parallel ensemble training (2 models at once with 64GB VRAM)
PARALLEL_ENSEMBLE_TRAINING = True
MODELS_PER_BATCH = 2
