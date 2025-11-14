# Violence Detection Models

This directory contains trained models and checkpoints.

## Model Locations

### Local Development (Desktop/Laptop)
```
/home/admin/Desktop/NexaraVision/violence_detection_mvp/models/
├── best_model.h5                    ← Best model (highest val_accuracy)
├── checkpoint_epoch_001.h5          ← Checkpoint after epoch 1
├── checkpoint_epoch_002.h5          ← Checkpoint after epoch 2
├── checkpoint_epoch_003.h5          ← And so on...
└── ...
```

### Cloud Training (vast.ai / RunPod)
```
/workspace/violence_detection_mvp/models/
├── best_model.h5
├── checkpoint_epoch_001.h5
└── ...
```

## Auto-Detection

The training script automatically detects which location to use:
1. **First tries:** `/workspace/violence_detection_mvp/models/` (cloud)
2. **Falls back to:** Project directory `/models/` (local)

## Model Files

### best_model.h5
- **Purpose:** Best model based on validation accuracy
- **Auto-saved:** When val_accuracy improves
- **Usage:** This is the model you want to use for inference

### checkpoint_epoch_XXX.h5
- **Purpose:** Regular checkpoints every epoch
- **Usage:** Resume training if interrupted
- **Example:**
  ```bash
  python train_rtx5000_dual_IMPROVED.py \
      --dataset-path /path/to/dataset \
      --resume models/checkpoint_epoch_025.h5
  ```

## Training Output Locations

### Models
- **Location:** `violence_detection_mvp/models/`
- **Files:** `best_model.h5`, `checkpoint_epoch_*.h5`

### Training Logs
- **Location:** `checkpoints_improved/`
- **Files:**
  - `training_history.csv` - Metrics per epoch
  - `training_results.json` - Final results
  - `training_config.json` - Configuration used

### TensorBoard
- **Location:** `checkpoints_improved/tensorboard/`
- **View:** `tensorboard --logdir checkpoints_improved/tensorboard/`

## Model Info

### Architecture
- **Type:** Bidirectional LSTM with Attention
- **Input:** VGG19 features (20 frames × 4096 features)
- **Output:** Binary classification (violent / non-violent)
- **Parameters:** ~3.8M trainable parameters

### Expected Performance
- **Accuracy:** 93-95% (with 30K balanced dataset)
- **Precision:** ~94%
- **Recall:** ~93%
- **AUC:** ~96%

### Model Size
- **File size:** ~45-50 MB (uncompressed .h5)
- **Memory:** ~200 MB in RAM during inference
- **VRAM:** ~1-2 GB during inference (with batch processing)

## Loading Models

### Python
```python
from tensorflow import keras

# Load best model
model = keras.models.load_model('violence_detection_mvp/models/best_model.h5')

# Make predictions
predictions = model.predict(features)
```

### Resume Training
```bash
python train_rtx5000_dual_IMPROVED.py \
    --dataset-path /path/to/dataset \
    --resume violence_detection_mvp/models/checkpoint_epoch_050.h5
```

## Backup Recommendations

### Local Development
```bash
# Backup models to external drive or cloud
cp -r violence_detection_mvp/models/ /backup/location/
```

### Cloud Training
```bash
# Download models from cloud instance
scp user@cloud:/workspace/violence_detection_mvp/models/best_model.h5 ./local/models/
```

## Model Versioning

When training multiple versions, use descriptive names:
```
models/
├── best_model_14k_91pct.h5          ← 14K dataset, 91% accuracy
├── best_model_30k_93pct.h5          ← 30K dataset, 93% accuracy
├── best_model_30k_bidirectional.h5  ← With bidirectional LSTM
└── best_model_ensemble.h5           ← Ensemble model
```

## Troubleshooting

### "Permission denied" error
- Check directory permissions: `ls -la violence_detection_mvp/`
- Create directory manually: `mkdir -p violence_detection_mvp/models/`

### "Models not found" after training
- Check training logs for actual save location
- Look for: `💾 Models will be saved to: /path/to/models/`

### Large checkpoint files
- Each checkpoint is ~45-50 MB
- 100 epochs = ~5 GB total
- Solution: Reduce checkpoint frequency or delete old checkpoints
