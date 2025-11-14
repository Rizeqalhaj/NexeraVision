# 📂 Model Save Locations - Updated Configuration

## ✅ Changes Made

The training script now saves models to a dedicated `models/` directory with automatic location detection.

## 📍 Save Locations

### Priority 1: Cloud Environment (vast.ai / RunPod)
```
/workspace/violence_detection_mvp/models/
```

### Priority 2: Local Development (Fallback)
```
/home/admin/Desktop/NexaraVision/violence_detection_mvp/models/
```

## 🤖 Auto-Detection

The script automatically:
1. **Tries** `/workspace/` first (for cloud training)
2. **Falls back** to project directory if `/workspace/` not accessible
3. **Creates** the directory if it doesn't exist
4. **Logs** the actual location used: `💾 Models will be saved to: /path/`

## 📦 Files Saved

### In `models/` directory:
```
violence_detection_mvp/models/
├── best_model.h5                    ← Best model (highest val_accuracy)
├── checkpoint_epoch_001.h5          ← Epoch 1 checkpoint
├── checkpoint_epoch_002.h5          ← Epoch 2 checkpoint
├── checkpoint_epoch_003.h5          ← Epoch 3 checkpoint
└── ... (one checkpoint per epoch)
```

### In `checkpoints_improved/` directory:
```
checkpoints_improved/
├── training_history.csv             ← Metrics per epoch (CSV format)
├── training_results.json            ← Final test results (JSON)
├── training_config.json             ← Training configuration used
└── tensorboard/                     ← TensorBoard logs
    ├── events.out.tfevents...
    └── ...
```

## 🚀 Usage

### Start Training
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp

python train_rtx5000_dual_IMPROVED.py \
    --dataset-path /path/to/organized_dataset \
    --epochs 100 \
    --batch-size 64
```

### Check Save Location
Look for this line in training output:
```
💾 Models will be saved to: /home/admin/Desktop/NexaraVision/violence_detection_mvp/models
```

### Resume Training
```bash
python train_rtx5000_dual_IMPROVED.py \
    --dataset-path /path/to/organized_dataset \
    --resume models/checkpoint_epoch_025.h5
```

### Load Best Model
```python
from tensorflow import keras

model = keras.models.load_model(
    'violence_detection_mvp/models/best_model.h5'
)
```

## 📊 Model Information

### Best Model (`best_model.h5`)
- **Auto-saved when:** Validation accuracy improves
- **Monitor metric:** `val_accuracy`
- **Mode:** Maximize accuracy
- **File size:** ~45-50 MB

### Checkpoints (`checkpoint_epoch_XXX.h5`)
- **Saved:** Every epoch
- **Purpose:** Resume training if interrupted
- **Total size:** ~5 GB for 100 epochs

## 🎯 Early Stopping Behavior

With patience=10, early stopping will:
```
Epoch 45: val_acc=0.9350  ✅ BEST (saved to best_model.h5)
Epoch 46: val_acc=0.9340  ❌ No improvement
Epoch 47: val_acc=0.9345  ❌ Still below best
...
Epoch 55: val_acc=0.9330  ❌ Patience exhausted (10 epochs)

🛑 EARLY STOPPING TRIGGERED!
✅ Auto-restored weights from Epoch 45
✅ best_model.h5 contains the best model (val_acc=0.9350)
```

## 📥 Accessing Models

### Local Machine
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp/models
ls -lh
```

### Cloud Instance (vast.ai)
```bash
cd /workspace/violence_detection_mvp/models
ls -lh
```

### Download from Cloud
```bash
# From local machine
scp user@cloud-ip:/workspace/violence_detection_mvp/models/best_model.h5 ./local/models/
```

## 🔧 Troubleshooting

### "Models not found after training"
**Check training logs:**
```bash
grep "Models will be saved" checkpoints_improved/training_history.csv
```

**Or check last training run:**
```bash
cat checkpoints_improved/training_results.json | grep -A 5 "model"
```

### "Permission denied"
**Fix permissions:**
```bash
chmod 755 violence_detection_mvp/models/
```

**Or create manually:**
```bash
mkdir -p violence_detection_mvp/models/
```

### "Disk full" during training
**Each checkpoint = 45-50 MB**
**100 epochs = 5 GB**

**Solution 1:** Reduce checkpoint frequency (modify script)
**Solution 2:** Delete old checkpoints
```bash
# Keep only last 10 checkpoints
cd violence_detection_mvp/models/
ls -t checkpoint_epoch_*.h5 | tail -n +11 | xargs rm
```

## ✅ Verification

After training starts, verify models are being saved:

```bash
# Watch for new model files
watch -n 60 ls -lth violence_detection_mvp/models/

# Check best model exists
ls -lh violence_detection_mvp/models/best_model.h5

# Count checkpoints
ls -1 violence_detection_mvp/models/checkpoint_epoch_*.h5 | wc -l
```

## 🎓 Best Practices

### 1. Backup Important Models
```bash
# After training completes
cp violence_detection_mvp/models/best_model.h5 \
   violence_detection_mvp/models/best_model_backup_$(date +%Y%m%d).h5
```

### 2. Version Your Models
```bash
# Rename after training
mv violence_detection_mvp/models/best_model.h5 \
   violence_detection_mvp/models/best_model_30k_93pct_v1.h5
```

### 3. Keep Training Logs
```bash
# Archive results
tar -czf training_run_$(date +%Y%m%d).tar.gz \
    checkpoints_improved/ \
    violence_detection_mvp/models/best_model.h5
```

## 📖 Summary

| File Type | Location | Purpose |
|-----------|----------|---------|
| **Best Model** | `violence_detection_mvp/models/best_model.h5` | Production model (highest accuracy) |
| **Checkpoints** | `violence_detection_mvp/models/checkpoint_epoch_*.h5` | Resume training |
| **Training Logs** | `checkpoints_improved/training_history.csv` | Metrics per epoch |
| **Final Results** | `checkpoints_improved/training_results.json` | Test set performance |
| **TensorBoard** | `checkpoints_improved/tensorboard/` | Visualize training |

**All model files are now saved to:** `violence_detection_mvp/models/` ✅

Ready to train! 🚀
