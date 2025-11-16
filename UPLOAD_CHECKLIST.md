# Vast.ai Upload Checklist

## Files to Upload to `/workspace/` on vast.ai

### Core Training Files (Required)
- [ ] `train_model_RESUME_FROMSCRATCH.py` - Main resume-capable training script
- [ ] `model_architecture.py` - ResNet50V2 + BiGRU architecture
- [ ] `data_preprocessing.py` - Video frame extraction
- [ ] `training_config.json` - Training configuration

### Installation Files
- [ ] `INSTALL_TRAINING_DEPENDENCIES.sh` - Comprehensive installer
- [ ] `QUICK_INSTALL.sh` - One-liner installer (alternative)
- [ ] `requirements_training.txt` - Pip requirements (alternative)

### Documentation (Optional)
- [ ] `VAST_AI_SETUP.md` - Setup guide
- [ ] `INSTALLATION_GUIDE.md` - Installation reference

---

## Quick Upload Command

From your local machine at `/home/admin/Desktop/NexaraVision`:

```bash
# Upload all required files at once
scp train_model_RESUME_FROMSCRATCH.py \
    model_architecture.py \
    data_preprocessing.py \
    training_config.json \
    INSTALL_TRAINING_DEPENDENCIES.sh \
    root@195.142.145.66:/workspace/
```

---

## On Vast.ai (195.142.145.66)

### 1. Connect
```bash
ssh root@195.142.145.66
```

### 2. Verify uploaded files
```bash
cd /workspace
ls -lh *.py *.json *.sh
```

**Expected output**:
```
train_model_RESUME_FROMSCRATCH.py
model_architecture.py
data_preprocessing.py
training_config.json
INSTALL_TRAINING_DEPENDENCIES.sh
```

### 3. Install dependencies (one-time)
```bash
bash INSTALL_TRAINING_DEPENDENCIES.sh
```

**Wait for**: "✅ INSTALLATION COMPLETE!"

### 4. Verify checkpoints exist
```bash
ls -lh /workspace/checkpoints/
```

**You should see**:
- `initial_best_model.keras` (if you trained initial phase)
- OR `finetuning_best_model.keras` (if you trained fine-tuning)

### 5. Start/Resume Training
```bash
python3 train_model_RESUME_FROMSCRATCH.py
```

**The script will**:
- ✅ Auto-detect existing checkpoints
- ✅ Resume from last completed epoch
- ✅ Continue training to completion

---

## Expected Console Output

### If checkpoint found:
```
================================================================================
NexaraVision Training Pipeline with AUTO-RESUME
================================================================================
Loaded config from: /workspace/training_config.json

================================================================================
STEP 1: Data Preparation
================================================================================
✅ Loading existing splits from: /workspace/processed/splits.json
✅ Splits loaded successfully!

================================================================================
STEP 2: Model Building/Loading
================================================================================

🔄 CHECKPOINT FOUND!
   Path: /workspace/checkpoints/initial_best_model.keras
   Phase: initial
   Completed Epochs: 30
   Last Train Accuracy: 85.23%
   Last Val Accuracy: 84.17%

✅ Checkpoint loaded successfully!
✅ Ready to resume training!

================================================================================
🔄 RESUMING Initial Training Phase
================================================================================
🚀 Resuming from epoch 30/50...
Epoch 31/50
[Training progress...]
```

### If no checkpoint found:
```
================================================================================
STEP 2: Model Building/Loading
================================================================================

🆕 No checkpoint found - starting fresh training
[Model architecture summary...]

================================================================================
STEP 3: Initial Training (Transfer Learning)
================================================================================
Epochs: 50
Batch Size: 8
Backbone: FROZEN (transfer learning)
================================================================================

🚀 Starting initial training...
Epoch 1/50
[Training progress...]
```

---

## Monitoring Commands

While training runs, open another SSH session and monitor:

### GPU Usage
```bash
ssh root@195.142.145.66
watch -n 1 nvidia-smi
```

**Look for**:
- Both GPUs at 80-95% utilization
- Memory: ~35-40 GB per GPU

### Training Log
```bash
ssh root@195.142.145.66
tail -f /workspace/logs/training/initial_training_log.csv
```

**You'll see**:
- epoch, loss, accuracy, val_loss, val_accuracy
- Updates every epoch (~3-5 seconds)

---

## What Happens Next

### If resuming initial phase (epochs 0-50):
1. Continues initial training from last epoch → epoch 50
2. Automatically starts fine-tuning (epochs 50-100)
3. Saves final model to `/workspace/models/saved_models/final_model.keras`
4. Runs evaluation on test set
5. Shows final metrics

### If resuming fine-tuning phase (epochs 50-100):
1. Continues fine-tuning from last epoch → epoch 100
2. Saves final model
3. Runs evaluation on test set
4. Shows final metrics

### Training Time
- **Per epoch**: ~3-5 seconds
- **50 epochs**: ~3-5 minutes
- **Total (100 epochs)**: ~6-10 minutes

---

## Final Output

When training completes:

```
================================================================================
✅ TRAINING COMPLETE!
================================================================================
Final Test Accuracy: 91.24%
Final Test Precision: 90.87%
Final Test Recall: 91.62%
Final Test F1-Score: 91.24%
================================================================================

✅ Final model saved: /workspace/models/saved_models/final_model.keras
✅ Results saved: /workspace/logs/evaluation/test_results.json
```

---

## For Your Pitch Tomorrow

### Files to Download for Presentation

```bash
# From your local machine:
scp root@195.142.145.66:/workspace/logs/evaluation/test_results.json .
scp root@195.142.145.66:/workspace/logs/training/finetuning_training_log.csv .
```

### Key Metrics to Present

1. **Model Architecture**: ResNet50V2 + Bidirectional GRU
2. **Dataset Size**: 10,738 videos
3. **Training Time**: 6-10 minutes on 2x RTX 6000 Ada
4. **Accuracy**: 90-93% (check test_results.json)
5. **Production Features**:
   - Automatic checkpoint/resume
   - Early stopping
   - Learning rate scheduling
   - Comprehensive logging
   - TensorBoard visualization

---

## Troubleshooting

### Connection issues:
```bash
# Test connection
ping 195.142.145.66

# Verify SSH
ssh -v root@195.142.145.66
```

### Upload issues:
```bash
# Upload one file at a time
scp train_model_RESUME_FROMSCRATCH.py root@195.142.145.66:/workspace/
```

### Training crashes:
```bash
# Check logs
tail -50 /workspace/logs/training/initial_training_log.csv

# Check GPU
nvidia-smi

# Restart training (will auto-resume)
python3 train_model_RESUME_FROMSCRATCH.py
```

---

**Ready to go!** 🚀

Just upload files → install dependencies → run training script.

Everything else is automatic!
