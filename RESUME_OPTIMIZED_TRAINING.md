# Resume Optimized Training on New Vast.ai Machine

## 🎯 Your Exact Setup

**Training Script**: `train_model_optimized.py` (Nov 14, 21:56)
**Model**: ResNet50V2 + Bidirectional GRU
**Data Format**: **PRE-EXTRACTED .npy frames** (10x faster!)
**GPU Config**: Single GPU mode (GPU 1, with GPU 0 occupied)

---

## 📋 Critical Files to Transfer

From old vast.ai machine to new machine:

### 1. **Pre-Extracted Frames** (MOST IMPORTANT - 50GB+)
```
/workspace/processed/frames/
├── 0.npy          # Pre-extracted frames from video 0
├── 1.npy
├── 2.npy
...
└── 10737.npy     # ~10,738 .npy files total
```
**Size**: ~50GB
**Critical**: Without these, you'll need to re-extract frames (takes hours!)

### 2. **Splits Configuration**
```
/workspace/processed/splits.json    # Train/val/test split info
```

### 3. **Training Configuration**
```
/workspace/training_config.json     # Batch size, epochs, etc.
```

### 4. **Checkpoints** (if exists)
```
/workspace/models/checkpoints/
├── initial_best_model.keras
└── finetuning_best_model.keras
```

### 5. **Logs** (optional, for resume)
```
/workspace/logs/training/
├── initial_training_log.csv
└── finetuning_training_log.csv
```

---

## 🔍 Step 1: Verify Transfer Completed

SSH into your new vast.ai instance:

```bash
ssh root@YOUR_NEW_VASTAI_IP
```

### Check All Critical Files:

```bash
# 1. Check pre-extracted frames
echo "📂 Checking pre-extracted frames..."
ls -lh /workspace/processed/frames/ | head -5
find /workspace/processed/frames/ -name "*.npy" | wc -l
# Should show: ~10,738 files

# 2. Check splits file
echo "📋 Checking splits.json..."
cat /workspace/processed/splits.json | head -20

# 3. Check training config
echo "⚙️ Checking training_config.json..."
cat /workspace/training_config.json

# 4. Check for existing checkpoints
echo "💾 Checking checkpoints..."
ls -lh /workspace/models/checkpoints/ 2>/dev/null || echo "No checkpoints found"

# 5. Check training logs
echo "📊 Checking training logs..."
ls -lh /workspace/logs/training/*.csv 2>/dev/null || echo "No logs found"
```

---

## 🚀 Step 2: Resume or Start Training

### Option A: Starting Fresh (No Checkpoints)

```bash
cd /workspace

# Verify Python scripts are present
ls -lh train_model_optimized.py
ls -lh model_architecture.py
ls -lh data_preprocessing.py

# Start training
python3 train_model_optimized.py
```

**Expected Output**:
```
================================================================================
NexaraVision OPTIMIZED Training Pipeline
================================================================================
🎮 GPU Configuration
================================================================================
Detected 2 GPU(s):
  GPU 0: /physical_device:GPU:0
  GPU 1: /physical_device:GPU:1
✅ Memory growth enabled (dynamic allocation)
✅ Using GPU 0 only (single-GPU mode)

================================================================================
STEP 1: Loading Pre-Extracted Data
================================================================================

Train Set: 7,516 videos
  Violence: 3,758
  Non-Violence: 3,758

Val Set: 1,611 videos
  Violence: 805
  Non-Violence: 806

Test Set: 1,611 videos
  Violence: 803
  Non-Violence: 808

✅ Data loaded (using pre-extracted frames)
================================================================================

STEP 2: Model Building
================================================================================
...
```

### Option B: Resuming from Checkpoint

If you have checkpoints, modify `train_model_optimized.py`:

```python
# Around line 200-210, in build_model():

def build_model(self):
    """Build and compile model"""

    print("\n" + "=" * 80)
    print("STEP 2: Model Building")
    print("=" * 80)

    # Check for existing checkpoint
    checkpoint_path = f"{self.config['paths']['models']}/checkpoints/finetuning_best_model.keras"

    if os.path.exists(checkpoint_path):
        print(f"\n🔄 LOADING CHECKPOINT: {checkpoint_path}")
        self.model = tf.keras.models.load_model(checkpoint_path)
        print("✅ Checkpoint loaded successfully!")
        self.model_builder = ViolenceDetectionModel()
        self.model_builder.model = self.model
        return self.model

    # Otherwise build new model...
    self.model_builder = ViolenceDetectionModel(
        frames_per_video=self.config['training']['frames_per_video'],
        ...
    )
```

Then run:
```bash
python3 train_model_optimized.py
```

---

## 📊 Step 3: Monitor Training

### Terminal Output

You'll see:
```
Epoch 1/30
236/236 [==============================] - 145s 612ms/step
  loss: 0.3921
  accuracy: 0.8234
  precision: 0.8312
  recall: 0.8156
  val_loss: 0.3456
  val_accuracy: 0.8567
  val_precision: 0.8623
  val_recall: 0.8512
```

### GPU Monitoring

In a separate terminal:
```bash
watch -n 1 nvidia-smi
```

**Expected**:
```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  RTX 3090 Ti       On   | 00000000:01:00.0 Off |                  N/A |
| 45%   65C    P2   280W / 450W |  18234MiB / 24564MiB |     98%      Default |
```

- **GPU Utilization**: Should be 90-100%
- **Memory Usage**: 15-20GB (out of 24GB)
- **Temperature**: Below 80°C

### TensorBoard (Optional)

```bash
# In separate SSH session:
tensorboard --logdir /workspace/logs/training --host 0.0.0.0 --port 6006

# Access from browser:
# http://YOUR_VASTAI_IP:6006
```

### Check Logs in Real-Time

```bash
# Watch training progress
tail -f /workspace/logs/training/initial_training_log.csv

# Check last 10 epochs
tail -10 /workspace/logs/training/initial_training_log.csv
```

---

## ⏱️ Training Timeline

| Phase | Epochs | Duration | What's Happening |
|-------|--------|----------|------------------|
| **Initial Training** | 1-30 | 2-3 hours | ResNet50V2 backbone frozen, training GRU + classifier |
| **Fine-Tuning** | 31-50 | 2-3 hours | Entire model unfrozen, fine-tuning all layers |
| **Total** | 50 | **4-6 hours** | Complete training cycle |

**Speed**: ~145-200 seconds per epoch (with pre-extracted frames)

---

## 💾 What Gets Saved

```
/workspace/
├── models/
│   ├── checkpoints/
│   │   ├── initial_best_model.keras       # Best validation accuracy (Phase 1)
│   │   └── finetuning_best_model.keras    # Best validation accuracy (Phase 2)
│   ├── saved_models/
│   │   └── final_model.keras               # Final trained model
│   └── architecture_config.json            # Model architecture config
├── logs/
│   ├── training/
│   │   ├── initial_training_log.csv        # Epoch-by-epoch metrics (Phase 1)
│   │   ├── finetuning_training_log.csv     # Epoch-by-epoch metrics (Phase 2)
│   │   └── initial_YYYYMMDD_HHMMSS/        # TensorBoard logs
│   └── evaluation/
│       └── test_results.json                # Final test performance
└── processed/
    ├── frames/                              # Pre-extracted .npy files
    └── splits.json                          # Train/val/test splits
```

---

## 🆘 Troubleshooting

### Problem: "No such file: /workspace/processed/frames/0.npy"

**Cause**: Pre-extracted frames weren't transferred

**Solution**:
```bash
# Check if frames exist
ls -lh /workspace/processed/frames/ | head -5

# If missing, you need to transfer them from old machine
# OR re-extract frames (takes 2-3 hours)
```

### Problem: "No such file: /workspace/processed/splits.json"

**Cause**: Splits file wasn't transferred

**Solution**:
```bash
# Check if splits.json exists
cat /workspace/processed/splits.json

# If missing, transfer from old machine
# OR regenerate splits (requires running data preprocessing)
```

### Problem: Out of Memory (OOM)

**Solution**: Edit `/workspace/training_config.json`:
```json
{
  "training": {
    "batch_size": 16  // Change from 32 to 16 or 8
  }
}
```

### Problem: Training Very Slow (<100% GPU utilization)

**Solution**: Increase batch size:
```json
{
  "training": {
    "batch_size": 48  // Increase if you have VRAM available
  }
}
```

### Problem: "CUDA_ERROR_OUT_OF_MEMORY"

**Temporary Fix**:
```bash
# Edit train_model_optimized.py line 13:
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Try different GPU

# OR reduce batch size in training_config.json
```

---

## ⚡ Quick Commands Reference

```bash
# Start training
python3 /workspace/train_model_optimized.py

# Monitor GPU
watch -n 1 nvidia-smi

# Watch training progress
tail -f /workspace/logs/training/initial_training_log.csv

# Check latest checkpoint
ls -lht /workspace/models/checkpoints/ | head -3

# Count pre-extracted frames
find /workspace/processed/frames/ -name "*.npy" | wc -l

# Check current epoch
tail -1 /workspace/logs/training/initial_training_log.csv

# Stop training gracefully (Ctrl+C once - saves checkpoint)

# Check disk space
df -h /workspace
```

---

## 📥 After Training: Download Model

```bash
# From your local machine:
scp root@YOUR_VASTAI_IP:/workspace/models/saved_models/final_model.keras ./

# Or use vast.ai web interface file browser
```

---

## 🎤 Tomorrow's Pitch - What to Show

### If Training is Complete (Best Case):
```
✅ "We trained ResNet50V2 + BiGRU on 10,738 videos"
✅ "Final accuracy: 90-93%"
✅ "Model ready for deployment"
✅ "Next: Upgrade to CrimeNet ViT for 95-99% accuracy"
```

### If Training is In Progress (Likely Case):
```
✅ "Training ResNet50V2 + BiGRU on 10,738 videos"
✅ "Currently at epoch X/50 with Y% accuracy"
✅ "Expected final accuracy: 90-93%"
✅ "Using optimized pipeline with pre-extracted frames (10x faster)"
✅ "Roadmap: CrimeNet ViT upgrade for 95-99% accuracy"
```

### Show This:
1. **Terminal Output**: Live training progress
2. **TensorBoard**: Real-time accuracy/loss graphs
3. **GPU Utilization**: `nvidia-smi` showing 100% usage
4. **Architecture**: Explain ResNet50V2 (spatial) + BiGRU (temporal)

---

## ✅ Pre-Pitch Checklist

- [ ] Training running successfully
- [ ] GPU utilization >80%
- [ ] Logs updating in `/workspace/logs/training/`
- [ ] Current epoch and accuracy known
- [ ] TensorBoard accessible (optional)
- [ ] Can explain architecture (ResNet50V2 + BiGRU)
- [ ] Know upgrade path (CrimeNet ViT → 95-99%)

---

## 🚀 Ready to Start

**Run This Now**:
```bash
# SSH into new vast.ai machine
ssh root@YOUR_VASTAI_IP

# Verify data
find /workspace/processed/frames/ -name "*.npy" | wc -l

# Start training
cd /workspace
python3 train_model_optimized.py
```

**Expected Training Time**: 4-6 hours (much faster with pre-extracted frames!)

Good luck with training and tomorrow's pitch! 🎯
