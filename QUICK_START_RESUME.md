# Quick Start: Resume Training on New Machine

## 🚀 Upload to Vast.ai

Upload these 3 files to `/workspace/`:
```
train_model_RESUME.py       ← New resume-capable script
model_architecture.py
data_preprocessing.py
```

---

## ✅ One Command to Resume Training

```bash
cd /workspace
python3 train_model_RESUME.py
```

**That's it!** The script automatically:
- ✅ Detects existing checkpoints
- ✅ Loads last saved model
- ✅ Resumes from last epoch
- ✅ Continues training seamlessly

---

## 📊 What Happens

### Scenario 1: No Checkpoint (Fresh Start)
```
🆕 No checkpoint found - starting fresh training

STEP 3: Initial Training
Epochs: 30
Batch Size: 32

🚀 Starting training...
Epoch 1/30
```

### Scenario 2: Has Checkpoint (Resume)
```
🔄 CHECKPOINT FOUND!
   Path: /workspace/models/checkpoints/initial_best_model.keras
   Phase: initial

✅ Checkpoint loaded successfully!

📊 Training Progress:
   Completed Epochs: 15
   Last Train Accuracy: 86.70%
   Last Val Accuracy: 84.90%

✅ Ready to resume training!

🚀 Resuming from epoch 15/30...
Epoch 16/30
```

---

## 🎯 Key Features

### Automatic Checkpoint Detection
- Looks for `finetuning_best_model.keras` first (most recent)
- Falls back to `initial_best_model.keras`
- Reads training logs to determine last epoch

### Smart Resume Logic
- **Epoch 0-29**: Continues initial training
- **Epoch 30**: Automatically switches to fine-tuning
- **Epoch 31-50**: Continues fine-tuning
- **Epoch 50+**: Training complete, runs evaluation

### Safe Interruption
Press `Ctrl+C` once → checkpoint saves → resume anytime!

---

## 📁 Required Files on Server

```
/workspace/
├── train_model_RESUME.py          ← Upload this
├── model_architecture.py           ← Upload this
├── data_preprocessing.py           ← Upload this
├── training_config.json            ← Must exist
├── processed/
│   ├── splits.json                 ← Must exist
│   └── frames/                     ← Must exist (10,738 .npy files)
│       ├── 0.npy
│       ├── 1.npy
│       └── ...
└── models/
    └── checkpoints/                 ← Auto-created, or contains checkpoints
        ├── initial_best_model.keras     (optional - will resume if exists)
        └── finetuning_best_model.keras  (optional - will resume if exists)
```

---

## ⚙️ Configuration

Edit GPU in `train_model_RESUME.py` line 14:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Change to '0,1' for dual GPU
```

For **2x RTX 6000 Ada**, you can use both:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both GPUs
```

---

## 📊 Monitor Progress

```bash
# Watch training logs in real-time
tail -f /workspace/logs/training/initial_training_log.csv

# Check GPU usage
watch -n 1 nvidia-smi

# See latest checkpoint
ls -lht /workspace/models/checkpoints/ | head -3

# Check current epoch
tail -1 /workspace/logs/training/initial_training_log.csv
```

---

## 🔄 Resume Examples

### Example 1: Stopped at Epoch 10
```
Previous run: Epoch 1-10 (stopped)
Resume: Automatically continues from Epoch 11-30
```

### Example 2: Stopped at Epoch 35 (Fine-tuning)
```
Previous run: Epoch 1-35 (stopped during fine-tuning)
Resume: Automatically continues from Epoch 36-50
```

### Example 3: Training Complete
```
Previous run: Epoch 1-50 (complete)
Resume: Skips training, runs evaluation only
```

---

## ⏱️ Expected Timeline

| Status | Remaining Epochs | Time Left |
|--------|------------------|-----------|
| **Fresh Start (Epoch 0)** | 50 epochs | 4-6 hours |
| **Stopped at Epoch 10** | 40 epochs | 3-5 hours |
| **Stopped at Epoch 25** | 25 epochs | 2-3 hours |
| **Stopped at Epoch 40** | 10 epochs | 1 hour |
| **Stopped at Epoch 48** | 2 epochs | 15 minutes |

---

## ✅ Verification Commands

Before starting, verify everything is ready:

```bash
# 1. Check pre-extracted frames
find /workspace/processed/frames/ -name "*.npy" | wc -l
# Should show: 10738

# 2. Check splits file
cat /workspace/processed/splits.json | head -5

# 3. Check config
cat /workspace/training_config.json

# 4. Check for checkpoints (optional)
ls -lh /workspace/models/checkpoints/ 2>/dev/null || echo "No checkpoints (fresh start)"

# 5. Check Python script
ls -lh /workspace/train_model_RESUME.py
```

---

## 🎤 For Tomorrow's Pitch

### Show This Terminal Output:
```
🔄 CHECKPOINT FOUND!
   Path: /workspace/models/checkpoints/initial_best_model.keras
   Phase: initial

✅ Checkpoint loaded successfully!

📊 Training Progress:
   Completed Epochs: 25
   Last Train Accuracy: 90.80%
   Last Val Accuracy: 89.50%

✅ Ready to resume training!

🚀 Resuming from epoch 25/30...
Epoch 26/30
236/236 [==============================] - 145s 612ms/step
  loss: 0.2456
  accuracy: 0.9121
  val_accuracy: 0.8978
```

**Talk Track**:
"Our ResNet50V2 + BiGRU model is currently at epoch 25 with 91% training accuracy. Expected final: 93%."

---

## 🚨 Troubleshooting

### "FileNotFoundError: /workspace/processed/splits.json"
**Fix**: Transfer `splits.json` from old machine

### "FileNotFoundError: /workspace/processed/frames/0.npy"
**Fix**: Transfer all `.npy` files from old machine (this is ~50GB)

### "ModuleNotFoundError: No module named 'model_architecture'"
**Fix**: Upload `model_architecture.py` to `/workspace/`

### Out of Memory
**Fix**: Edit `/workspace/training_config.json`:
```json
{"training": {"batch_size": 16}}
```

---

**Ready to train?**

```bash
python3 /workspace/train_model_RESUME.py
```

Good luck! 🚀
