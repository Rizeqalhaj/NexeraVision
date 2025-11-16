# QUICK START - Resume Training (1 Minute)

## Upload 3 Files to Vast.ai

```bash
cd /home/admin/Desktop/NexaraVision

scp SIMPLE_RESUME.py \
    model_architecture.py \
    data_preprocessing.py \
    root@195.142.145.66:/workspace/
```

## Run on Vast.ai

```bash
ssh root@195.142.145.66
cd /workspace
python3 SIMPLE_RESUME.py
```

**That's it!**

---

## What It Does

✅ Auto-detects your checkpoint at `/workspace/checkpoints/`
✅ Reads last epoch from CSV log
✅ Resumes exactly where you left off
✅ No config file needed - everything hardcoded
✅ Works with your existing data at `/workspace/datasets/tier1/`
✅ Uses your existing splits at `/workspace/processed/splits.json`

---

## Output You'll See

### If checkpoint found:
```
🔄 CHECKPOINT FOUND!
   Path: /workspace/checkpoints/initial_best_model.keras
   Phase: initial
   Last Epoch: 30

📥 Loading checkpoint...
✅ Checkpoint loaded!

📂 Loading data splits...
   Train: 7516 videos
   Val: 1611 videos

🚀 Resuming initial training from epoch 30/50
Epoch 31/50
[Training continues...]
```

### If no checkpoint:
```
🆕 No checkpoint - starting fresh

📂 Loading data splits...
🔧 Setting up data pipeline...
🚀 Starting initial training...
Epoch 1/50
[Training starts...]
```

---

## Training Sequence

**Phase 1: Initial (epochs 0-50)**
- Frozen ResNet50V2 backbone
- Checkpoint: `/workspace/checkpoints/initial_best_model.keras`
- Log: `/workspace/logs/training/initial_training_log.csv`

**Phase 2: Fine-tuning (epochs 50-100)**
- Unfrozen backbone
- Checkpoint: `/workspace/checkpoints/finetuning_best_model.keras`
- Log: `/workspace/logs/training/finetuning_training_log.csv`

**Final model**: `/workspace/models/final_model.keras`

---

## Monitor Training

```bash
# Open another terminal
ssh root@195.142.145.66

# Watch GPU
watch -n 1 nvidia-smi

# Watch training log
tail -f /workspace/logs/training/initial_training_log.csv
```

---

## If Training Crashes

Just run it again - it will resume automatically:

```bash
python3 SIMPLE_RESUME.py
```

No need to do anything else!

---

**Simple. Clean. Works.** 🚀
