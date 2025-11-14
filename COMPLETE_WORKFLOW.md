# Complete Violence Detection Dataset Workflow

**Instance**: 2× NVIDIA RTX 5000 Ada Generation (64GB VRAM total)

---

## 📊 Your Current Dataset Status

| Dataset | Location | Videos | Category | Status |
|---------|----------|--------|----------|--------|
| phase1 | `/workspace/datasets/phase1` | 15,994 | **MIXED** | ⚠️ Needs Separation |
| youtube_fights | `/workspace/datasets/youtube_fights` | 105 | Violent | ✅ Ready |
| reddit_videos_massive | `/workspace/datasets/reddit_videos_massive` | 2,667 | Violent | ✅ Ready |
| reddit_videos | `/workspace/datasets/reddit_videos` | 1,669 | Violent | ✅ Ready |
| nonviolent_safe | `/workspace/datasets/nonviolent_safe` | 30 | Non-Violent | ✅ Ready |

**Total**: ~20,000 videos

---

## 🎯 Complete Workflow (5 Steps)

### Step 1: Analyze Phase1 (5 minutes)

Phase1 is 79% of your data and needs separation first.

```bash
cd /workspace
python3 analyze_phase1_optimized.py > phase1_analysis.txt
cat phase1_analysis.txt
```

**What this shows**:
- Uses path structure: `/Violence/`, `/NonViolence/`, `/Fight/`, `/Fighting/`
- Uses filename patterns: `_Fighting`, `_Normal`, `_Shooting`
- Expected: 95%+ confident categorization
- Shows: ~12,500 violent + ~4,600 non-violent

**Key checks**:
```bash
# Verify RoadAccidents are VIOLENT
grep -A 3 "RoadAccidents" phase1_analysis.txt

# Check confidence levels
grep "Path structure" phase1_analysis.txt
```

---

### Step 2: Separate Phase1 (30-60 minutes)

```bash
python3 separate_phase1_optimized.py
```

**When prompted, choose**:
- Mode 1 (all_confident) - Uses Path + Filename + Keywords (85-99% accuracy) ⭐ **RECOMMENDED**

**Output location**:
- `/workspace/datasets/phase1_categorized/violent/`
- `/workspace/datasets/phase1_categorized/nonviolent/`

**Verify**:
```bash
find /workspace/datasets/phase1_categorized/violent/ -name "*.mp4" -o -name "*.avi" | wc -l
find /workspace/datasets/phase1_categorized/nonviolent/ -name "*.mp4" -o -name "*.avi" | wc -l
```

---

### Step 3: Combine ALL Datasets (20-30 minutes)

Now combine phase1 + youtube + reddit + nonviolent into single violent/nonviolent folders.

```bash
python3 combine_all_datasets.py
```

**What this does**:
1. Checks if phase1 is separated ✅
2. Finds all videos in:
   - phase1_categorized/violent → violent
   - phase1_categorized/nonviolent → nonviolent
   - youtube_fights → violent
   - reddit_videos_massive → violent
   - reddit_videos → violent
   - nonviolent_safe → nonviolent
3. Combines into: `/workspace/datasets/combined_dataset/violent` and `/nonviolent`

**When prompted**:
- Mode 1 (copy) - Preserves originals ⭐ **RECOMMENDED**
- Mode 2 (move) - Saves space, but empties source folders

**Expected output**:
```
⚠️  Total Violent:     ~16,500 videos
✅ Total Non-Violent: ~4,600 videos
⚖️  CLASS BALANCE: 22% (SEVERE IMBALANCE!)

⚠️  Recommend collecting ~3,600 more non-violent videos
⚠️  For 50/50 balance, need: 8,250 non-violent total
```

---

### Step 4: Balance Dataset (Optional but Recommended)

You have two options:

**Option A: Downsample Violent** (Quick)
- Keep 4,600 violent + 4,600 non-violent = 9,200 total
- Smaller dataset, but balanced
- Good for initial training

**Option B: Collect More Non-Violent** (Better)
- Need ~3,600 more non-violent videos
- Larger balanced dataset = better accuracy
- Recommended sources:
  - UCF-Crime Normal videos
  - ActivityNet daily activities
  - CCTV footage datasets
  - YouTube daily life videos

For now, let's proceed with current imbalanced data or downsample.

---

### Step 5a: Create Train/Val/Test Splits (5 minutes)

You need to update `analyze_and_split_dataset.py` to point to the combined dataset:

```python
# Edit analyze_and_split_dataset.py
# Change DATASET_PATH to:
DATASET_PATH = "/workspace/datasets/combined_dataset"
```

Then run:
```bash
python3 analyze_and_split_dataset.py
```

**This creates**:
- `/workspace/organized_dataset/train/` (70%)
- `/workspace/organized_dataset/val/` (15%)
- `/workspace/organized_dataset/test/` (15%)

With text files:
- `train_videos.txt` (path,label format)
- `val_videos.txt`
- `test_videos.txt`

---

### Step 5b: Train Model! (4-6 hours)

```bash
python3 train_dual_rtx5000.py
```

**Training Configuration**:
- **Model**: VGG19 + Bi-LSTM + Multi-Head Attention
- **Batch size**: 32 per GPU = 64 total
- **Gradient accumulation**: 4 steps → Effective batch = 256
- **Mixed precision**: FP16 (2× faster)
- **Multi-GPU**: DataParallel on 2× RTX 5000 Ada
- **Epochs**: 50 (with early stopping)
- **Expected accuracy**: 90-96%

**Monitor progress**:
```bash
# In another terminal
watch -n 5 nvidia-smi

# Check checkpoint
ls -lh /workspace/checkpoints/
```

**Output**:
- Best model: `/workspace/checkpoints/best_model.pth`
- Logs: `/workspace/logs/`

---

## 📋 Quick Command Reference

```bash
# === COMPLETE WORKFLOW ===

# 1. Analyze phase1
cd /workspace
python3 analyze_phase1_optimized.py

# 2. Separate phase1
python3 separate_phase1_optimized.py
# Choose: Mode 1

# 3. Combine all datasets
python3 combine_all_datasets.py
# Choose: Mode 1 (copy)

# 4. Create splits
python3 analyze_and_split_dataset.py

# 5. Train!
python3 train_dual_rtx5000.py
```

---

## ⚠️ Disk Space Requirements

| Operation | Space Needed | Notes |
|-----------|--------------|-------|
| phase1 original | 410 GB | Already have |
| phase1 separated | 410 GB | Copy mode |
| Combined dataset | 480 GB | All datasets combined |
| **Total** | **1,300 GB** | With copy mode |

**If space limited**: Use move mode in combine_all_datasets.py (saves 820 GB)

---

## 🎯 Expected Results

### After Separation:
- Violent: ~12,500 (phase1) + 105 (youtube) + 2,667 (reddit_massive) + 1,669 (reddit) = **~16,500**
- Non-Violent: ~4,600 (phase1) + 30 (nonviolent_safe) = **~4,600**
- **Imbalance**: 78% violent / 22% non-violent

### After Training (Current Data):
- **Accuracy**: 88-92% (imbalanced data)
- **F1-Score**: 0.85-0.90
- **False Positive Rate**: 8-12%

### After Balancing + Training:
- **Accuracy**: 90-96% (balanced data)
- **F1-Score**: 0.90-0.95
- **False Positive Rate**: 4-8%

---

## 🔧 Troubleshooting

### "Out of disk space"
```bash
# Check space
df -h /workspace

# Solution 1: Use move mode
python3 combine_all_datasets.py
# Choose mode 2 (move)

# Solution 2: Delete phase1 after combination
rm -rf /workspace/datasets/phase1
rm -rf /workspace/datasets/phase1_categorized
```

### "Out of memory during training"
```python
# Edit train_dual_rtx5000.py
# Reduce batch size:
'batch_size': 16,  # Instead of 32
'accumulation_steps': 8,  # Instead of 4
# Effective batch stays 256
```

### "Training very slow"
```bash
# Check GPU usage
nvidia-smi

# Make sure both GPUs are being used
# Should show ~90%+ utilization on both GPUs
```

---

## 📈 Next Steps After Training

1. **Evaluate on test set**:
   ```bash
   python3 evaluate_model.py --model checkpoints/best_model.pth
   ```

2. **Collect more non-violent data**:
   - Target: 8,250 non-violent videos (50/50 balance)
   - Sources: UCF-Crime, ActivityNet, CCTV datasets

3. **Retrain with balanced data**:
   - Expected accuracy: 90-96%
   - Better generalization

4. **Deploy model**:
   - Export to ONNX for inference
   - Real-time video analysis
   - Production deployment

---

## 📞 Support

If you encounter issues at any step, paste the error and I'll help debug!

**Current Status**: Ready to start Step 1 (Analyze Phase1) 🚀
