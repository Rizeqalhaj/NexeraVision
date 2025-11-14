# 🎯 Vast.ai Dataset Organization Guide

## What You Have

Based on your screenshot, you have multiple dataset folders on Vast.ai:
- ✅ **reddit_videos_massive** (36 GB) - Multiple fight subreddits
- ✅ **youtube_fights** (14 GB) - UFC, MMA, boxing, street fights
- ❓ **Other folders** - cctv_surveillance, nonviolent, phase1, phase3, etc.

## 🚀 Step-by-Step Organization

### Step 1: Quick Summary (2 minutes)

In your Vast.ai terminal, run:

```bash
cd /workspace/datasets
bash quick_dataset_summary.sh
```

This will show you:
- All folders with video counts
- Categories (Violent/Non-Violent/Mixed)
- Size of each folder
- Estimated train/val/test splits

**Sample output:**
```
Folder                                   | Videos |   Size | Category
---------|--------|------|----------
reddit_videos_massive/                   |  1,234 |   36G | 🔀 MIXED
youtube_fights/                          |    156 |   14G | ⚠️  VIOLENT
nonviolent/                              |    800 |   10G | ✅ NON-VIOLENT
...
```

### Step 2: Full Analysis & Organization (30-60 minutes)

Once you've reviewed the summary, organize everything:

```bash
cd /workspace
python3 analyze_and_split_dataset.py
```

This will:
1. ✅ Scan all folders
2. ✅ Categorize as Violent/Non-Violent
3. ✅ Show detailed statistics
4. ✅ Ask for confirmation
5. ✅ Create train/val/test splits (70/15/15)
6. ✅ Copy videos to organized structure

**Output structure:**
```
/workspace/organized_dataset/
├── train/
│   ├── Violence/      (70% of violent videos)
│   └── NonViolence/   (70% of non-violent videos)
├── val/
│   ├── Violence/      (15% of violent videos)
│   └── NonViolence/   (15% of non-violent videos)
├── test/
│   ├── Violence/      (15% of violent videos)
│   └── NonViolence/   (15% of non-violent videos)
└── dataset_info.json  (metadata)
```

---

## 🎯 Expected Results

Based on your folders, you likely have:

### Violent Videos (~1,500-2,000)
- reddit_videos_massive/r_fights/
- reddit_videos_massive/r_StreetFighting/
- reddit_videos_massive/r_RealFights/
- reddit_videos_massive/r_BrutalFights/
- reddit_videos_massive/r_femalemma/
- reddit_videos_massive/r_fightclub/
- youtube_fights/ (all subdirectories)

### Non-Violent Videos (need more!)
- cctv_surveillance/
- nonviolent/
- nonviolent_kaggle/
- nonviolent_safe/

### Mixed (will be excluded)
- reddit_videos_massive/r_instantkarma/
- reddit_videos_massive/r_instant_regret/
- reddit_videos_massive/r_ActualPublicFreakouts/

---

## ⚠️ Important Notes

### 1. Class Balance

For best training results, you need **equal numbers** of violent and non-violent videos.

**If imbalanced:**
```bash
# Option A: Downsample majority class
# Script will use only N videos from each class (where N = minority class size)

# Option B: Use class weights in training
# The training script can handle imbalance automatically
```

### 2. Disk Space

The organization script **copies** videos (doesn't move them).

**Space needed:**
- Original videos: 50 GB
- Organized dataset: 50 GB
- **Total: 100 GB**

If space is limited:
```bash
# Edit analyze_and_split_dataset.py
# Change shutil.copy2() to shutil.move()
# This moves instead of copying (saves 50 GB)
```

### 3. Mixed Categories

Some folders contain both violent and non-violent content:
- `r_instantkarma` - Some fights, some accidents
- `r_ActualPublicFreakouts` - Some fights, mostly yelling

**These will be excluded by default.** You can manually review later.

---

## 🔧 Customization

### Change Split Ratios

Edit `analyze_and_split_dataset.py`:

```python
# Default: 70/15/15
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Alternative: 80/10/10
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
```

### Change Output Location

```python
OUTPUT_DIR = "/workspace/organized_dataset"
# Change to wherever you want
```

### Balance Classes

Add after line with `all_videos`:

```python
# Balance classes by downsampling majority
violent_count = len(all_videos['violent'])
nonviolent_count = len(all_videos['nonviolent'])
min_count = min(violent_count, nonviolent_count)

if violent_count != nonviolent_count:
    print(f"\n⚖️  Balancing classes to {min_count} each...")
    all_videos['violent'] = random.sample(all_videos['violent'], min_count)
    all_videos['nonviolent'] = random.sample(all_videos['nonviolent'], min_count)
```

---

## 📊 Expected Training Results

With properly organized data:

| Dataset Size | Expected Accuracy | Training Time (L40S) |
|--------------|------------------|---------------------|
| 500-1,000 videos | 85-90% | 30-45 min |
| 1,000-2,000 videos | 90-94% | 1-1.5 hours |
| 2,000-4,000 videos | 93-96% | 1.5-2.5 hours |
| 4,000+ videos | 95-98% | 2-3 hours |

---

## 🚀 Quick Start Commands

### On Vast.ai (Copy-Paste Ready)

```bash
# 1. Navigate to datasets folder
cd /workspace/datasets

# 2. Quick summary (see what you have)
bash quick_dataset_summary.sh

# 3. Full organization (creates train/val/test)
cd /workspace
python3 analyze_and_split_dataset.py

# 4. Verify organized dataset
ls -la /workspace/organized_dataset/train/*/

# 5. Start training!
cd /workspace/NexaraVision/violence_detection_mvp
python runpod_train_l40s.py --dataset-path /workspace/organized_dataset
```

---

## 🔍 Troubleshooting

### "Directory not found"
```bash
# Find your datasets location
find / -name "reddit_videos_massive" -type d 2>/dev/null

# Update script with correct path
nano analyze_and_split_dataset.py
# Change BASE_DIR to your actual path
```

### "Permission denied"
```bash
chmod +x quick_dataset_summary.sh
chmod +x analyze_and_split_dataset.py
```

### "Not enough space"
```bash
# Check available space
df -h /workspace

# If low, change script to move instead of copy
# Or delete original files after organizing
```

### Videos not detected
```bash
# Check what extensions you have
find /workspace/datasets -type f -name "*.*" | sed 's/.*\.//' | sort | uniq -c

# Add to VIDEO_EXTENSIONS in script if needed
```

---

## 💡 Pro Tips

### 1. Test First
```bash
# Create small test with just 100 videos
# Edit script: limit to 100 per category
# Verify structure before full run
```

### 2. Save Disk Space
```bash
# After organizing, compress originals
tar -czf datasets_backup.tar.gz /workspace/datasets
# Then delete originals
rm -rf /workspace/datasets
```

### 3. Verify Quality
```bash
# After organization, sample some videos
ls /workspace/organized_dataset/train/Violence/ | shuf -n 5
# Watch a few to verify they're correctly categorized
```

### 4. Monitor Progress
```bash
# In another terminal, watch progress
watch -n 5 "du -sh /workspace/organized_dataset/*"
```

---

## ✅ Checklist

Before training:
- [ ] Run `quick_dataset_summary.sh` - reviewed categories
- [ ] Run `analyze_and_split_dataset.py` - created splits
- [ ] Verified train/val/test folders exist
- [ ] Checked class balance (should be close to 50/50)
- [ ] Confirmed total video counts match expectations
- [ ] Ready to start training!

---

**Ready to organize your dataset?** Run the commands above! 🚀
