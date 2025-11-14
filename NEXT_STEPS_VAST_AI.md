# 🎯 EXACT NEXT STEPS - Vast.ai Dataset Organization

## Your Current Situation

Based on your summary, you have:

| Folder | Videos | Size | Status |
|--------|--------|------|--------|
| ✅ youtube_fights | 105 | 14G | VIOLENT |
| ✅ nonviolent_safe | 30 | 438M | NON-VIOLENT |
| ❓ **phase1** | **15,994** | **410G** | **UNKNOWN** ⚠️ |
| ❓ reddit_videos_massive | 2,667 | 36G | UNKNOWN (likely violent) |
| ❓ reddit_videos | 1,669 | 20G | UNKNOWN |

**CRITICAL**: phase1 has 15,994 videos (79% of your data!)

---

## 🚨 URGENT: Investigate phase1

### Step 1: What's in phase1? (5 minutes)

Run this on Vast.ai:

```bash
cd /workspace/datasets
bash /workspace/investigate_unknown_folders.sh > investigation_report.txt
cat investigation_report.txt
```

This will show:
- Directory structure of phase1
- Subdirectory names and counts
- Sample video filenames

**Look for clues:**
- Folder names with "fight", "violence", "assault" → VIOLENT
- Folder names with "normal", "daily", "cctv", "activity" → NON-VIOLENT
- Look at sample filenames for patterns

### Step 2: Analyze Reddit Videos (2 minutes)

```bash
cd /workspace/datasets
bash /workspace/analyze_reddit_videos.sh
```

This will categorize reddit_videos_massive by subreddit:
- r_fights, r_StreetFighting, r_femalemma → VIOLENT
- r_instantkarma, r_ActualPublicFreakouts → MIXED
- Shows exact counts per subreddit

---

## 📋 Three Scenarios & Solutions

### Scenario A: phase1 is VIOLENT (fights/violence)

**If phase1 contains fights, MMA, violence:**

```bash
# You have:
# - Violent: 15,994 (phase1) + 105 (youtube) + ~2,000 (reddit) = ~18,000
# - Non-violent: 30
# ⚠️ MASSIVE IMBALANCE!

# SOLUTION: Download more non-violent videos OR downsample violent
```

**Quick fix - Downsample to balance:**
```python
# Edit analyze_and_split_dataset.py
# Add after line 147 (after all_videos is created):

# Balance classes
violent_count = len(all_videos['violent'])
nonviolent_count = len(all_videos['nonviolent'])
min_count = min(violent_count, nonviolent_count)

print(f"\n⚖️  Balancing: Using {min_count} videos per class")
all_videos['violent'] = random.sample(all_videos['violent'], min_count)
all_videos['nonviolent'] = random.sample(all_videos['nonviolent'], min_count)
```

**Result**: 30 violent + 30 non-violent (tiny but balanced)

---

### Scenario B: phase1 is NON-VIOLENT (normal activities)

**If phase1 contains normal activities, CCTV, daily life:**

```bash
# You have:
# - Violent: 105 (youtube) + ~2,000 (reddit) = ~2,100
# - Non-violent: 15,994 (phase1) + 30 = ~16,000
# ⚠️ IMBALANCE (reversed)

# SOLUTION: Downsample non-violent OR get more violent
```

**This is IDEAL but unlikely** - phase1 is probably violent based on the name.

---

### Scenario C: phase1 is MIXED or ORGANIZED

**If phase1 has subdirectories like:**
- `phase1/violent/` and `phase1/nonviolent/`
- `phase1/fight/` and `phase1/normal/`

**Update the script to detect these:**

```bash
# Edit analyze_and_split_dataset.py line 50-90
# Add detection for phase1 subdirectories
```

---

## 🎯 RECOMMENDED ACTION PLAN

### Step 1: Investigate (10 minutes) ← DO THIS NOW

```bash
# On Vast.ai:
cd /workspace/datasets

# Check phase1 structure
ls -la phase1/ | head -20

# Check phase1 subdirectories
find phase1/ -maxdepth 2 -type d | head -30

# Sample phase1 videos
find phase1/ -name "*.mp4" -o -name "*.avi" | head -20

# Analyze reddit
bash /workspace/analyze_reddit_videos.sh
```

**Paste the output here** and I'll tell you exactly how to proceed!

---

### Step 2: Based on Investigation Results

#### If phase1 = VIOLENT:
```bash
# Option A: Use just reddit + youtube (skip phase1 for now)
# - Violent: ~2,100 from reddit + youtube
# - Non-violent: Need to download more!

# Option B: Downsample to 30 per class (tiny dataset)
# Quick test only, not production-ready
```

#### If phase1 = NON-VIOLENT:
```bash
# PERFECT! Use everything:
# - Violent: ~2,100 (reddit + youtube)
# - Non-violent: ~16,000 (phase1)
# Downsample non-violent to 2,100 → balanced 2,100 per class
```

#### If phase1 = MIXED/ORGANIZED:
```bash
# Update categorization script to detect phase1 structure
# Then organize normally
```

---

## 🚀 Quick Commands (Run These Now)

```bash
# 1. Investigate phase1
cd /workspace/datasets
echo "=== phase1 STRUCTURE ===" && ls -la phase1/ | head -20
echo "" && echo "=== phase1 SUBDIRECTORIES ===" && find phase1/ -maxdepth 2 -type d | head -30
echo "" && echo "=== phase1 SAMPLE VIDEOS ===" && find phase1/ -name "*.mp4" -o -name "*.avi" | head -20

# 2. Analyze reddit
bash /workspace/analyze_reddit_videos.sh

# 3. Check reddit_videos (regular, not massive)
echo "" && echo "=== reddit_videos STRUCTURE ===" && ls -la reddit_videos/ | head -20
find reddit_videos/ -name "*.mp4" -o -name "*.avi" | head -10
```

**PASTE THE OUTPUT** and I'll create the exact organization script for your data!

---

## 💡 Alternative: Start with Small Subset

If investigation takes too long, start with what we know:

```bash
# Use only confirmed data:
# - Violent: youtube_fights (105)
# - Non-violent: nonviolent_safe (30)
# Result: 30+30 = 60 videos (tiny, but balanced)

# Quick 5-epoch test:
python runpod_train_l40s.py \
    --dataset-path /workspace/small_test_dataset \
    --epochs 5 \
    --batch-size 8

# Validates pipeline works (~10 min, $0.20)
# Then organize full dataset properly
```

---

## ✅ Action Items (Priority Order)

1. [ ] **NOW**: Run investigation commands above
2. [ ] **5 min**: Paste output here so I can help
3. [ ] **10 min**: I'll create custom organization script for your data
4. [ ] **30-60 min**: Run organization script
5. [ ] **2-3 hours**: Train on L40S GPU
6. [ ] **DONE**: 90-96% accuracy model!

---

**STOP HERE** - Run the investigation commands and paste the output!

Don't run full organization yet - we need to understand phase1 first! 🔍
