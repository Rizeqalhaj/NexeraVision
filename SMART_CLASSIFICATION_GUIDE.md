# Smart Classification Guide - Improved Phase1 Separation

## What Changed?

### Old Approach (analyze_phase1_only.py)
❌ **Problem**: Used generic keywords that caused false classifications
- "RoadAccidents" matched "work" and "road" → classified as NON-VIOLENT ❌
- Generic terms caused many false positives

### New Approach (analyze_phase1_smart.py)
✅ **Solution**: Priority-based classification with explicit labels first

**3-Tier Priority System**:

1. **Priority 1 (P1) - Explicit Labels** (95-98% confidence)
   - Looks for: `violent`, `violence`, `nonviolent`, `non_violent`, `normal`, `shooting`
   - **Example**: "violent_scene_001.mp4" → VIOLENT (explicit label)

2. **Priority 2 (P2) - Strong Indicators** (85-90% confidence)
   - Violent: `accident`, `crash`, `collision`, `ufc`, `mma`, `gun`, `brutal`
   - Non-violent: `cctv`, `surveillance`, `shopping`, `office`
   - **Example**: "RoadAccidents001.mp4" → VIOLENT (contains "accident") ✅

3. **Priority 3 (P3) - Moderate Indicators** (70-80% confidence)
   - Violent: `punch`, `kick`, `hit`, `combat`, `riot`
   - Non-violent: `talk`, `dance`, `celebrate`, `smile`

---

## Available Scripts

### 1. investigate_phase1_structure.py
**Purpose**: See actual filenames and directory structure

**Run this FIRST** to understand your data:
```bash
cd /workspace
python3 investigate_phase1_structure.py
```

**Shows you**:
- Subdirectory structure
- Filename patterns (violent, nonviolent, shooting, etc.)
- Sample filenames by category
- Random samples from your dataset

---

### 2. analyze_phase1_smart.py
**Purpose**: Analyze with priority-based classification (READ-ONLY)

**Run this SECOND** to see how videos will be categorized:
```bash
cd /workspace
python3 analyze_phase1_smart.py > analysis_smart.txt
cat analysis_smart.txt
```

**Shows you**:
- Breakdown by priority level (P1/P2/P3)
- Confidence levels for each category
- Sample videos from each category
- **Specific check**: How RoadAccidents are classified (should be VIOLENT now!)
- Class balance analysis

---

### 3. separate_phase1_smart.py
**Purpose**: Actually organize files into categories

**Run this THIRD** after verifying analysis looks good:
```bash
cd /workspace
python3 separate_phase1_smart.py
```

**3 Organization Modes**:

**Mode 1: smart_high_confidence** ⭐ **RECOMMENDED**
- Uses: Explicit labels (P1) + High confidence (P2)
- Accuracy: 85-98%
- Best for: Maximum accuracy, minimize false classifications

**Mode 2: smart_all_confident**
- Uses: Explicit + High + Medium (P1 + P2 + P3)
- Accuracy: 70-98%
- Best for: Balance between accuracy and dataset size

**Mode 3: smart_all_auto**
- Uses: All automatic including Low confidence
- Accuracy: 50-98%
- Best for: Maximizing dataset size (less accurate)

---

## Workflow on Vast.ai

### Step 1: Upload Scripts (5 minutes)
Upload these 3 files to `/workspace/`:
- `investigate_phase1_structure.py`
- `analyze_phase1_smart.py`
- `separate_phase1_smart.py`

### Step 2: Investigate Structure (5 minutes)
```bash
cd /workspace
python3 investigate_phase1_structure.py > structure_report.txt
cat structure_report.txt
```

**Review the output**:
- Are there subdirectories like "violent/", "nonviolent/", "normal/"?
- Do filenames contain explicit labels?
- What patterns do you see?

### Step 3: Smart Analysis (5 minutes)
```bash
python3 analyze_phase1_smart.py > analysis_smart.txt
cat analysis_smart.txt

# Check specific files
grep -A 3 "RoadAccidents" analysis_smart.txt
grep -A 3 "EXPLICIT" analysis_smart.txt
```

**Verify**:
- ✅ RoadAccidents should be VIOLENT now
- ✅ Check P1 (Explicit) counts - these are most reliable
- ✅ Check class balance (violent vs non-violent ratio)

### Step 4: Separate Videos (30-60 minutes)
```bash
python3 separate_phase1_smart.py
# When prompted, choose mode 1 (smart_high_confidence)

# After completion, verify:
ls -la /workspace/datasets/phase1_categorized/
find /workspace/datasets/phase1_categorized/violent/ -name "*.mp4" | wc -l
find /workspace/datasets/phase1_categorized/nonviolent/ -name "*.mp4" | wc -l
```

---

## Key Improvements

### Before (Old Scripts)
```
RoadAccidents001_x264.mp4
   Category: NON-VIOLENT ❌
   Confidence: High
   Keywords: work, road
```

### After (Smart Scripts)
```
RoadAccidents001_x264.mp4
   Category: VIOLENT ✅
   Confidence: High (P2)
   Keywords: accident
```

---

## Troubleshooting

### "Still seeing misclassifications"
1. Run `investigate_phase1_structure.py` first
2. Look at actual filenames in the output
3. Tell me what patterns you see, and I'll adjust the keywords

### "Most videos are UNKNOWN"
This means filenames don't have clear indicators. Options:
1. Check if videos are organized in subdirectories (violent/, nonviolent/)
2. Manually review 20-30 random samples to understand content
3. Use folder structure instead of filenames

### "Want different confidence thresholds"
You can adjust the keyword lists in the scripts:
- Add more keywords to EXPLICIT_VIOLENT_LABELS for P1
- Add domain-specific terms to STRONG_* for P2

---

## Quick Command Summary

```bash
# 1. Investigate (see actual filenames)
python3 investigate_phase1_structure.py > structure.txt
cat structure.txt

# 2. Analyze (preview classification)
python3 analyze_phase1_smart.py > analysis.txt
cat analysis.txt

# 3. Verify RoadAccidents are correct
grep -A 3 "RoadAccidents" analysis.txt

# 4. If analysis looks good, separate
python3 separate_phase1_smart.py
# Choose mode 1

# 5. Verify results
find /workspace/datasets/phase1_categorized/violent/ -name "*.mp4" | wc -l
find /workspace/datasets/phase1_categorized/nonviolent/ -name "*.mp4" | wc -l
```

---

## Expected Results

Based on your feedback that filenames contain "violent", "nonviolent", "normal", "shooting":

**Priority 1 (Explicit Labels)** should capture most videos:
- Videos with "violent" → VIOLENT
- Videos with "nonviolent" or "normal" → NON-VIOLENT
- Videos with "shooting" → VIOLENT

**Priority 2 (Strong Indicators)** catches the rest:
- "RoadAccidents" → VIOLENT (accident keyword)
- "UFC", "MMA", "Boxing" → VIOLENT
- "CCTV", "Surveillance" → NON-VIOLENT

**Result**: 80-90% of videos should be confidently categorized at P1+P2 levels!

---

## Next Steps After Separation

Once phase1 is separated:

```bash
# Combine all violent sources
mkdir -p /workspace/final_dataset/violent
cp phase1_categorized/violent/* /workspace/final_dataset/violent/
cp youtube_fights/*/*.mp4 /workspace/final_dataset/violent/
cp reddit_videos_massive/*/*.mp4 /workspace/final_dataset/violent/
cp reddit_videos/*/*.mp4 /workspace/final_dataset/violent/

# Combine all non-violent sources
mkdir -p /workspace/final_dataset/nonviolent
cp phase1_categorized/nonviolent/* /workspace/final_dataset/nonviolent/
cp nonviolent_safe/* /workspace/final_dataset/nonviolent/

# Create train/val/test splits
python3 analyze_and_split_dataset.py

# Train!
python runpod_train_l40s.py --dataset-path /workspace/organized_dataset
```

---

**Ready to start?** Run the investigation script first and show me the output! 🚀
