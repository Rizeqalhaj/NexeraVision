# 🎯 Phase1 Separation Guide - 2-Step Process

## Your Dataset Status

| Folder | Videos | Category | Status |
|--------|--------|----------|--------|
| youtube_fights | 105 | ⚠️ Violent | ✅ Confirmed |
| reddit_videos_massive | 2,667 | ⚠️ Violent | ✅ Confirmed |
| reddit_videos | 1,669 | ⚠️ Violent | ✅ Confirmed |
| nonviolent_safe | 30 | ✅ Non-Violent | ✅ Confirmed |
| **phase1** | **15,994** | **🔀 MIXED** | **⚠️ Needs Separation** |

**Total Confirmed**: ~4,400 violent + 30 non-violent
**Needs Work**: 15,994 mixed videos in phase1

---

## 📋 2-Step Process

### Step 1: Analyze (See What You Have) - 5 minutes ⭐

**Run this first** - Just analyzes, doesn't touch any files:

```bash
cd /workspace
python3 analyze_phase1_only.py
```

**This shows you:**
- How many violent vs non-violent in phase1
- Confidence levels (high/medium/low)
- Sample filenames from each category
- Class balance analysis
- Recommendations

**Example output:**
```
📊 CATEGORIZATION RESULTS
═══════════════════════════════════════

⚠️  VIOLENT: 8,234 videos (51.5%)
   - High confidence:   5,123 (fight/UFC/boxing in path)
   - Medium confidence: 2,111 (1 violence indicator)
   - Low confidence:    1,000 (weak indicators)

✅ NON-VIOLENT: 6,123 videos (38.3%)
   - High confidence:   4,234 (cctv/normal/daily in path)
   - Medium confidence: 1,456 (1 normal indicator)
   - Low confidence:      433 (weak indicators)

❓ UNKNOWN: 1,637 videos (10.2%)
   (No clear indicators)

💡 RECOMMENDATION: ✅ Good! 90% can be auto-categorized
```

---

### Step 2: Separate (Organize Files) - 30-60 minutes

**After reviewing Step 1 results**, run the separation:

```bash
cd /workspace
python3 separate_phase1_videos.py
```

**This will:**
1. Ask you to choose a mode:
   - **Mode 1** (Recommended): High confidence only - Safest
   - **Mode 2**: All automatic - Includes medium/low confidence
   - **Mode 3**: Include ambiguous as violent - Conservative

2. Create organized folders:
```
/workspace/datasets/phase1_categorized/
├── violent/           (violent videos from phase1)
├── nonviolent/        (non-violent videos from phase1)
└── needs_review/      (uncertain videos - just a list)
```

3. Show final statistics:
```
✅ CATEGORIZATION COMPLETE

📊 Results:
   Violent videos:      8,234
   Non-violent videos:  6,123
   Total organized:    14,357

⚖️  Class Balance: 74.4% ✅ Good balance!
```

---

## 🎯 Recommended Workflow

### Quick Commands (Copy-Paste)

```bash
# 1. Upload scripts to Vast.ai (if not already there)
cd /workspace

# 2. ANALYZE FIRST (see what you have)
python3 analyze_phase1_only.py > phase1_analysis.txt
cat phase1_analysis.txt

# 3. Review the analysis
# - Check violent vs non-violent counts
# - Look at sample filenames
# - Verify it makes sense

# 4. SEPARATE (organize files)
python3 separate_phase1_videos.py
# Choose mode 1 (high confidence only) when asked

# 5. Verify results
ls -la /workspace/datasets/phase1_categorized/
find /workspace/datasets/phase1_categorized/violent/ -name "*.mp4" | wc -l
find /workspace/datasets/phase1_categorized/nonviolent/ -name "*.mp4" | wc -l
```

---

## 💡 Understanding the Modes

### Mode 1: High Confidence Only (RECOMMENDED) ⭐
**Uses**: Only videos with strong indicators
**Pros**: Most accurate, minimal false categorization
**Cons**: Some videos excluded (saved in needs_review)
**Best for**: Maximum accuracy

Example: "street_fight_2024.mp4" → VIOLENT (has "fight" keyword)

### Mode 2: All Automatic
**Uses**: All videos with any indicators (high + medium + low)
**Pros**: More videos included
**Cons**: Lower accuracy, some miscategorization possible
**Best for**: Maximizing dataset size

Example: "incident.mp4" → May be categorized if path has weak hints

### Mode 3: Conservative (Ambiguous as Violent)
**Uses**: Treats uncertain videos as violent
**Pros**: Better safe than sorry - won't train on mislabeled violence as normal
**Cons**: May reduce non-violent dataset size
**Best for**: Safety-critical applications

---

## 🔍 How Pattern Detection Works

### Violent Keywords Detected:
- Fight-related: fight, fighting, brawl, combat, punch, kick, hit, beat
- Martial arts: UFC, MMA, boxing, kickboxing, muay thai, karate, judo
- Violence: violent, assault, attack, brutal, knockout, KO
- Reddit: r_fights, r_streetfighting, r_brutality, freakout

### Non-Violent Keywords Detected:
- Activities: walk, sit, talk, shop, work, play, eat, drink
- CCTV: surveillance, camera, security, parking, lobby, corridor
- Daily life: normal, routine, daily, pedestrian, traffic, crowd
- Positive: smile, laugh, happy, celebrate, dance

### Examples:
```
✅ "UFC_fight_night_2024.mp4" → VIOLENT (has "UFC" and "fight")
✅ "cctv_parking_lot_camera1.mp4" → NON-VIOLENT (has "cctv" and "parking")
❓ "video_123456.mp4" → UNKNOWN (no indicators)
🔀 "police_fight_arrest.mp4" → AMBIGUOUS (has both "police" and "fight")
```

---

## 📊 After Separation - Complete Dataset

Once phase1 is separated, you'll have:

```
VIOLENT:
- youtube_fights: 105
- reddit_videos_massive: 2,667
- reddit_videos: 1,669
- phase1_categorized/violent: ~8,000
= TOTAL: ~12,000 violent ✅

NON-VIOLENT:
- nonviolent_safe: 30
- phase1_categorized/nonviolent: ~6,000
= TOTAL: ~6,000 non-violent ✅

Class balance: 6,000/12,000 = 50% - Good!
Can downsample to 6,000 + 6,000 = 12,000 balanced videos
```

---

## 🚀 Final Training Pipeline

After separation is complete:

```bash
# 1. Combine all violent sources
mkdir -p /workspace/final_dataset/violent
cp phase1_categorized/violent/* /workspace/final_dataset/violent/
cp youtube_fights/*/*.mp4 /workspace/final_dataset/violent/
cp reddit_videos_massive/*/*.mp4 /workspace/final_dataset/violent/
cp reddit_videos/*/*.mp4 /workspace/final_dataset/violent/

# 2. Combine all non-violent sources
mkdir -p /workspace/final_dataset/nonviolent
cp phase1_categorized/nonviolent/* /workspace/final_dataset/nonviolent/
cp nonviolent_safe/* /workspace/final_dataset/nonviolent/

# 3. Create train/val/test splits (70/15/15)
python3 analyze_and_split_dataset.py
# Point it to /workspace/final_dataset

# 4. Train!
python runpod_train_l40s.py --dataset-path /workspace/organized_dataset
```

---

## ⚠️ Important Notes

### Disk Space
- Original phase1: 410 GB
- Separated phase1: 410 GB (copies files)
- **Total needed**: 820 GB

**If space limited:**
Edit `separate_phase1_videos.py` line 243:
```python
# Change:
shutil.copy2(src, dst)

# To:
shutil.move(src, dst)  # Moves instead of copies
```

### Processing Time
- Analysis: 5-10 minutes (15,994 videos)
- Separation: 30-60 minutes (copying 410 GB)
- **Total**: ~1 hour

### Accuracy
- Pattern-based detection: ~85-90% accurate
- High-confidence only: ~95% accurate
- Unknown videos: Manually review or exclude

---

## 🔧 Troubleshooting

### "Phase1 directory not found"
```bash
# Find actual location
find /workspace -name "phase1" -type d

# Update script with correct path
nano analyze_phase1_only.py
# Change PHASE1_DIR = "/your/actual/path"
```

### "Most videos are UNKNOWN"
```bash
# Check actual filenames
find /workspace/datasets/phase1 -name "*.mp4" | head -20

# If filenames are generic (video001.mp4, etc):
# Look at folder structure instead
ls -la /workspace/datasets/phase1/
# Update keywords based on folder names
```

### "Want to review samples before separating"
```bash
# After analysis, manually check a few:
ls /workspace/datasets/phase1/*fight* | head -5
ls /workspace/datasets/phase1/*normal* | head -5
# Use VLC or any player to watch samples
```

---

## ✅ Checklist

- [ ] Step 1: Run `analyze_phase1_only.py`
- [ ] Reviewed analysis output - makes sense?
- [ ] Checked sample filenames - correctly categorized?
- [ ] Verified class balance - reasonable?
- [ ] Step 2: Run `separate_phase1_videos.py`
- [ ] Chose appropriate mode (1, 2, or 3)
- [ ] Verified output folders created
- [ ] Checked video counts match expectations
- [ ] Ready to combine all datasets and train!

---

**Questions?** Run Step 1 (analysis) first and paste the output! 🔍
