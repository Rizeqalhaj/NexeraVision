# Dataset Quality Validation Guide

## 🎯 Why Validate?

**Problem**: Downloaded videos may not actually be violent or may be mislabeled:
- ❌ Static images labeled as videos
- ❌ Non-violent content (dancing, sports, etc.)
- ❌ Very short clips (< 3 seconds) without actual violence
- ❌ Corrupted or broken videos
- ❌ Slideshows or compilations

**Solution**: Automatic validation + manual review to ensure dataset quality

---

## 🔍 Validation Process

### Automated Checks

The validator automatically checks for:

1. **Video Duration**
   - ✅ OK: 3-600 seconds (3s - 10min)
   - ⚠️ SUSPICIOUS: < 3 seconds (too short for real fight)
   - 🔍 REVIEW: > 600 seconds (too long, might be compilation)

2. **Motion Analysis**
   - ✅ OK: Motion score > 10 (active fighting)
   - ⚠️ SUSPICIOUS: Motion score < 5 (static/slideshow)
   - 🔍 REVIEW: Motion score 5-10 (minimal action)

3. **Video Metadata**
   - Resolution (CCTV typically 480p-720p)
   - Frame rate (15-30 fps typical)
   - Codec information

4. **Thumbnail Generation**
   - Creates 3×3 grid of keyframes
   - Visual inspection of video content
   - Spot-check for actual violence

---

## 🚀 Quick Start: Validate Your Dataset

### Step 1: Download Videos
```bash
# Example: Download CCTV violent footage
python3 download_cctv_surveillance.py --sources reddit --max-reddit 1000
```

### Step 2: Run Validation (Sample 100 videos)
```bash
python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance \
    --output-dir ./validation_results \
    --sample-size 100 \
    --random-sample
```

**Output**:
- `validation_results/validation_report.json` - Machine-readable results
- `validation_results/validation_report.html` - Visual inspection report
- `validation_results/thumbnails/` - Video keyframe grids

### Step 3: Review Results
```bash
# Open HTML report in browser (on local machine)
# Download validation_results folder from RunPod
# Open validation_report.html

# Or review in terminal:
cat validation_results/validation_report.json | jq '.[] | select(.suspicion_level=="SUSPICIOUS")'
```

### Step 4: Clean Dataset (Dry Run First)
```bash
# Dry run - see what would be removed
python3 clean_dataset.py \
    --validation-report validation_results/validation_report.json \
    --action move \
    --dry-run

# If satisfied, remove suspicious videos
python3 clean_dataset.py \
    --validation-report validation_results/validation_report.json \
    --action move
```

---

## 📊 Validation Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│             DOWNLOAD VIDEOS (Phase 1/2/3)                   │
│  download_cctv_surveillance.py → ~20K videos                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          VALIDATE SAMPLE (100-500 videos)                   │
│  validate_violent_videos.py --sample-size 100               │
│  → Generate thumbnails, motion analysis, metadata           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         REVIEW VALIDATION REPORT (Manual)                   │
│  Open validation_report.html in browser                     │
│  → Check thumbnails for actual violence                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   ┌────────┴────────┐
                   │  Quality OK?     │
                   └────────┬────────┘
                     YES │   │ NO
                         │   │
        ┌────────────────┘   └────────────────┐
        ↓                                      ↓
┌─────────────────────┐          ┌─────────────────────────┐
│  PROCEED TO FULL    │          │  CLEAN DATASET          │
│  VALIDATION         │          │  clean_dataset.py       │
│  (entire dataset)   │          │  → Remove suspicious    │
└─────────────────────┘          └─────────────────────────┘
        ↓                                      ↓
┌─────────────────────┐          ┌─────────────────────────┐
│  CLEAN SUSPICIOUS   │          │  RE-VALIDATE            │
│  VIDEOS             │          │  Confirm quality OK     │
└─────────────────────┘          └─────────────────────────┘
        ↓                                      ↓
        └──────────────────┬───────────────────┘
                           ↓
                ┌─────────────────────┐
                │ CLEAN DATASET READY │
                │ FOR TRAINING        │
                └─────────────────────┘
```

---

## 🎯 Validation Strategies

### Strategy 1: Quick Sample (RECOMMENDED FIRST)
```bash
# Validate random 100 videos to get quality estimate
python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance \
    --sample-size 100 \
    --random-sample

# Expected time: 5-10 minutes
# Purpose: Quick quality check before full validation
```

**Decision point**:
- If < 10% suspicious → Good quality, proceed to full validation
- If 10-30% suspicious → Medium quality, clean and re-download
- If > 30% suspicious → Poor quality, change download source

### Strategy 2: Full Validation (After Sample Passes)
```bash
# Validate entire dataset
python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance \
    --sample-size 0  # 0 = all videos

# Expected time: 30-120 minutes for 20K videos
# Purpose: Comprehensive quality assurance
```

### Strategy 3: Per-Source Validation
```bash
# Validate each source separately to identify best sources
python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance/reddit_cctv \
    --sample-size 0

python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance/youtube_cctv \
    --sample-size 0

# Compare quality between sources
# Keep best sources, discard poor sources
```

---

## 📊 Quality Metrics

### Expected Quality Benchmarks

| Dataset Source | Expected Suspicious % | Expected Motion Score | Expected Duration |
|----------------|----------------------|----------------------|-------------------|
| Reddit (r/fightporn) | 5-15% | 15-30 | 10-120s |
| YouTube CCTV | 10-20% | 12-25 | 15-180s |
| Kaggle RWF-2000 | 2-5% | 20-35 | 5-30s |
| Kaggle UCF Crime | 5-10% | 15-28 | 10-60s |
| Vimeo | 8-15% | 18-32 | 30-300s |

### Quality Thresholds

**Excellent Quality** (< 5% suspicious):
- Proceed directly to training
- High confidence in dataset

**Good Quality** (5-15% suspicious):
- Clean suspicious videos
- Proceed to training after cleanup

**Medium Quality** (15-30% suspicious):
- Clean suspicious videos
- Re-validate after cleanup
- Consider adding more from better sources

**Poor Quality** (> 30% suspicious):
- Discard entire source
- Try different download sources
- Focus on known-quality datasets (Kaggle)

---

## 🔧 Advanced Validation

### Custom Suspicion Rules

Edit `validate_violent_videos.py` to add custom checks:

```python
# Example: Flag videos with people in suits (likely not fights)
# Requires additional ML model for person detection

# Example: Flag videos with watermarks (compilations)
# Use image processing to detect watermarks

# Example: Flag videos with audio indicators
# Check for commentary audio (sports broadcasts vs raw CCTV)
```

### Manual Review Process

1. **Open HTML Report**:
   ```bash
   # Download validation_results/ from RunPod to local machine
   # Open validation_results/validation_report.html in browser
   ```

2. **Review Thumbnails**:
   - Look for actual punching, kicking, wrestling
   - Check for CCTV camera angle (top-down, corner)
   - Verify motion/action in frames

3. **Mark for Removal**:
   - Note video filenames that are not violent
   - Add to manual blacklist

4. **Re-validate**:
   ```bash
   # After manual review, remove additional videos
   python3 clean_dataset.py \
       --validation-report validation_results/validation_report.json \
       --action move
   ```

---

## 📈 Quality Improvement Workflow

### Phase 1: Initial Download
```bash
python3 download_cctv_surveillance.py --sources reddit --max-reddit 5000
# Downloads 5,000 videos from Reddit
```

### Phase 2: Quick Quality Check
```bash
python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance \
    --sample-size 200 \
    --random-sample
```

**Result**: 15% suspicious (75 out of 500)

### Phase 3: Decision
- **If OK** (< 20%): Proceed to full validation
- **If Poor** (> 20%): Try different source

### Phase 4: Full Validation
```bash
python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance \
    --sample-size 0
```

**Result**: 750 suspicious out of 5,000 (15%)

### Phase 5: Cleanup
```bash
python3 clean_dataset.py \
    --validation-report validation_results/validation_report.json \
    --action move
```

**Result**: 4,250 clean videos remaining

### Phase 6: Re-download to Target
```bash
# Need 20,000 violent videos total
# Have 4,250 clean (21% of target)
# Need to download 4.7× more

python3 download_cctv_surveillance.py --sources reddit --max-reddit 24000
# Download 24K, expect ~20K after cleanup
```

---

## 💡 Pro Tips

### 1. Validate Early
Don't wait until you've downloaded 100K videos to validate. Check quality after first 1,000-5,000.

### 2. Source-Specific Validation
Different sources have different quality:
- Reddit r/fightporn: Usually good quality
- YouTube "caught on camera": Variable quality
- Vimeo: Generally high quality
- Random YouTube searches: Often poor quality

### 3. Progressive Cleaning
```bash
# Validate in batches
for dir in /workspace/datasets/*/; do
    python3 validate_violent_videos.py --dataset-dir "$dir" --sample-size 100
done

# Clean all at once
find /workspace/datasets -name "validation_report.json" -exec python3 clean_dataset.py --validation-report {} \;
```

### 4. Motion Score Calibration
- **CCTV fights**: Motion score 10-30 (normal)
- **UFC/MMA**: Motion score 20-40 (higher action)
- **Static images**: Motion score < 5 (remove)

Adjust thresholds in `validate_violent_videos.py`:
```python
if motion_score < 10:  # Change from 5 to 10 for stricter filtering
    suspicion_level = "SUSPICIOUS"
```

---

## 🚨 Common Issues

### Issue 1: Too Many False Positives (> 30% suspicious)
**Cause**: Download source has poor quality
**Solution**: Switch to better source (Kaggle > Reddit > YouTube)

### Issue 2: All Videos Flagged as Suspicious
**Cause**: Motion detection sensitivity too high
**Solution**: Lower motion threshold in validator

### Issue 3: Thumbnails Not Showing
**Cause**: Video codec not supported by OpenCV
**Solution**: Install ffmpeg: `apt-get install ffmpeg`

### Issue 4: Validation Very Slow
**Cause**: Processing full 4K videos
**Solution**: Already optimized (uses 320x240 thumbnails)

---

## ✅ Quality Checklist

Before training, ensure:
- [ ] Validated at least 10% of dataset (random sample)
- [ ] Suspicious rate < 20%
- [ ] Reviewed HTML report thumbnails
- [ ] Cleaned suspicious videos
- [ ] Re-validated after cleanup
- [ ] Final dataset has clear violent action in samples
- [ ] Motion scores reasonable (> 10 average)
- [ ] Video durations reasonable (3-600s)

---

## 📝 Example Complete Workflow

```bash
# 1. Download CCTV violent footage
python3 download_cctv_surveillance.py --sources reddit --max-reddit 10000

# 2. Quick quality check (200 random videos)
python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance \
    --sample-size 200 \
    --random-sample

# 3. Review results
cat validation_results/validation_report.json | jq '.[] | select(.suspicion_level!="OK") | {video: .video, reasons: .reasons}'

# 4. If quality OK (< 20% suspicious), validate all
python3 validate_violent_videos.py \
    --dataset-dir /workspace/datasets/cctv_surveillance \
    --sample-size 0

# 5. Clean suspicious videos (dry run first)
python3 clean_dataset.py \
    --validation-report validation_results/validation_report.json \
    --dry-run

# 6. Actually clean
python3 clean_dataset.py \
    --validation-report validation_results/validation_report.json \
    --action move

# 7. Verify final count
find /workspace/datasets/cctv_surveillance -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l

# 8. If needed, download more to reach target
# Target: 20,000 violent videos
# If have 15,000 after cleanup, download 6,000 more (assuming 20% will be cleaned)
```

---

## 🎉 Success Criteria

Your dataset is **production-ready** when:
- ✅ Suspicious rate < 10% after first validation
- ✅ Manual review confirms violence in random samples
- ✅ Average motion score > 15
- ✅ Average duration 10-120 seconds
- ✅ No corrupted videos (all thumbnails generated)
- ✅ Final count meets target (20K+ for violent)

**Ready to train with confidence!** 🚀
