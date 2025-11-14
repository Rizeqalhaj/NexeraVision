# SOLUTION: Training on Shorts for CCTV Deployment

## Problem
- Violence videos: Shorts (action-packed)
- Non-violence videos: Shorts (action-packed daily activities)
- CCTV reality: Hours of boring footage with rare violence

## Why This Still Works (With Adjustments)

### ✅ Good News:
Your shorts-based training **CAN work** for CCTV if you:
1. Use sliding window inference (already planned)
2. Add negative "boring" samples during training
3. Adjust confidence thresholds for CCTV deployment
4. Use temporal smoothing to reduce false positives

## Three-Tier Training Strategy

### Tier 1: Action Recognition (Your Current Shorts) - 60%
**Violence Shorts:**
- Fighting, punching, kicking
- Teaches: "This is what violence looks like"

**Non-Violence Shorts:**
- Walking, eating, talking, cooking
- Teaches: "This is what normal activities look like"

**Purpose:** Core violence vs normal action discrimination

### Tier 2: "Boring" Negative Samples - 30%
**NEW REQUIREMENT - Need to add:**
- Empty hallways (nobody there)
- People sitting still
- Slow walking in distance
- Standing/waiting
- Minimal motion scenes

**Purpose:** Teach model that "nothing happening" = non-violence

### Tier 3: CCTV-Style Long Sequences - 10%
**UCF Crime Normal Videos (longer clips):**
- 1-5 minute clips of normal surveillance
- Sample random 16-frame windows from these
- Most windows = boring/nothing

**Purpose:** Simulate CCTV sampling distribution

## Practical Implementation

### Current Dataset Split (What You Have)
```
Violence (50%):
├─ YouTube Shorts: 5,000 videos × 20 frames = 100K clips
└─ Kaggle datasets: 2,000 videos × 20 frames = 40K clips
Total: 140K violence clips

Non-Violence (50%):
├─ YouTube Shorts daily activities: 10,000 videos × 20 frames = 200K clips
└─ UCF Crime normal: TBD
Total: 200K+ non-violence clips
```

### Enhanced Dataset (What You Need)
```
Violence (40%):
├─ Shorts: 140K clips [HAVE ✓]

Non-Violence Active (30%):
├─ Walking/eating/talking shorts: 100K clips [HAVE ✓]

Non-Violence Boring (30%):
├─ Minimal motion: Need 100K clips [NEED ⚠️]
├─ Standing still: Need
├─ Empty scenes: Need
└─ Slow walking: Need
```

## How to Get "Boring" Non-Violence Videos

### Option A: Sample from UCF Crime Normal (EASIEST)
```python
# Extract long UCF Crime normal videos
# Sample random 2-second clips with LOW motion
# These are naturally "boring" CCTV footage
```

### Option B: YouTube CCTV Surveillance (RECOMMENDED)
```python
# Search for:
BORING_KEYWORDS = [
    "empty hallway cctv",
    "parking lot surveillance",
    "office lobby camera",
    "waiting room footage",
    "corridor surveillance",
    "building entrance camera",
    "elevator security camera",
    "street surveillance quiet",
    "mall cctv empty",
    "store surveillance no customers"
]
```

### Option C: Generate Synthetic Boring Clips
```python
# Take your existing non-violence videos
# Apply heavy motion blur to simulate "boring" versions
# Or extract frames with minimal optical flow
```

## Training Code Adjustment

### Current Approach (Needs Fix):
```python
# Problem: All samples are "interesting"
train_data = violence_shorts + nonviolence_shorts
# Model learns: every clip has clear action
```

### Fixed Approach:
```python
# Mix of interesting + boring samples
train_data = {
    'violence_active': violence_shorts,           # 40%
    'nonviolence_active': walking_eating_shorts,  # 30%
    'nonviolence_boring': cctv_low_motion        # 30%
}

# Sample with proper distribution
def sample_batch(batch_size=32):
    violence = sample(violence_active, int(batch_size * 0.4))
    active_normal = sample(nonviolence_active, int(batch_size * 0.3))
    boring_normal = sample(nonviolence_boring, int(batch_size * 0.3))
    return shuffle(violence + active_normal + boring_normal)
```

## CCTV Deployment Strategy

### Inference with Temporal Smoothing
```python
# Don't trigger on single frame prediction
# Require sustained violence detection

violence_buffer = []
BUFFER_SIZE = 5  # Need 5 consecutive positive predictions
THRESHOLD = 0.7

for window in sliding_windows(cctv_stream):
    pred = model.predict(window)

    violence_buffer.append(pred > THRESHOLD)
    if len(violence_buffer) > BUFFER_SIZE:
        violence_buffer.pop(0)

    # Only trigger if 4 out of 5 recent predictions are violence
    if sum(violence_buffer) >= 4:
        TRIGGER_ALERT()
```

### Adaptive Thresholding
```python
# Different thresholds for different motion levels
motion_level = calculate_optical_flow(window)

if motion_level < LOW_THRESHOLD:
    # Boring scene - require higher confidence
    threshold = 0.9
elif motion_level < MEDIUM_THRESHOLD:
    # Normal activity - standard threshold
    threshold = 0.7
else:
    # High motion - lower threshold (might be fight)
    threshold = 0.6

if prediction > threshold:
    violence_buffer.append(True)
```

## Immediate Action Items

### 🔴 CRITICAL - Get Boring Samples
1. **Extract UCF Crime Normal long videos**
   - Sample low-motion clips
   - Target: 50K+ "boring" clips

2. **Scrape CCTV surveillance footage**
   - Search YouTube for empty hallways, parking lots
   - Download 2-3K videos
   - Extract low-motion segments

### 🟡 MEDIUM - Training Script Updates
1. **Modify data loader** to sample from 3 categories:
   - Violence active
   - Non-violence active
   - Non-violence boring

2. **Add motion-based filtering** during training:
   - Calculate optical flow
   - Ensure "boring" samples have flow < threshold

3. **Adjust class weights**:
   ```python
   class_weights = {
       'violence': 1.0,
       'nonviolence_active': 1.0,
       'nonviolence_boring': 1.5  # Slightly higher weight
   }
   ```

### 🟢 NICE TO HAVE - Post-Processing
1. **Temporal smoothing** in inference
2. **Motion-adaptive thresholds**
3. **Multi-stage detection** (fast motion → violence classifier)

## Expected Impact

| Training Mix | CCTV False Positives | Violence Detection | Deployable? |
|-------------|---------------------|-------------------|-------------|
| Current (shorts only) | **VERY HIGH** | 90% | ❌ NO |
| + Boring samples (30%) | Medium | 88% | ⚠️ Maybe |
| + Temporal smoothing | Low | 87% | ✅ YES |
| + Adaptive thresholds | Very Low | 90% | ✅✅ BEST |

## Quick Test Before Deployment

```python
# Test on 1-hour CCTV footage with 2 violence incidents
# Measure:
# - True Positives: Did it catch both incidents?
# - False Positives: How many false alarms per hour?
# - Latency: How fast did it detect?

# Acceptable: <1 false positive per hour, >90% detection rate
```

## Recommendation

**Immediate next steps:**

1. ✅ Keep downloading your YouTube shorts (already doing)
2. 🔴 **ADD: Scrape "boring" CCTV footage** (empty hallways, parking lots)
3. 🔴 **EXTRACT: UCF Crime Normal videos** - sample low-motion clips
4. 🟡 Modify training to use 40/30/30 split (violence/active/boring)
5. 🟡 Add temporal smoothing to inference

**This will dramatically reduce false positives while maintaining accuracy.**
