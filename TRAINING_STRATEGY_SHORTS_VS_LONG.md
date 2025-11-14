# Training Strategy: Shorts vs Long Videos for CCTV Violence Detection

## Problem Statement
Training on YouTube Shorts (15-60s) but deploying on CCTV (hours of footage) creates a domain mismatch.

## Optimal Training Strategy

### Phase 1: Core Violence Recognition (Shorts OK)
**Use:** YouTube Shorts, Kaggle datasets (RWF-2000, UCF Crime short clips)
**Goal:** Learn what violence LOOKS like (punching, kicking, aggressive movements)
**Duration:** 50-60% of training data
**Why:** Shorts are excellent for learning violence visual features

### Phase 2: Context & False Positive Reduction (Need Longer Videos)
**Use:** Full-length CCTV datasets, longer UCF Crime sequences
**Goal:** Learn normal behavior patterns, reduce false positives
**Duration:** 40-50% of training data
**Why:** Teaches model that fast movement ≠ always violence

## Recommended Dataset Mix

```
VIOLENCE (50%):
├─ YouTube Shorts fights: 30% (quick action recognition)
├─ UCF Crime Assault/Fight: 15% (realistic CCTV quality)
└─ RWF-2000 Fight clips: 5% (research quality)

NON-VIOLENCE (50%):
├─ YouTube Shorts daily activities: 20% (fast movements that aren't violence)
├─ UCF Crime Normal videos: 20% (real CCTV normal behavior)
└─ Surveillance datasets (long): 10% (realistic idle/walking/talking)
```

## Technical Adaptations

### 1. Temporal Window Adjustment
```python
# For Shorts-trained model
TEMPORAL_WINDOW = 16 frames  # ~0.5s @ 30fps
STRIDE = 8 frames

# For CCTV deployment - use sliding window
CCTV_WINDOW = 32 frames  # ~1s @ 30fps
CCTV_STRIDE = 8 frames  # Check every 0.25s
```

### 2. Training with Variable-Length Clips
```python
# Sample clips of different lengths during training
CLIP_LENGTHS = [16, 32, 64, 96]  # 0.5s to 3s
# Randomly sample clip length per batch
# This teaches model to work with different temporal contexts
```

### 3. Data Augmentation for Context
```python
# Add "normal frames" before/after violence clips
def add_context(violence_clip, normal_clips, context_frames=30):
    """
    violence_clip: 16 frames of fighting
    Add 30 frames of normal behavior before/after
    Total: 76 frames showing escalation and aftermath
    """
    before = random.choice(normal_clips)[:context_frames]
    after = random.choice(normal_clips)[:context_frames]
    return concat(before, violence_clip, after)
```

## Model Architecture Recommendations

### Current Model (Good for Shorts)
```
Input: 16 frames × 224×224
Feature Extraction: VGG16/ResNet50
Temporal: LSTM/3D CNN
Output: Violence probability
```

### Enhanced Model (Better for CCTV)
```
Input: 32-64 frames × 224×224 (longer temporal context)
Feature Extraction: Two-Stream Network
  ├─ Spatial Stream: VGG16 (what is happening)
  └─ Temporal Stream: Optical Flow (how it's moving)
Temporal: LSTM with Attention (focus on key moments)
Output: Violence probability + Confidence score
```

## Deployment Strategy for CCTV

### Option A: Sliding Window (Current Approach)
```python
# Process 1-second windows every 0.25s
for window in sliding_windows(cctv_stream, window_size=32, stride=8):
    prediction = model.predict(window)
    if prediction > THRESHOLD:
        trigger_alert()
```

### Option B: Two-Stage Detection (Recommended)
```python
# Stage 1: Fast motion detector (lightweight)
if fast_motion_detected(frame):
    # Stage 2: Violence classifier (your trained model)
    prediction = model.predict(frames)
    if prediction > THRESHOLD:
        trigger_alert()
```

## Current Dataset Status vs Ideal

| Dataset Type | Current | Ideal | Status |
|-------------|---------|-------|--------|
| Short violence clips | ✓ Have | Need | ✓ |
| Short non-violence | ✓ Have | Need | ✓ |
| Long CCTV violence | ? | Need | ⚠️ Missing |
| Long CCTV normal | ✓ UCF Crime | Need | ⚠️ Partial |

## Action Items

### 🔴 CRITICAL - Missing Long-Form CCTV Data
You need more **full-length CCTV footage** with violence incidents:

1. **UCF Crime Dataset** (You're downloading):
   - Has longer sequences (~30s to 2min)
   - Real surveillance camera footage
   - **Extract full video clips, not just snippets**

2. **Additional CCTV Fight Datasets**:
   - RLVS (Hockey fights - longer sequences)
   - VSD (Violent Scene Detection)
   - Surveillance Fight Dataset

### 🟡 RECOMMENDED - Training Adjustments

1. **Increase Temporal Window**:
   ```python
   # Change from 16 frames to 32-64 frames
   SEQUENCE_LENGTH = 64  # ~2 seconds @ 30fps
   ```

2. **Add Temporal Context Augmentation**:
   ```python
   # Randomly add normal frames before/after violence
   ```

3. **Two-Stage Training**:
   ```python
   # Stage 1: Train on shorts (quick violence recognition)
   # Stage 2: Fine-tune on longer CCTV clips (reduce false positives)
   ```

## Expected Accuracy Impact

| Training Approach | Accuracy on Shorts | Accuracy on CCTV | False Positives |
|------------------|-------------------|------------------|-----------------|
| Only Shorts | 90-95% | 70-80% | High |
| Mixed (50/50) | 88-92% | 85-90% | Medium |
| Mostly Long-form | 85-90% | 90-95% | Low |

## Final Recommendation

**For your 110 CCTV camera deployment:**

1. **Train with current Shorts data first** → Get to 90% accuracy
2. **Fine-tune with UCF Crime long sequences** → Reduce false positives
3. **Test with realistic CCTV footage** → Validate before deployment
4. **Adjust temporal window to 32-64 frames** → Better context understanding

**Your current approach is good for Phase 1, but you MUST add longer CCTV footage for reliable real-world deployment.**

## Quick Test

After training, test your model on:
- ✅ Shorts → Should be 90%+ accurate
- ⚠️ 5-minute CCTV clip with 10s of violence in middle → Can it detect without triggering on normal activity?

If it triggers on people walking fast, running to bus, kids playing → You need more long-form training data.
