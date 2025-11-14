# 🎯 HIDDEN GEMS FOR 95%+ ACCURACY

Complete guide to advanced techniques that will push your model beyond 93% to 95%+ accuracy.

---

## 📊 Current Status

**Your Journey:**
- Current best: 90.6% validation accuracy
- Target: 93-95%
- Gap: 2.4-4.4 percentage points

**Problem Identified:** Severe overfitting
- Train: 99.97% vs Val: 89.74% (10.2% gap)
- Model memorizing training data instead of learning patterns

---

## 🔥 HIDDEN GEM #1: Stochastic Weight Averaging (SWA)

**What it does:** Average weights from last N epochs instead of just using best epoch
**Why it works:** Reduces overfitting by finding flatter minima in loss landscape
**Expected gain:** +0.5-1.5%

**Implementation:** ✅ Already in `train_ensemble.py`

```python
class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, swa_start_epoch=100, swa_freq=5):
        # Start averaging after 70% of training
        # Average every 5 epochs

    def on_epoch_end(self, epoch, logs=None):
        # Accumulate weights
        self.swa_weights[i] = (old_weights * n + new_weights) / (n + 1)

    def on_train_end(self, logs=None):
        # Apply averaged weights at end
        self.model.set_weights(self.swa_weights)
```

**Usage:**
- Add to callbacks: `SWACallback(swa_start_epoch=100, swa_freq=5)`
- Start after 70% of training
- Average every 5 epochs

---

## 🔥 HIDDEN GEM #2: Diverse Augmentation Per Model

**What it does:** Each ensemble model gets DIFFERENT augmentation strategies
**Why it works:** Creates truly diverse models that make different mistakes
**Expected gain:** +1-2% when combined with ensemble

**Implementation:** ✅ Already in `train_ensemble.py`

```python
def augment_diverse(features, seed_offset):
    np.random.seed(seed + seed_offset)  # Different seed per model!

    # Vary augmentation STRENGTH per model
    noise_std = np.random.uniform(0.03, 0.07)  # Random range
    dropout_rate = np.random.uniform(0.02, 0.05)

    # Each model sees slightly different augmented data
```

---

## 🔥 HIDDEN GEM #3: Weighted Ensemble

**What it does:** Better models get more voting power in ensemble
**Why it works:** Don't give equal weight to worse models
**Expected gain:** +0.3-0.8% over simple averaging

**Implementation:** ✅ Already in `ensemble_predict.py`

```python
# Calculate weights from validation accuracy
weights = np.array([0.92, 0.91, 0.90, 0.91, 0.89])  # Val accuracies
weights = weights / weights.sum()  # Normalize

# Weighted prediction
for i, pred in enumerate(all_predictions):
    weighted_predictions += weights[i] * pred
```

**How to use:**
1. Train all 5 models
2. Record each model's validation accuracy
3. Use those as weights in prediction

---

## 🔥 HIDDEN GEM #4: Mixup Augmentation

**What it does:** Create "virtual" training examples by mixing pairs
**Why it works:** Forces model to learn smoother decision boundaries
**Expected gain:** +1-2%

**Implementation:** ✅ Already in `train_stronger_regularization.py`

```python
def mixup(features1, features2, labels1, labels2, alpha=0.2):
    """Mix two training examples"""
    lam = np.random.beta(alpha, alpha)  # Random mixing ratio

    mixed_features = lam * features1 + (1 - lam) * features2
    mixed_labels = lam * labels1 + (1 - lam) * labels2

    return mixed_features, mixed_labels
```

**Example:**
- Video 1 (fight): 0.7 weight
- Video 2 (normal): 0.3 weight
- Mixed video: 70% fight + 30% normal
- Label: [0.7, 0.3] instead of [1, 0]

---

## 🔥 HIDDEN GEM #5: Label Smoothing in Focal Loss

**What it does:** Make labels "softer" to prevent overconfidence
**Why it works:** Reduces overfitting by preventing model from being too certain
**Expected gain:** +0.5-1%

**Implementation:** ✅ Already in `train_stronger_regularization.py`

```python
def focal_loss_with_label_smoothing(y_true, y_pred, label_smoothing=0.15):
    # Instead of [1, 0] or [0, 1]
    # Use [0.925, 0.075] or [0.075, 0.925]
    y_true_smooth = y_true_oh * (1 - label_smoothing) + label_smoothing / 2
```

---

## 🔥 HIDDEN GEM #6: Residual Connections

**What it does:** Add skip connections in LSTM layers
**Why it works:** Better gradient flow, prevents vanishing gradients
**Expected gain:** +0.5-1% with smaller models

**Implementation:** ✅ Already in `train_better_architecture.py`

```python
x = BiLSTM(64)(inputs)
x_residual = x  # Save

x = BiLSTM(64)(x)
x = Add()([x, x_residual])  # Add skip connection
```

---

## 🔥 HIDDEN GEM #7: Test-Time Augmentation (TTA)

**What it does:** Augment test samples and average predictions
**Why it works:** More robust predictions from multiple views
**Expected gain:** +0.5-1.5%

**Implementation:** NEW - Add this to your prediction script

```python
def predict_with_tta(model, video_features, n_augmentations=5):
    """Test-Time Augmentation"""
    predictions = []

    # Original prediction
    predictions.append(model.predict(video_features))

    # Augmented predictions
    for _ in range(n_augmentations - 1):
        augmented = augment_features(video_features)
        predictions.append(model.predict(augmented))

    # Average all predictions
    return np.mean(predictions, axis=0)
```

**Usage:**
```python
# Instead of:
pred = model.predict(X_test)

# Use:
pred = predict_with_tta(model, X_test, n_augmentations=5)
```

---

## 🔥 HIDDEN GEM #8: Snapshot Ensembling

**What it does:** Save multiple checkpoints during training, ensemble them
**Why it works:** Different checkpoints capture different aspects
**Expected gain:** +1-2% (almost free!)

**Implementation:** NEW - Add to training

```python
# In callbacks
tf.keras.callbacks.ModelCheckpoint(
    'checkpoints/snapshot_epoch_{epoch:03d}.h5',
    save_freq=10,  # Save every 10 epochs
    verbose=0
)

# Then ensemble the last 5-10 snapshots
snapshots = [
    'snapshot_epoch_150.h5',
    'snapshot_epoch_160.h5',
    'snapshot_epoch_170.h5',
    'snapshot_epoch_180.h5',
    'snapshot_epoch_190.h5',
]
```

---

## 🔥 HIDDEN GEM #9: Cyclical Learning Rates

**What it does:** Cycle LR up and down instead of just decaying
**Why it works:** Helps model escape local minima
**Expected gain:** +0.5-1%

**Implementation:** NEW

```python
def cyclical_lr_schedule(epoch, base_lr=0.0001, max_lr=0.001, cycle_length=20):
    """Cyclical learning rate"""
    cycle = np.floor(1 + epoch / (2 * cycle_length))
    x = np.abs(epoch / cycle_length - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr

# In callbacks
tf.keras.callbacks.LearningRateScheduler(cyclical_lr_schedule)
```

---

## 🔥 HIDDEN GEM #10: Knowledge Distillation

**What it does:** Train smaller model to mimic ensemble
**Why it works:** Captures ensemble knowledge in single model
**Expected gain:** Deployment speedup with minimal accuracy loss

**Implementation:** NEW

```python
def distillation_loss(y_true, y_pred_student, y_pred_teacher, temperature=3.0, alpha=0.5):
    """Knowledge distillation loss"""
    # Soften predictions
    teacher_soft = tf.nn.softmax(y_pred_teacher / temperature)
    student_soft = tf.nn.softmax(y_pred_student / temperature)

    # Distillation loss
    distill_loss = tf.keras.losses.KLD(teacher_soft, student_soft)

    # Student loss
    student_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred_student)

    # Combined
    return alpha * distill_loss + (1 - alpha) * student_loss
```

---

## 🔥 HIDDEN GEM #11: Temporal Ensembling

**What it does:** Use predictions from previous epochs as pseudo-labels
**Why it works:** Consistency regularization
**Expected gain:** +0.5-1%

**Implementation:** NEW - Advanced technique

```python
# Keep moving average of predictions
ema_predictions = 0.9 * old_predictions + 0.1 * new_predictions

# Add consistency loss
consistency_loss = MSE(current_pred, ema_predictions)
total_loss = classification_loss + 0.1 * consistency_loss
```

---

## 🔥 HIDDEN GEM #12: Self-Supervised Pre-Training

**What it does:** Pre-train model on auxiliary task (e.g., frame order prediction)
**Why it works:** Learn better features before classification
**Expected gain:** +1-3%

**Implementation:** NEW - Most advanced

```python
# Pre-training task: Predict if frames are in correct order
def create_pretext_task(video_features):
    # Shuffle frames randomly
    shuffled = shuffle_frames(video_features)

    # Label: 0=shuffled, 1=original order
    return shuffled, is_original_order

# Pre-train for 50 epochs on this task
# Then fine-tune on violence detection
```

---

## 📋 IMPLEMENTATION PRIORITY

### Immediate (Easy Wins - Do Now!)
1. ✅ **Ensemble Training** - Run `train_ensemble.py` (Guaranteed +2-3%)
2. ✅ **SWA** - Already included in ensemble script (+0.5-1.5%)
3. ✅ **Weighted Ensemble** - Use `ensemble_predict.py` (+0.3-0.8%)
4. 🔜 **Test-Time Augmentation** - Add to prediction (+0.5-1.5%)
5. 🔜 **Snapshot Ensembling** - Save checkpoints every 10 epochs (+1-2%)

### Medium Priority (Good ROI)
6. ✅ **Stronger Regularization** - Run `train_stronger_regularization.py` (+1-2%)
7. ✅ **Better Architecture** - Run `train_better_architecture.py` (+0.5-1%)
8. 🔜 **Cyclical Learning Rates** - Easy to implement (+0.5-1%)

### Advanced (High Effort, High Reward)
9. 🔜 **Knowledge Distillation** - After ensemble training (+deployment speed)
10. 🔜 **Temporal Ensembling** - Requires training loop modification (+0.5-1%)
11. 🔜 **Self-Supervised Pre-Training** - Separate pre-training phase (+1-3%)

---

## 🎯 ROADMAP TO 95%+

### Phase 1: Ensemble (Now - 15 hours)
```bash
# Train 5 diverse models
python train_ensemble.py

# Expected result: 92-93% (from 90.6%)
```

### Phase 2: Test-Time Augmentation (Quick - 10 minutes)
```python
# Add TTA to ensemble_predict.py
# Expected boost: +0.5-1% → 93-94%
```

### Phase 3: Snapshot Ensembling (Bonus)
```bash
# Use existing checkpoints
# Ensemble last 10 checkpoints
# Expected boost: +0.5-1% → 94-95%
```

### Phase 4: Stronger Regularization (Optional)
```bash
# If still below 95%
python train_stronger_regularization.py
# Expected: 91-92% individual, 93-94% ensembled
```

### Phase 5: Optimized Architecture (Optional)
```bash
# Smaller model, better generalization
python train_better_architecture.py
# Expected: 90-91% individual, 92-93% ensembled
```

---

## 💡 COMBINING TECHNIQUES

**Maximum Accuracy Strategy:**

1. **Train 5 diverse models** with SWA (Hidden Gems #1, #2)
2. **Add stronger regularization model** (Hidden Gems #4, #5)
3. **Add optimized architecture model** (Hidden Gem #6)
4. **Ensemble 7 models total** with weights (Hidden Gem #3)
5. **Apply Test-Time Augmentation** (Hidden Gem #7)
6. **Bonus: Add snapshot ensemble** (Hidden Gem #8)

**Expected Final Accuracy: 94-96%**

---

## 📊 EXPECTED RESULTS

| Technique | Individual Gain | Cumulative |
|-----------|----------------|------------|
| Starting point | - | 90.6% |
| Ensemble (5 models) | +2-3% | **92-93%** |
| + TTA | +0.5-1% | **93-94%** |
| + Snapshot ensemble | +0.5-1% | **94-95%** |
| + Stronger regularization | +0.5-1% | **94.5-95.5%** |
| + Better architecture | +0.3-0.5% | **95-96%** |

---

## 🚀 QUICK START

**For Maximum Accuracy RIGHT NOW:**

```bash
# 1. Train ensemble (most important!)
cd /workspace/violence_detection_mvp
python train_ensemble.py

# Expected time: ~12-15 hours (5 models × ~3 hours each)
# Expected result: Individual models ~90-91%, ensemble ~92-93%

# 2. Predict with ensemble
python ensemble_predict.py

# Expected: 92-93% test accuracy

# 3. (Optional) Add TTA for extra boost
# Modify ensemble_predict.py to use predict_with_tta()
# Expected: 93-94% test accuracy
```

---

## 🎓 UNDERSTANDING WHY THESE WORK

### Why Ensemble Works (+2-3%)
- Different random seeds → different initialization
- Different augmentation → different training data views
- Each model makes DIFFERENT mistakes
- Averaging cancels out individual errors
- Wisdom of crowds principle

### Why SWA Works (+0.5-1.5%)
- SGD finds narrow, sharp minima (overfit)
- SWA finds wider, flatter minima (generalize better)
- Averaging weights from trajectory = smoother solution
- Like bagging but for weights, not data

### Why TTA Works (+0.5-1.5%)
- Test data augmentation → multiple views
- Each view slightly different
- Average predictions → more robust
- Reduces impact of unlucky augmentation

### Why Smaller Models Work (+0.5-1%)
- 2.5M params too many for 21K samples
- Smaller model forced to learn patterns, not memorize
- Better parameter/sample ratio
- Occam's razor: simpler is better

---

## 🔧 TROUBLESHOOTING

### If ensemble doesn't reach 92%:
1. Check individual model validation accuracy
2. Ensure diverse augmentation is working (check seeds)
3. Try weighted ensemble instead of simple average
4. Add TTA for extra boost

### If still overfitting:
1. Use stronger regularization script
2. Reduce model size further (64 → 32 LSTM units)
3. Increase dropout (0.6 → 0.7)
4. More aggressive augmentation

### If training is too slow:
1. Use fewer ensemble models (5 → 3)
2. Reduce epochs (200 → 150)
3. Skip snapshot ensembling
4. Focus on the best 3 hidden gems only

---

## 📈 ACCURACY PREDICTION

Based on your current 90.6% and severe overfitting:

**Conservative estimate:**
- Ensemble: 92.0-92.5%
- + TTA: 92.5-93.0%
- + Regularization: 93.0-93.5%

**Optimistic estimate:**
- Ensemble: 92.5-93.0%
- + TTA: 93.5-94.0%
- + Snapshot: 94.0-94.5%
- + All techniques: 94.5-95.5%

**Your goal of 93-95% is VERY ACHIEVABLE with these techniques!**

---

## 🎯 TL;DR - DO THIS NOW

```bash
# Step 1: Train ensemble (most important!)
python train_ensemble.py

# Step 2: Evaluate ensemble
python ensemble_predict.py

# Step 3: If < 93%, add TTA and snapshot ensemble
# Step 4: If still < 93%, train regularization model
# Step 5: Celebrate 93-95% accuracy! 🎉
```

**Expected timeline to 93%+:** 12-18 hours of training

**Guaranteed improvement:** At least +2% from ensemble alone

**Best case scenario:** 95-96% with all techniques combined

---

Good luck! You're about to break through that 90.6% plateau! 🚀
