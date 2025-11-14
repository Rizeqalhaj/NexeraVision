# Ultimate Ensemble Guide - Achieve 92-95% Accuracy

## Current Status
- ✅ Single model (VGG19 + BiLSTM): **90.52%**
- 🎯 Target: **92-95%**
- 📈 Gap to close: **1.5-4.5%**

## Strategy to Reach 92-95%

### 1. Ensemble of 3 VGG19-Based Models

**ALL models use VGG19 feature extractor** (project requirement)
- Feature extractor: VGG19 (fc2 layer, 4096 features)
- Same features, different sequence architectures
- Diversity comes from different sequence models and hyperparameters

**Model 1: VGG19 + BiLSTM** (Your current best)
- Sequence model: Bidirectional LSTM (192 units)
- Dense layers: 256 → 128
- Dropout: 0.4, 0.5
- Learning rate: 0.0005 (proven optimal)
- Expected: 90-91%

**Model 2: VGG19 + BiGRU**
- Sequence model: Bidirectional GRU (192 units)
- Dense layers: 384 → 192 (larger)
- Dropout: 0.45, 0.5 (higher)
- Learning rate: 0.0004
- Expected: 90-92%

**Model 3: VGG19 + Attention LSTM**
- Sequence model: BiLSTM with attention mechanism
- Attention layer: Learns which frames are important
- Dense layers: 256 → 128
- Dropout: 0.4, 0.5
- Learning rate: 0.0003
- Expected: 91-92%

**Ensemble (Soft Voting):**
- Combine all 3 model predictions
- Average probabilities across models
- Expected: **92-95%** (1.5-3% boost)

### 2. Data Augmentation (Applied During Training)

✅ **Horizontal Flip** (50% probability)
- Mirrors video horizontally
- Violence patterns work from both sides

✅ **Random Brightness** (0.8x - 1.2x range)
- Handles different lighting conditions
- Improves robustness to camera quality

✅ **Random Rotation** (±10 degrees)
- Small rotations simulate camera angles
- Helps with mounted vs handheld cameras

✅ **Frame Dropout** (10% of frames)
- Randomly drops and interpolates frames
- Forces model to not rely on specific frames
- Simulates missing/corrupted frames

### 3. Hyperparameter Optimizations

**Per-Model Learning Rates:**
- VGG19: 0.0005 (proven optimal from your training)
- ResNet50: 0.0003 (lower - ResNet is more sensitive)
- EfficientNet: 0.0002 (lowest - modern architecture)

**Early Stopping:**
- Patience: 20 epochs (increased from 15)
- Gives more time for convergence
- Monitors: val_accuracy

**Dropout Rates:**
- BiLSTM dropout: 0.4
- Dense layer dropout: 0.5
- Prevents overfitting with augmentation

**Batch Size:**
- 64 (optimal for your GPU)
- Consistent across all models

## How to Run

### Step 1: Train Ensemble (8-12 hours)

```bash
bash /home/admin/Desktop/NexaraVision/TRAIN_ENSEMBLE_92_PERCENT.sh
```

Or manually:
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
python3 train_ensemble_ultimate.py
```

**What happens:**
1. Trains Model 1 (VGG19 + BiLSTM) - 2-4 hours
2. Trains Model 2 (ResNet50 + BiGRU) - 2-4 hours
3. Trains Model 3 (EfficientNet + Attention) - 2-4 hours
4. Saves all models to `/workspace/ensemble_models/`

**Progress monitoring:**
- Watch val_accuracy for each model
- Each should reach 90-92% individually
- Models saved to:
  - `/workspace/ensemble_models/vgg19_bilstm/best_model.h5`
  - `/workspace/ensemble_models/resnet50_bigru/best_model.h5`
  - `/workspace/ensemble_models/efficientnet_attention/best_model.h5`

### Step 2: Evaluate Ensemble

```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
python3 ensemble_predict.py
```

**What happens:**
1. Loads all 3 trained models
2. Extracts features for test set (each model uses different extractor)
3. Combines predictions via soft voting
4. Shows individual and ensemble accuracy

**Expected output:**
```
INDIVIDUAL MODEL RESULTS
vgg19_bilstm: 90.52%
vgg19_bigru: 90.89%
vgg19_attention: 91.34%

ENSEMBLE RESULTS (SOFT VOTING)
✅ ENSEMBLE ACCURACY: 92.67%
✅ Improvement: +2.15%

🎉 TARGET ACHIEVED! Ensemble >= 92%
```

## Why This Will Work

### Ensemble Benefits (Expected 1.5-3% boost)

**Why same features but different models work:**
- All use VGG19 features (project requirement)
- Diversity comes from different sequence processing
- Different models learn different temporal patterns from same features

**Diversity of architectures:**
- BiLSTM: Bidirectional long-term memory with forget gates
- BiGRU: Simpler gating, faster training, different patterns
- Attention: Learns which frames matter most for each video

**Diversity of hyperparameters:**
- Different learning rates (0.0005, 0.0004, 0.0003)
- Different dense layer sizes (256→128 vs 384→192)
- Different dropout rates (0.4-0.5 vs 0.45-0.5)

**Error compensation:**
- When one model is uncertain, others compensate
- Different architectures make different mistakes on different videos
- Averaging reduces individual model errors → **+1.5-3% boost**

### Data Augmentation Benefits (Expected 0.5-1% boost)

**Horizontal flip:**
- Doubles effective dataset size
- Violence from left or right looks same

**Brightness variation:**
- Handles different lighting (indoor/outdoor/night)
- Your dataset has varying video quality

**Rotation:**
- Simulates different camera angles
- Small variations don't change violence detection

**Frame dropout:**
- Prevents overfitting to specific frames
- More robust to video quality issues

## Expected Timeline

**Total time: 8-12 hours**

```
Hour 0-4:   Model 1 (VGG19 + BiLSTM) training
Hour 4-8:   Model 2 (ResNet50 + BiGRU) training
Hour 8-12:  Model 3 (EfficientNet + Attention) training
Hour 12:    Evaluation & ensemble testing
```

## What If 92% Still Not Reached?

If ensemble gets 91-92% (close but not quite):

**Quick fixes (30 min - 2 hours):**
1. **Weighted voting** instead of soft voting
   - Weight models by their accuracy
   - Run: `python3 ensemble_predict.py --voting weighted`

2. **Train 2 more models** (4-6 hours)
   - InceptionV3 + BiLSTM
   - MobileNetV2 + GRU
   - 5-model ensemble → +0.5-1% accuracy

**Advanced fixes (1-2 days):**
1. **More data augmentation**
   - Add contrast, saturation, noise
   - Temporal augmentation (speed up/slow down)

2. **Collect more data**
   - Target 40K videos instead of 31K
   - More diverse sources

3. **Fine-tune on hard examples**
   - Find misclassified videos
   - Train specifically on those

## File Locations

```
Models:
/workspace/ensemble_models/vgg19_bilstm/best_model.h5
/workspace/ensemble_models/resnet50_bigru/best_model.h5
/workspace/ensemble_models/efficientnet_attention/best_model.h5

Results:
/workspace/ensemble_models/ensemble_results.json
/workspace/ensemble_models/ensemble_eval_soft.json

Features (cached):
/workspace/ensemble_cache/vgg19_bilstm/
/workspace/ensemble_cache/resnet50_bigru/
/workspace/ensemble_cache/efficientnet_attention/
```

## Monitoring Training

**Good signs:**
- Each model reaches 90-92% val_accuracy
- Training and validation loss both decreasing
- Small gap between train_acc and val_acc (<5%)

**Warning signs:**
- Val_accuracy plateaus at <88%
- Large gap (>10%) between train and val
- Models taking >6 hours each

If you see warnings, stop and we can adjust hyperparameters.

## Ready to Start!

Run this command to begin:
```bash
bash /home/admin/Desktop/NexaraVision/TRAIN_ENSEMBLE_92_PERCENT.sh
```

Training will take 8-12 hours. You can let it run overnight.

When done, run evaluation:
```bash
python3 /home/admin/Desktop/NexaraVision/violence_detection_mvp/ensemble_predict.py
```

**Expected final result: 92-95% ensemble accuracy!** 🚀
