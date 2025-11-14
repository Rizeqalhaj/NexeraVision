# Recursive Reasoning Improvements for Violence Detection

Based on paper: **"Less is More: Recursive Reasoning with Tiny Networks"** (arXiv 2510.04871)

## Current Baseline Performance

- **Validation Accuracy**: 87.84%
  - Violent: 91.00%
  - Non-Violent: 84.21%
- **Model Size**: 2.9M parameters
- **Architecture**: VGG19 features → BiLSTM → Attention → Dense

## Paper's Key Insights

1. **Recursive Processing**: Small networks that process information multiple times
2. **Hierarchical Reasoning**: Multi-level reasoning at different frequencies
3. **Efficiency**: 7M parameters outperforms 100M+ models on complex tasks
4. **Iterative Refinement**: Models can "reconsider" decisions through iterations

## Applied Improvements to Violence Detection

### 1. Multi-Scale Temporal Processing (+2-3% expected)

**Problem**: Current model processes all frames at same temporal scale
**Solution**: Dual-path processing at different frequencies

```
Fast Path (Frame-level):
20 frames → BiLSTM(64) → Fine-grained motion details

Slow Path (Segment-level):
20 frames → Pool to 5 segments → BiLSTM(64) → Contextual patterns

Combined: Fast + Slow → Captures both micro (punches) and macro (escalation)
```

**Why it works**:
- Violence has multi-scale patterns
- Fast: Individual actions (punch, kick)
- Slow: Contextual buildup (argument → fight)

### 2. Recursive Refinement (+1-2% expected)

**Problem**: Single-pass processing can't reconsider decisions
**Solution**: 3 recursive iterations with residual connections

```
Initial State → LSTM Iteration 1 (+residual)
             → LSTM Iteration 2 (+residual)
             → LSTM Iteration 3 (+residual)
             → Final Decision
```

**Why it works**:
- First pass: Initial detection
- Second pass: Refine based on context
- Third pass: Final verification
- Reduces false positives by allowing model to "think twice"

### 3. Hierarchical Reasoning (+1-2% expected)

**Problem**: Model doesn't separate "activity detection" from "violence classification"
**Solution**: Two-level hierarchy

```
Level 1: Is there motion/activity? (Motion Detector)
         ↓
Level 2: If active, is it violent? (Violence Detector)
         ↓
Combined hierarchical decision
```

**Why it works**:
- Mimics human reasoning
- Reduces false positives on static/calm scenes
- Focuses violence detection only on active scenes

## Architecture Comparison

### Current Model (87.84%)
```
VGG19 Features (20, 4096)
    ↓
Dense(512) + Dropout
    ↓
BiLSTM(96) × 3 with residual
    ↓
Attention Pooling
    ↓
Dense(128) → Dense(64) → Dense(2)
```

### Recursive Reasoning Model (Expected: 92-95%)
```
VGG19 Features (20, 4096)
    ↓
╔═══════════════════════════════════════╗
║ MULTI-SCALE TEMPORAL PROCESSING       ║
╠═══════════════════════════════════════╣
║ Fast Path:    20 frames → BiLSTM(64) ║
║ Slow Path:    5 segments → BiLSTM(64)║
║ Combined:     Concatenate paths       ║
╚═══════════════════════════════════════╝
    ↓
╔═══════════════════════════════════════╗
║ RECURSIVE REFINEMENT (3 iterations)   ║
╠═══════════════════════════════════════╣
║ Iteration 1:  BiLSTM(48) +residual   ║
║ Iteration 2:  BiLSTM(48) +residual   ║
║ Iteration 3:  BiLSTM(48) +residual   ║
╚═══════════════════════════════════════╝
    ↓
Attention Pooling
    ↓
╔═══════════════════════════════════════╗
║ HIERARCHICAL REASONING                ║
╠═══════════════════════════════════════╣
║ Motion Branch:    Dense(64)           ║
║ Violence Branch:  Dense(128)          ║
║ Combined:         Concatenate         ║
╚═══════════════════════════════════════╝
    ↓
Dense(96) → Dense(2)
```

## Expected Performance Improvements

| Improvement | Accuracy Gain | Reasoning |
|-------------|---------------|-----------|
| Multi-scale processing | +2-3% | Captures both micro and macro patterns |
| Recursive refinement | +1-2% | Iterative decision refinement |
| Hierarchical reasoning | +1-2% | Separates motion from violence |
| **Total Expected** | **+4-7%** | Combined synergistic effects |

**Target Accuracy**: 92-95% (from baseline 87.84%)

## Model Size Comparison

| Model | Parameters | Accuracy |
|-------|------------|----------|
| Current Baseline | 2.9M | 87.84% |
| Recursive Reasoning | ~3.5M (+20%) | 92-95% (expected) |
| Paper's TRM | 7M | N/A (different task) |

Still in "tiny network" range while improving accuracy significantly.

## How to Train

### Prerequisites
Cached features must exist (from previous training):
```bash
/workspace/violence_detection_mvp/cache/
  ├── X_train.npy
  ├── y_train.npy
  ├── X_val.npy
  └── y_val.npy
```

### Training Command
```bash
cd /workspace/violence_detection_mvp
python3 train_recursive_reasoning.py 2>&1 | tee recursive_training.log
```

### Expected Training Time
- **With 192 CPU + RTX 3090**: ~2-3 hours (100 epochs with early stopping)
- **Expected convergence**: 60-80 epochs

### Outputs
- **Best model**: `/workspace/violence_detection_mvp/checkpoints/recursive_best.h5`
- **Final model**: `/workspace/violence_detection_mvp/models/recursive_reasoning_model.h5`
- **Training log**: `/workspace/violence_detection_mvp/checkpoints/recursive_training.log`

## Testing the New Model

After training, test with TTA:

### Step 1: Update test script to use new model

```python
# In test_tta_clean.py, update build_model() function
# Copy the recursive reasoning architecture
```

Or create a new test script specifically for recursive model.

### Step 2: Run TTA testing

```bash
python3 test_tta_clean.py  # Update model path to recursive_best.h5
```

## Implementation Timeline

### Phase 1: Training (2-3 hours)
1. Upload `train_recursive_reasoning.py` to Vast.ai
2. Verify cached features exist
3. Run training
4. Monitor for convergence

### Phase 2: Testing (30-45 minutes)
1. Load best recursive model
2. Run TTA testing
3. Compare with baseline

### Phase 3: Comparison
```
Baseline (Current):
- Accuracy: 87.84%
- Violent: 91.00%
- Non-Violent: 84.21%

Recursive Reasoning (Expected):
- Accuracy: 92-95%
- Violent: 93-96%
- Non-Violent: 90-93%
```

## Key Advantages

### 1. **Efficiency**
- Only +20% parameters (+600K)
- Still "tiny network" philosophy
- Fast inference (<50ms per video)

### 2. **Accuracy**
- +4-7% absolute improvement expected
- Reduces class imbalance gap
- Better generalization

### 3. **Interpretability**
- Clear reasoning stages (multi-scale → recursive → hierarchical)
- Can visualize attention at each recursive iteration
- Hierarchical branches show motion vs violence signals

### 4. **Robustness**
- Multi-scale: Handles different video speeds
- Recursive: Reduces false positives through refinement
- Hierarchical: Better on edge cases (calm scenes, sudden violence)

## Alternative: Lighter Version

If training is too slow, here's a lighter version:

### Quick Recursive Model (2.5M parameters)

```python
# Only recursive refinement (no multi-scale, no hierarchy)
Features → BiLSTM → Recursive(2 iterations) → Attention → Dense
```

Expected gain: +2-3% (simpler but still effective)

## Comparison to Paper's Results

| Metric | Paper (ARC-AGI) | Our Model (Violence) |
|--------|-----------------|----------------------|
| Task | Abstract reasoning | Video classification |
| Parameters | 7M | 3.5M |
| Key Innovation | Recursive reasoning | Applied to temporal video |
| Performance | 45% (complex puzzles) | 92-95% (expected) |
| Baseline beaten | Large LMs (100M+) | Single-pass models |

## Next Steps

1. ✅ Run current TTA test (`test_tta_clean.py`) to establish baseline
2. 🔄 Train recursive reasoning model
3. 🧪 Test recursive model with TTA
4. 📊 Compare results
5. 🚀 Deploy if >92% accuracy achieved

## Questions?

**Q: Why not just increase model size?**
A: Paper shows recursive reasoning beats larger models. Efficiency matters for deployment.

**Q: Will this work with our data?**
A: Yes - violence detection benefits from multi-scale temporal reasoning (micro actions + macro context).

**Q: Training time?**
A: ~2-3 hours on your Vast.ai setup (192 CPU + RTX 3090).

**Q: Can we combine with current model?**
A: Yes! Can ensemble both models for even higher accuracy.

## References

- Paper: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv 2510.04871)
- Our baseline: BiLSTM with attention (87.84%)
- Target: 92-95% with recursive reasoning
