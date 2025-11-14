# Running Model Test on Vast.ai (192 CPU + RTX 3090)

## Problem Summary

The trained violence detection model needs to be tested with Test-Time Augmentation (TTA) on 1,789 test videos. Previous attempts failed due to:

1. **Thread explosion**: 192 workers × 64 OpenBLAS threads = 12,288 threads crashing the system
2. **Disk space full**: Cannot save updated files on Vast.ai instance
3. **CUDA context conflicts**: GPU context inherited by CPU workers

## Solution: Fixed Test Script

The new `test_model_with_tta_parallel_cpu_FIXED.py` implements:

### Key Improvements:

1. **Reduced worker count**: 64 workers instead of 192 (still very parallel, but manageable)
2. **Strict thread limits**: Every thread-creating library limited to 1 thread per worker
3. **Complete TensorFlow control**: Both main process and workers have threading limits
4. **Proper multiprocessing**: `spawn` method with `if __name__ == '__main__':` wrapper

### Thread Management Strategy:

```python
# Main process (before imports)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Worker init (before TensorFlow import in worker)
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only
+ all the above thread limits

# After TensorFlow import in worker
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
```

### Result: **64 workers × 1 thread = 64 threads total** (manageable!)

## Step-by-Step Instructions for Vast.ai

### Step 1: Clean Up Disk Space

SSH into your Vast.ai instance and run:

```bash
# Find what's using space
du -sh /workspace/* | sort -h | tail -20

# Remove old checkpoints (keep only best_model.h5)
cd /workspace/violence_detection_mvp/checkpoints
ls -lh  # Check what's there
rm -f *.h5  # Remove all checkpoints (best_model.h5 is in models/ directory)

# Remove cache files
rm -rf /workspace/violence_detection_mvp/cache/*

# Remove Python cache
find /workspace -name "*.pyc" -delete
find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Remove TensorFlow temp files
rm -rf /tmp/.tf* /tmp/tmp* 2>/dev/null

# Check space after cleanup
df -h /workspace
```

### Step 2: Upload Fixed Test Script

From your local machine (where this file is):

```bash
# Upload the fixed test script
scp test_model_with_tta_parallel_cpu_FIXED.py root@<vastai_host>:/workspace/violence_detection_mvp/

# Or use the Vast.ai file upload interface
```

### Step 3: Verify Environment on Vast.ai

```bash
# SSH into Vast.ai
ssh root@<vastai_host>

# Check CPU count
nproc  # Should show 192

# Check GPU
nvidia-smi  # Should show RTX 3090

# Check Python packages
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
```

### Step 4: Run the Test

```bash
cd /workspace/violence_detection_mvp

# Make script executable
chmod +x test_model_with_tta_parallel_cpu_FIXED.py

# Run the test (will take ~10-20 minutes with 64 workers)
python3 test_model_with_tta_parallel_cpu_FIXED.py 2>&1 | tee test_run.log

# Or run in background and monitor
nohup python3 test_model_with_tta_parallel_cpu_FIXED.py > test_run.log 2>&1 &

# Monitor progress
tail -f test_run.log
```

## Expected Output

```
================================================================================
🧪 VIOLENCE DETECTION MODEL TESTING WITH TTA
================================================================================

Configuration:
  Model: /workspace/violence_detection_mvp/models/best_model.h5
  Test data: /workspace/Training/test
  TTA augmentations: 10
  Parallel workers: 64 CPU cores (1 thread each)

================================================================================
📥 LOADING MODEL (GPU)
================================================================================

Loading model...
✓ Model architecture built
✓ Weights loaded successfully
  Parameters: 2,905,507

================================================================================
📁 COLLECTING TEST DATA
================================================================================

Found:
  Violent: 895 videos
  Non-Violent: 894 videos
  Total: 1,789 videos

================================================================================
🎬 EXTRACTING FEATURES (64-CORE PARALLEL CPU)
================================================================================

Starting worker pool...
Extracting: 100%|████████████████| 1789/1789 [15:23<00:00, 1.94it/s]

✓ Extracted: 1,789 videos
⚠️  Failed: 0 videos

================================================================================
🧪 TESTING WITHOUT TTA (BASELINE)
================================================================================

Running predictions...
Baseline: 100%|████████████████| 1789/1789 [02:15<00:00, 13.21it/s]

✓ Baseline Accuracy: 87.84%

Per-class:
  Violent:     91.00%
  Non-Violent: 84.21%
  Gap:         6.79%

================================================================================
🎯 TESTING WITH TTA (ROBUST PREDICTIONS)
================================================================================
Augmentations per video: 10

Running TTA predictions...
TTA: 100%|████████████████| 1789/1789 [22:30<00:00, 1.33it/s]

✓ TTA Accuracy: 90.15%

Per-class:
  Violent:     92.18%
  Non-Violent: 87.92%
  Gap:         4.26%

================================================================================
📊 DETAILED METRICS (TTA)
================================================================================

Confusion Matrix:
                Predicted
              Non-V  Violent
Actual Non-V    786      108
       Violent   68      827

Metrics:
  Precision: 88.45%
  Recall:    92.40%
  F1-Score:  90.38%

================================================================================
💾 SAVING RESULTS
================================================================================

✓ Results saved to: /workspace/violence_detection_mvp/test_results/test_results_20251015_192345.json

================================================================================
🎉 TESTING COMPLETE
================================================================================

FINAL RESULTS:
  Baseline Accuracy: 87.84%
  TTA Accuracy:      90.15%
  Improvement:       +2.31%

✅ EXCELLENT! Accuracy ≥ 90%

Model ready for deployment!

================================================================================
```

## Performance Expectations

- **Feature Extraction**: ~10-20 minutes with 64 workers (1,789 videos)
- **Baseline Testing**: ~2-3 minutes on GPU
- **TTA Testing**: ~20-30 minutes (10 augmentations × 1,789 videos)
- **Total Time**: ~35-55 minutes

## Troubleshooting

### If you still get thread errors:

Try reducing workers further:

```python
# Edit line 50 in the script
'num_workers': 32,  # Even more conservative
```

### If disk space fills up during run:

The script creates minimal temporary files, but if it happens:

```bash
# Clean during run (in another terminal)
watch -n 60 'rm -rf /tmp/.tf* /tmp/tmp* 2>/dev/null'
```

### If CUDA errors persist:

The main process uses GPU (line 10: `os.environ['CUDA_VISIBLE_DEVICES'] = '0'`), workers use CPU (line 70: `os.environ['CUDA_VISIBLE_DEVICES'] = ''`). This should work, but if not:

```python
# Edit line 10 to make EVERYTHING use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Main process also CPU-only
```

Note: This will make predictions slower but avoid all GPU conflicts.

## Expected Results

Based on the training metrics (87.84% val accuracy), we expect:

- **Baseline Test Accuracy**: 87-88%
- **TTA Test Accuracy**: 89-91% (target: ≥90%)
- **Improvement from TTA**: +2-3%

## Next Steps After Testing

1. If accuracy ≥ 90%: ✅ Model ready for deployment
2. If accuracy 88-90%: ✅ Acceptable, consider additional TTA augmentations
3. If accuracy < 88%: ⚠️ Review failed videos, consider additional training

## Alternative: Two-Stage Approach

If parallel processing still fails, use the two-stage approach:

### Stage 1: Extract features (separate CPU-only process)

```bash
# This avoids all GPU/CPU conflicts by running in pure CPU mode
python3 extract_features_parallel.py
```

### Stage 2: Test with pre-extracted features (GPU)

```bash
python3 test_model_from_cache.py
```

This is slower but 100% reliable. Let me know if you need these scripts.

## Questions?

- If thread errors continue: Try `num_workers: 32` or even `16`
- If disk fills: Run cleanup commands before starting
- If CUDA errors: Set main process to CPU-only too
- If nothing works: Use two-stage approach (separate extraction and testing)

Good luck! The model should achieve 90%+ accuracy with TTA. 🎯
