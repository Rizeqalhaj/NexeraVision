# A100 Quick Start Guide

## Installation (5 minutes)

```bash
cd /home/admin/Desktop/NexaraVision
bash COMPLETE_A100_INSTALL.sh
```

Wait for completion, then verify:

```bash
python3 /tmp/test_training_setup.py
```

## Expected Output

```
[1] TensorFlow + GPU: ✅ TensorFlow 2.15.x, 1 GPU detected
[2] ResNet50V2 Model: ✅ Loaded (23.5M params)
[3] BiGRU Layer: ✅ Functional
[4] OpenCV: ✅ Version 4.8.x
[5] Data Pipeline: ✅ Working
[6] Memory Estimation: 80GB total, recommended batch size: 32-48

✅ ALL TESTS PASSED - Environment ready for training
```

## Start Training

```bash
# Monitor GPU in one terminal
watch -n 1 nvidia-smi

# Run training in another terminal
python3 scripts/train_model.py --batch-size 32 --epochs 50
```

## GPU Configuration (Add to training script)

```python
import tensorflow as tf

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Enable mixed precision (2-3x speedup)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

## Performance Tips

| Optimization | Speedup | How to Enable |
|--------------|---------|---------------|
| Mixed Precision | 2-3x | `mixed_precision.set_global_policy('mixed_float16')` |
| XLA Compilation | 1.3x | `model.compile(..., jit_compile=True)` |
| Batch Size 48 | 1.4x | `--batch-size 48` |
| **Combined** | **~4x** | All of the above |

## Batch Size Guide

- **Start**: 32 (safe, ~45GB)
- **Optimal**: 48 (best performance, ~65GB)
- **Maximum**: 64 (aggressive, ~75GB)

Monitor with: `watch -n 1 nvidia-smi`

## Troubleshooting

### GPU not detected
```bash
nvidia-smi  # Should show A100
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Out of memory
```bash
# Reduce batch size
python3 scripts/train_model.py --batch-size 16
```

### Slow training
```bash
# Enable optimizations in training script:
# 1. Mixed precision: mixed_precision.set_global_policy('mixed_float16')
# 2. XLA: model.compile(..., jit_compile=True)
# 3. Increase batch size: --batch-size 48
```

## Complete Documentation

- **Full Setup Guide**: `A100_SETUP_README.md`
- **Installation Methods**: `INSTALLATION_GUIDE.md`
- **Project Workflow**: `COMPLETE_WORKFLOW.md`

---

**Ready in 5 minutes. Train in 2 hours with mixed precision.**
