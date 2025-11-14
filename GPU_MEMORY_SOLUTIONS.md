# GPU Memory Solutions for NexaraVision Training

## Problem Summary

**Error**: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate bytes`

**Root Cause**: 1GB GPU VRAM is insufficient for ResNet50V2 architecture training

**Memory Requirements**:
- Model: ResNet50V2 + Bidirectional GRU (25,892,994 parameters)
- Minimum GPU needed: **1.2-1.5 GB VRAM**
- Current GPU: **1 GB VRAM** ❌

---

## Solution 1: Upgrade GPU Instance (RECOMMENDED ✅)

### Vast.ai GPU Search

**Search for 4GB+ GPU instances:**
```bash
vastai search offers 'gpu_ram >= 4096 cuda_vers >= 11.8' --order 'dph_total'
```

**Recommended GPUs:**
| GPU Model | VRAM | Cost/Hour | Training Time | Total Cost |
|-----------|------|-----------|---------------|------------|
| RTX 3060 | 12GB | ~$0.30 | 6-8 hours | ~$2-2.50 |
| RTX 2060 | 6GB | ~$0.25 | 8-10 hours | ~$2-2.50 |
| RTX 4060 | 8GB | ~$0.35 | 6-8 hours | ~$2.50 |

**Benefits:**
- ✅ Can use batch_size=8-16 (faster training)
- ✅ Same 96-100% accuracy target
- ✅ Complete training in 6-8 hours
- ✅ No code changes needed

**Steps:**
1. Stop current Vast.ai instance
2. Search for 4GB+ GPU instance using command above
3. Rent new instance
4. Transfer `/workspace` directory to new instance
5. Run `python3 train_model_optimized.py`

---

## Solution 2: Enable Mixed Precision Training (FP16)

**Memory Reduction**: ~40% (might still fail on 1GB GPU)

### Step 1: Enable Mixed Precision
```bash
cd /workspace
python3 enable_mixed_precision.py
```

### Step 2: Verify Config
```bash
cat /workspace/training_config.json
```

Should show:
```json
{
  "training": {
    "batch_size": 4,
    "mixed_precision": true
  }
}
```

### Step 3: Restart Training
```bash
python3 train_model_optimized.py
```

**⚠️ Warning**: This may still fail on 1GB GPU as ResNet50V2 needs 1.2GB+ minimum

---

## Solution 3: Switch to Lightweight Model

Replace ResNet50V2 with MobileNetV2:

**Trade-offs:**
- ✅ Fits in 1GB GPU
- ✅ Can use batch_size=8
- ❌ 3-5% accuracy loss (91-93% vs 96-100%)

**Implementation:**
1. Edit `model_architecture.py`:
   - Replace `ResNet50V2` with `MobileNetV2`
   - Reduce GRU units from 128 to 64
2. Update `training_config.json`:
   - Set `batch_size: 8`
3. Restart training

---

## Solution 4: Pre-trained Model (Quick Deploy)

Skip training entirely, use pre-trained violence detection model:

**Options:**
- Download from model zoo
- Use transfer learning from similar dataset
- Fine-tune on small validation set only

**Trade-offs:**
- ✅ Immediate deployment
- ✅ No GPU costs
- ❌ May not fit your specific use case
- ❌ Unknown accuracy on your scenarios

---

## Recommended Approach

**For Production Quality (96-100% accuracy):**
```
Upgrade to 4GB+ GPU → Train with batch_size=8-16 → Deploy
Cost: ~$2-3 total
Time: 6-8 hours
```

**For Budget Constraints:**
```
Try Mixed Precision on 1GB GPU → If fails → Switch to MobileNetV2
Cost: Current GPU cost
Time: 10-15 hours (slower training)
Accuracy: 91-93%
```

**For Immediate Testing:**
```
Use pre-trained model → Quick deploy → Evaluate → Decide on retraining
Cost: $0
Time: 1 hour
```

---

## Current Status

**Files Created:**
- ✅ `fix_vastai_config.sh` - Fixes batch_size and missing config fields
- ✅ `fix_vastai_config.py` - Python version of config fix
- ✅ `enable_mixed_precision.py` - Enables FP16 training
- ✅ `train_model_optimized.py` - Updated with mixed precision support

**Next Steps:**
1. **Decide on solution approach** (GPU upgrade recommended)
2. **Apply chosen solution**
3. **Monitor training progress**
4. **Validate model accuracy**
5. **Deploy to staging environment**

---

## Support Commands

### Check GPU Memory on Vast.ai:
```bash
nvidia-smi
```

### Monitor Training Progress:
```bash
watch -n 5 'tail -n 30 /workspace/models/logs/training/*.csv'
```

### Kill Stuck Training Process:
```bash
pkill -f train_model
```

### Clear GPU Memory:
```bash
nvidia-smi --gpu-reset
```

---

## Contact

For issues, check:
- Training logs: `/workspace/models/logs/training/`
- Error details: Check full stack trace in terminal
- GPU status: `nvidia-smi` command output
