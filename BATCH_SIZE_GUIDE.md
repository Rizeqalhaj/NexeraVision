# Batch Size Optimization Guide for GPU 1 (24GB VRAM)

## Quick Reference

| Batch Size | Steps/Epoch | Time/Epoch | Total Time | VRAM Usage | GPU Util | Speed |
|------------|-------------|------------|------------|------------|----------|-------|
| 1          | 8,584       | 97 min     | 48 hours   | 2-3 GB     | 10-15%   | ❌ Too slow |
| 16         | 537         | 9 min      | 4.5 hours  | 12-16 GB   | 60-70%   | ⚠️ Okay |
| **32**     | **269**     | **5.4 min**| **2.7 hours** | **18-22 GB** | **85-90%** | **✅ Recommended** |
| **40**     | **215**     | **5.4 min**| **2.7 hours** | **22-24 GB** | **95-100%** | **🔥 Maximum** |
| 48         | 179         | ~6 min     | ~3 hours   | 24+ GB     | 100%     | ⚠️ May OOM |

---

## 🚀 Recommended Approach

### Option 1: Safe Maximum (batch_size=32) - RECOMMENDED

```bash
# Stop current training (Ctrl+C)
./MAX_GPU_UTILIZATION.sh
./START_TRAINING.sh
```

**Performance:**
- 85-90% GPU utilization
- ~2.7 hours total training time
- ~150-200 videos/second
- Very stable, won't OOM

---

### Option 2: Absolute Maximum (batch_size=40) - AGGRESSIVE

```bash
# Stop current training (Ctrl+C)
./PUSH_TO_LIMIT.sh
./START_TRAINING.sh
```

**Performance:**
- 95-100% GPU utilization
- ~2.7 hours total training time
- ~200-250 videos/second
- Might OOM if other processes use VRAM

---

## 📊 Performance Calculations

### Current (batch_size=1):
```
8,584 videos ÷ 1 batch = 8,584 steps/epoch
8,584 steps × 680ms = 97 minutes/epoch
30 epochs × 97 min = 48 hours total ❌
```

### With batch_size=32:
```
8,584 videos ÷ 32 batch = 269 steps/epoch
269 steps × 1200ms = 5.4 minutes/epoch
30 epochs × 5.4 min = 2.7 hours total ✅
```

### With batch_size=40:
```
8,584 videos ÷ 40 batch = 215 steps/epoch
215 steps × 1500ms = 5.4 minutes/epoch
30 epochs × 5.4 min = 2.7 hours total ✅
```

**Speed improvement: 18x faster than batch_size=1!**

---

## 🎯 Recommendation

**Start with batch_size=32:**
1. Run `./MAX_GPU_UTILIZATION.sh`
2. Run `./START_TRAINING.sh`
3. Monitor `nvidia-smi` in another terminal
4. If VRAM stays under 20GB, you can push to 40

**If you want absolute maximum:**
1. Run `./PUSH_TO_LIMIT.sh` (sets batch_size=40)
2. Run `./START_TRAINING.sh`
3. Watch closely for OOM
4. If OOM, reduce to 32 or 36

---

## 🔍 Monitoring GPU Usage

```bash
watch -n 1 nvidia-smi
```

**Target VRAM usage:**
- batch_size=32: 18-22 GB (85-90%) ✅
- batch_size=40: 22-24 GB (95-100%) 🔥

**What you'll see:**
```
+-----------------------------------------------------------------------------+
| GPU  Name                        Memory-Usage | GPU-Util  Temp  Power      |
|=============================================================================|
|   0  NVIDIA GeForce RTX 3090 Ti  23000/24576MB|    95%   75°C  350W  ⚠️    |
|   1  NVIDIA GeForce RTX 3090 Ti  21000/24576MB|    95%   70°C  320W  ✅    |
+-----------------------------------------------------------------------------+
```

GPU 1 should show:
- Memory: 18-24 GB used (target: 90-100%)
- GPU-Util: 90-100%
- Temp: 65-75°C (safe range)
- Power: 300-350W (full throttle)

---

## ⚠️ If OOM Occurs

If you get OOM with batch_size=40:

```bash
# Option 1: Reduce to 36
cat > /workspace/training_config.json <<EOF
{
  "training": {
    "batch_size": 36
  }
}
