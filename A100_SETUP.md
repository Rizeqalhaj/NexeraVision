# A100 Setup - Quick Start

## New Instance Details
- **GPU**: 1x A100 SXM4 (80GB VRAM)
- **IP**: 216.129.245.165
- **Instance ID**: 27895190
- **Performance**: 15.6 TFLOPS (MUCH faster than RTX 6000 Ada)

---

## Step 1: Upload Files to New Instance

**From local machine:**
```bash
cd /home/admin/Desktop/NexaraVision

scp A100_FAST_RESUME.py \
    model_architecture.py \
    data_preprocessing.py \
    CHECK_GPU.py \
    INSTALL_TRAINING_DEPENDENCIES.sh \
    root@216.129.245.165:/workspace/
```

---

## Step 2: Connect and Setup

```bash
# Connect to new A100 instance
ssh root@216.129.245.165

# Install dependencies
cd /workspace
bash INSTALL_TRAINING_DEPENDENCIES.sh

# Check GPU
python3 CHECK_GPU.py
```

**Expected output:**
```
✅ GPUs detected!
Physical GPUs: 1
  GPU 0: /physical_device:GPU:0 (A100-SXM4-80GB)
```

---

## Step 3: Upload Your Data

You need to upload:
1. Dataset: `/workspace/datasets/tier1/` (violence & non_violence folders)
2. Splits: `/workspace/processed/splits.json`
3. Checkpoints (if you have any): `/workspace/checkpoints/`

**From your old instance (if accessible):**
```bash
# From old instance, copy to local first
ssh root@195.142.145.66
cd /workspace
tar czf data.tar.gz datasets/ processed/ checkpoints/
exit

# Download to local
scp root@195.142.145.66:/workspace/data.tar.gz /tmp/

# Upload to new instance
scp /tmp/data.tar.gz root@216.129.245.165:/workspace/

# Extract on new instance
ssh root@216.129.245.165
cd /workspace
tar xzf data.tar.gz
```

---

## Step 4: Start Training

```bash
python3 /workspace/A100_FAST_RESUME.py
```

---

## A100 Advantages

✅ **3x faster** than RTX 6000 Ada
✅ **80GB VRAM** (can handle batch size 48+)
✅ **Tensor Cores** optimized for deep learning
✅ **No multi-GPU complexity**

**Expected speed:**
- Each epoch: **15-20 seconds** (vs 30s on 2x RTX 6000 Ada)
- Full training: **25-30 minutes** total

---

## Quick Commands

```bash
# Connect
ssh root@216.129.245.165

# Check GPU
python3 /workspace/CHECK_GPU.py

# Start training
python3 /workspace/A100_FAST_RESUME.py

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor training
tail -f /workspace/logs/training/initial_training_log.csv
```
