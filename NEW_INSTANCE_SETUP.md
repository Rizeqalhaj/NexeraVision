# NexaraVision - New Instance Setup Guide

## ✅ Pre-Requisites (ALREADY DONE)

- [x] SSH Key Generated: `~/.ssh/id_rsa.pub`
- [x] SSH Key Added to Vast.ai Account: https://cloud.vast.ai/manage-keys/
- [x] Kaggle Credentials Ready
- [x] Download Scripts Prepared

---

## 🚀 Step 1: Create New Instance

**Go to**: https://cloud.vast.ai/create/

**Recommended Specs**:
- **GPU**: RTX 4070/4090 (12GB+ VRAM) or RTX 3080
- **RAM**: 32GB+ (64GB+ preferred)
- **Storage**: 200GB+ (for 75GB of datasets + processing)
- **Template**: PyTorch or TensorFlow with CUDA 12.4

**IMPORTANT**: Make sure your SSH key is already added to your account **BEFORE** creating the instance!

**Verify**: https://cloud.vast.ai/manage-keys/ should show:
```
SHA256:0i+lHI8aa9zy0XqloZP7rBmsomPQHDjYLuaOOD/CBRU nexaravision-vastai
```

---

## 🔌 Step 2: Get New Connection Details

After instance is created, get the new SSH details:

1. Go to: https://cloud.vast.ai/instances/
2. Click **"Connect"** on your instance
3. Copy the SSH command (will be something like):
   ```
   ssh -p XXXXX root@XX.XX.XX.XX
   ```

---

## ✅ Step 3: Claude Will SSH In

Once you have the new SSH command, paste it here and I will:

1. ✅ SSH into the instance
2. ✅ Setup workspace directories
3. ✅ Configure Kaggle credentials
4. ✅ Download all 5 datasets (11,400 videos, 21GB)
5. ✅ Monitor progress in real-time
6. ✅ Update PROGRESS.md with results

---

## 📋 Quick Setup Script (Claude will run this via SSH)

```bash
#!/bin/bash
# NexaraVision Complete Setup

echo "=========================================="
echo "NexaraVision Setup - Starting"
echo "=========================================="

# Install dependencies
pip3 install --quiet kaggle

# Create workspace structure
mkdir -p /workspace/datasets/{tier1,tier2,tier3,processed}
mkdir -p /workspace/{models,logs,checkpoints,scripts}

# Configure Kaggle
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"issadalu","key":"5aabafacbfdefea1bf4f2171d98cc52b"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "✅ Setup Complete!"
echo ""
echo "Starting Dataset Downloads..."
echo ""

# Download datasets
python3 << 'PYTHON_EOF'
import subprocess, os, json, sys
from pathlib import Path
from datetime import datetime

datasets = [
    ('vulamnguyen/rwf2000', 'tier1/RWF2000', 'RWF-2000', 2000, 1.5),
    ('odins0n/ucf-crime-dataset', 'tier1/UCF_Crime', 'UCF-Crime', 1900, 12.0),
    ('toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd', 'tier1/SCVD', 'SmartCity-CCTV', 4000, 3.5),
    ('mohamedmustafa/real-life-violence-situations-dataset', 'tier1/RealLife', 'Real-Life Violence', 2000, 2.0),
    ('arnab91/eavdd-violence', 'tier1/EAVDD', 'EAVDD', 1500, 1.8),
]

print("="*80)
print("NexaraVision Dataset Download")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
sys.stdout.flush()

results = []
for i, (kid, path, name, vids, size) in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(datasets)}] {name} ({vids:,} videos, ~{size}GB)")
    print(f"{'='*80}")
    sys.stdout.flush()

    out = Path(f"/workspace/datasets/{path}")
    out.mkdir(parents=True, exist_ok=True)

    try:
        start = datetime.now()
        print(f"⏳ Downloading...")
        sys.stdout.flush()

        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', kid, '-p', str(out), '--unzip'],
            capture_output=True,
            text=True,
            timeout=3600
        )

        if result.returncode == 0:
            vcnt = sum(len(list(out.rglob(f'*.{ext}'))) for ext in ['mp4','avi','mkv','webm','mov','MP4','AVI','MKV'])
            sz = sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / (1024**3)
            elapsed = (datetime.now() - start).total_seconds()

            print(f"✅ SUCCESS!")
            print(f"   Videos: {vcnt:,}")
            print(f"   Size: {sz:.2f} GB")
            print(f"   Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            sys.stdout.flush()

            results.append({
                'name': name,
                'status': 'success',
                'videos': vcnt,
                'size_gb': round(sz, 2),
                'time_s': round(elapsed, 1)
            })
        else:
            print(f"❌ FAILED: {result.stderr[:300]}")
            sys.stdout.flush()
            results.append({'name': name, 'status': 'failed', 'error': result.stderr[:200]})

    except Exception as e:
        print(f"❌ ERROR: {str(e)[:300]}")
        sys.stdout.flush()
        results.append({'name': name, 'status': 'error', 'error': str(e)[:200]})

# Save results
with open('/workspace/datasets/download_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'results': results
    }, f, indent=2)

# Summary
successful = [r for r in results if r['status'] == 'success']
failed = [r for r in results if r['status'] != 'success']

print(f"\n{'='*80}")
print(f"📊 DOWNLOAD SUMMARY")
print(f"{'='*80}")
print(f"\n✅ Successful: {len(successful)}/{len(results)} datasets")
print(f"📹 Total Videos: {sum(r.get('videos',0) for r in successful):,}")
print(f"💾 Total Size: {sum(r.get('size_gb',0) for r in successful):.2f} GB")
print(f"⏱️  Total Time: {sum(r.get('time_s',0) for r in successful)/60:.1f} minutes")

if failed:
    print(f"\n❌ Failed: {len(failed)} datasets")
    for r in failed:
        print(f"   - {r['name']}: {r.get('error', 'Unknown')[:100]}")

print(f"\n📄 Results: /workspace/datasets/download_results.json")
print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

print(f"\n💾 Disk Usage:")
os.system("du -sh /workspace/datasets/*")
os.system("df -h /workspace | tail -1")

PYTHON_EOF

echo ""
echo "=========================================="
echo "✅ NexaraVision Setup Complete!"
echo "=========================================="
```

---

## 🎯 Expected Timeline

1. **Delete old instance**: 1 minute
2. **Create new instance**: 2-5 minutes
3. **SSH connection**: Immediate (key is pre-configured)
4. **Setup + Downloads**: 15-30 minutes

**Total**: ~20-40 minutes to full operation

---

## 📊 What You'll See

```
==========================================
NexaraVision Setup - Starting
==========================================
✅ Setup Complete!

Starting Dataset Downloads...

================================================================================
NexaraVision Dataset Download
Start: 2025-11-14 XX:XX:XX
================================================================================

================================================================================
[1/5] RWF-2000 (2,000 videos, ~1.5GB)
================================================================================
⏳ Downloading...
✅ SUCCESS!
   Videos: 2,000
   Size: 1.52 GB
   Time: 45.3s (0.8 min)

[... continues for all 5 datasets ...]

================================================================================
📊 DOWNLOAD SUMMARY
================================================================================

✅ Successful: 5/5 datasets
📹 Total Videos: 11,400
💾 Total Size: 20.82 GB
⏱️  Total Time: 28.5 minutes
```

---

**Ready when you are! Just:**
1. Delete the old instance
2. Create new instance (with SSH key already in account)
3. Tell me the new SSH command
4. I'll do everything else! 🚀
