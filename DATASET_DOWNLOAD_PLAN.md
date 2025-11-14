# NexaraVision Dataset Download Plan

## 📊 Target: 50,000+ Videos for 93-98% Accuracy

### TIER 1: Core Violence Detection Datasets (11,400 videos, ~21 GB)
**Priority: HIGHEST** - Download first

| Dataset | Videos | Size | Kaggle ID | Description |
|---------|--------|------|-----------|-------------|
| RWF-2000 | 2,000 | 1.5 GB | `vulamnguyen/rwf2000` | Real-world fight videos from surveillance |
| UCF-Crime | 1,900 | 12.0 GB | `odins0n/ucf-crime-dataset` | Untrimmed surveillance with 13 anomaly types |
| SmartCity-CCTV | 4,000 | 3.5 GB | `toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd` | Smart city CCTV violence detection |
| Real-Life Violence | 2,000 | 2.0 GB | `mohamedmustafa/real-life-violence-situations-dataset` | Real-life violence situations |
| EAVDD | 1,500 | 1.8 GB | `arnab91/eavdd-violence` | Extended abnormal video dataset |

**TIER 1 Total:** 11,400 videos, 20.8 GB

---

### TIER 2: Extended Datasets (5,754 videos, ~46 GB)
**Priority: HIGH** - Download after Tier 1

| Dataset | Videos | Size | Kaggle ID | Description |
|---------|--------|------|-----------|-------------|
| XD-Violence | 4,754 | 45.0 GB | `nguhaduong/xd-violence-video-dataset` | Large-scale violence detection |
| Hockey Fight | 1,000 | 0.8 GB | `dataset/hockey-fight-detection` | Hockey fight videos |

**TIER 2 Total:** 5,754 videos, 45.8 GB

---

### TIER 3: Non-Violence / Normal Activity (20,086 videos, ~9 GB)
**Priority: MEDIUM** - For balanced dataset

| Dataset | Videos | Size | Kaggle ID | Description |
|---------|--------|------|-----------|-------------|
| UCF-101 Normal | 13,320 | 7.0 GB | `dataset/ucf101` | Normal human activities (101 classes) |
| HMDB-51 | 6,766 | 2.0 GB | `dataset/hmdb51` | Normal human motion activities |

**TIER 3 Total:** 20,086 videos, 9.0 GB

---

## 📈 Combined Statistics

| Metric | Value |
|--------|-------|
| **Total Videos** | **37,240** |
| **Total Size** | **75.6 GB** |
| **Violence Videos** | ~17,000 |
| **Non-Violence Videos** | ~20,000 |
| **Expected Accuracy** | **93-97%** |

---

## ⏱️ Estimated Download Times

**With 1 Gbps connection:**
- Tier 1: ~3-5 minutes
- Tier 2: ~7-10 minutes
- Tier 3: ~2-3 minutes
- **Total:** ~15-20 minutes

**With 100 Mbps connection:**
- Tier 1: ~30-50 minutes
- Tier 2: ~60-90 minutes
- Tier 3: ~20-30 minutes
- **Total:** ~2-3 hours

---

## 🚀 Download Strategy

### Phase 1: Core Datasets (Week 1)
```bash
# Start with TIER 1 - highest priority
python3 VASTAI_DOWNLOAD_DATASETS.py
```

**What gets downloaded:**
- 11,400 violence videos
- All critical surveillance-style datasets
- Foundation for initial model training

### Phase 2: Extended Datasets (Week 1-2)
**After Tier 1 completes**, continue with:
- XD-Violence (large but comprehensive)
- Hockey Fight Detection

### Phase 3: Balance Dataset (Week 2)
**Final step:**
- UCF-101 Normal activities
- HMDB-51 for additional non-violence examples

---

## 💾 Storage Requirements

**Vast.ai Instance Recommendations:**
- **Minimum:** 100 GB storage
- **Recommended:** 200 GB storage (for processed data + models)
- **Optimal:** 500 GB+ storage (for checkpoints + experiments)

**Directory Structure:**
```
/workspace/
├── datasets/
│   ├── tier1/           # 21 GB (core violence)
│   ├── tier2/           # 46 GB (extended violence)
│   ├── tier3/           # 9 GB (non-violence)
│   ├── processed/       # ~50 GB (preprocessed frames)
│   └── cache/           # ~10 GB (temporary)
├── models/              # ~5 GB (trained models)
├── checkpoints/         # ~20 GB (training checkpoints)
└── logs/                # ~1 GB (training logs)
```

**Total Required:** ~162 GB minimum, **200 GB+ recommended**

---

## 📋 Download Monitoring

### Check download progress:
```bash
# Overall size
watch -n 5 'du -sh /workspace/datasets/*'

# File count
watch -n 10 'find /workspace/datasets -name "*.mp4" | wc -l'

# Detailed breakdown
tree -L 2 /workspace/datasets/
```

### Monitor network:
```bash
# Download speed
nethogs

# Bandwidth usage
iftop
```

---

## ✅ Verification Checklist

After download completes:

- [ ] **Tier 1:** 11,000+ videos in `/workspace/datasets/tier1/`
- [ ] **Tier 2:** 5,500+ videos in `/workspace/datasets/tier2/`
- [ ] **Tier 3:** 19,000+ videos in `/workspace/datasets/tier3/`
- [ ] **Total Size:** ~75 GB used
- [ ] **No corrupted files:** Run integrity check
- [ ] **Download log:** Check `/workspace/datasets/download_results.json`

---

## 🔄 If Downloads Fail

**Kaggle API errors:**
```bash
# Reconfigure credentials
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"issadalu","key":"5aabafacbfdefea1bf4f2171d98cc52b"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

**Network timeout:**
```bash
# Resume download (script will skip completed datasets)
python3 VASTAI_DOWNLOAD_DATASETS.py
```

**Corrupted files:**
```bash
# Remove and re-download specific dataset
rm -rf /workspace/datasets/tier1/RWF_2000
python3 VASTAI_DOWNLOAD_DATASETS.py
```

---

## 📞 Support

**Check PROGRESS.md:** Complete implementation guide
**GPU Issues:** Run `nvidia-smi`
**Disk Space:** Run `df -h`
**Network:** Run `speedtest-cli`
