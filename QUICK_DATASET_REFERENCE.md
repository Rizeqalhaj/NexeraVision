# Quick Violence Detection Dataset Reference

## 🚀 **INSTANT DOWNLOAD - NO REGISTRATION**

### Top 5 Recommended (Start Here):

| # | Dataset | Videos | Source | Download Command |
|---|---------|--------|--------|------------------|
| 1 | **RLVS** | 2,000 | Kaggle | `kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset` |
| 2 | **UCF Crime** | 1,900 | Kaggle | `kaggle datasets download -d odins0n/ucf-crime-dataset` |
| 3 | **XD-Violence** | 4,754 | Kaggle | `kaggle datasets download -d nguhaduong/xd-violence-video-dataset` |
| 4 | **AIRTLab** | 350 (HD) | GitHub | `git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git` |
| 5 | **ShanghaiTech** | 437 | Kaggle | `kaggle datasets download -d ravikagglex/shanghaitech-anomaly-detection` |

---

## 📋 **ALL FREE DATASETS (Kaggle)**

| Dataset | Videos | Type | Kaggle URL |
|---------|--------|------|------------|
| RLVS | 2,000 | Real fights | https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset |
| UCF Crime | 1,900 | 13 anomalies | https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset |
| XD-Violence | 4,754 | Audio+Video | https://www.kaggle.com/datasets/nguhaduong/xd-violence-video-dataset |
| ShanghaiTech | 437 | Campus | https://www.kaggle.com/datasets/ravikagglex/shanghaitech-anomaly-detection |
| SCVD | Various | CCTV | https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd |
| EAVDD | Extended | Violence | https://www.kaggle.com/datasets/arnab91/eavdd-violence |
| Movies | 1,000 | Film fights | https://www.kaggle.com/datasets/pratt3000/moviesviolencenonviolence |
| Audio Violence | Audio | Audio-only | https://www.kaggle.com/datasets/fangfangz/audio-based-violence-detection-dataset |
| UCF101 | 13,320 | Actions | https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition |

**Total Kaggle Videos: ~25,000+**

---

## 📋 **GITHUB DATASETS (git clone)**

| Dataset | Videos | GitHub URL |
|---------|--------|------------|
| AIRTLab | 350 (1920x1080) | https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos |
| VioPeru | 280 (Real CCTV) | https://github.com/hhuillcen/VioPeru |
| Fight Surv | 300 | https://github.com/seymanurakti/fight-detection-surv-dataset |
| UCA (CVPR 2024) | 1,854 | https://github.com/Xuange923/Surveillance-Video-Understanding |
| SCVD GitHub | CCTV | https://github.com/tolusophy/Violence_Detection |

---

## 🔐 **PREMIUM (Registration Required - Free for Academic)**

| Dataset | Videos | Registration URL |
|---------|--------|------------------|
| **RWF-2000** | 2,000 | Contact SMIIP Lab + sign agreement |
| **NTU CCTV-Fights** | 1,000 (8+ hours) | https://rose1.ntu.edu.sg/dataset/cctvFights/ |
| **VSD2014** | 32 movies + 86 videos | https://bionescu.aimultimedialab.ro/VSD.html |
| **Violent Flows** | Crowd violence | https://www.openu.ac.il/home/hassner/data/violentflows/ |
| **NTU RGB+D** | 114,480 (with depth) | https://rose1.ntu.edu.sg/dataset/actionRecognition/ |

---

## ⚡ **ONE-COMMAND DOWNLOAD ALL**

```bash
# Make script executable
chmod +x DOWNLOAD_ALL_DATASETS.sh

# Download everything (requires Kaggle API configured)
./DOWNLOAD_ALL_DATASETS.sh ./datasets
```

**Prerequisites:**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (get from https://www.kaggle.com/account)
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📊 **DATASET COMPARISON**

| Dataset | Size | Resolution | FPS | Duration | Best For |
|---------|------|------------|-----|----------|----------|
| XD-Violence | 4,754 | Variable | Variable | 217h | Multimodal (audio+video) |
| UCF Crime | 1,900 | Surveillance | Variable | 128h | Anomaly detection (13 types) |
| RLVS | 2,000 | Variable | 30 | ~5s/clip | Real street fights |
| RWF-2000* | 2,000 | Surveillance | 30 | 5s/clip | Gold standard (need agreement) |
| UCF101 | 13,320 | Variable | Variable | 27h | General actions + violence |
| AIRTLab | 350 | 1920x1080 | 30 | ~5.6s/clip | High-resolution training |
| ShanghaiTech | 437 | Variable | Variable | Various | Campus surveillance |
| NTU CCTV* | 1,000 | CCTV | Variable | 8h+ | Real CCTV fights |

*Requires registration

---

## 🎯 **QUICK START TRAINING COMBINATIONS**

### Combination 1: Balanced Training Set
- RLVS (2,000) + UCF Crime (1,900) + AIRTLab (350) = **4,250 videos**
- Good mix of real fights, anomalies, and high-res data

### Combination 2: Maximum Data
- XD-Violence (4,754) + UCF Crime (1,900) + RLVS (2,000) = **8,654 videos**
- Includes multimodal data (audio+video)

### Combination 3: CCTV-Focused
- ShanghaiTech (437) + SCVD + VioPeru (280) + Fight Surv (300) = **~1,000+ videos**
- All surveillance/CCTV footage

### Combination 4: Ultimate Dataset
- All Kaggle + All GitHub = **~25,000+ videos**
- Maximum diversity and coverage

---

## 📥 **DOWNLOAD SIZE ESTIMATES**

| Dataset | Approx Size | Download Time (100 Mbps) |
|---------|-------------|--------------------------|
| RLVS | ~5 GB | ~7 minutes |
| UCF Crime | ~30 GB | ~40 minutes |
| XD-Violence | ~100 GB | ~2.2 hours |
| ShanghaiTech | ~10 GB | ~13 minutes |
| AIRTLab | ~15 GB | ~20 minutes |
| UCF101 | ~7 GB | ~9 minutes |
| **All Datasets** | **~500 GB** | **~11 hours** |

---

## 🔍 **DATASET FINDER TOOLS**

### Search Tools:
- **Kaggle**: https://www.kaggle.com/datasets (search: "violence detection")
- **Papers with Code**: https://paperswithcode.com/datasets (search: "violence")
- **Academic Torrents**: https://academictorrents.com/ (search: "violence" or "fight")
- **IEEE DataPort**: https://ieee-dataport.org/ (search: "violence detection")
- **GitHub**: https://github.com/topics/violence-detection

### Meta Resources:
- Awesome Violence Detection: https://github.com/idejie/Awesome-Violence-Detection
- Official UCF CRCV: https://www.crcv.ucf.edu/data/
- MediaEval Benchmarks: http://www.multimediaeval.org/datasets/

---

## ✅ **VERIFICATION CHECKLIST**

After downloading, verify:
- [ ] Video files are not corrupted (use `ffmpeg -v error`)
- [ ] Folder structure matches documentation
- [ ] Train/test/val splits exist (if applicable)
- [ ] Annotation files are present
- [ ] README and license files reviewed
- [ ] Citation information saved

---

## 📚 **CITATION TEMPLATES**

### RLVS:
```
@dataset{rlvs2019,
  title={Real Life Violence Situations Dataset},
  author={Mohamed Mustafa},
  year={2019},
  publisher={Kaggle}
}
```

### UCF Crime:
```
@inproceedings{sultani2018real,
  title={Real-world anomaly detection in surveillance videos},
  author={Sultani, Waqas and Chen, Chen and Shah, Mubarak},
  booktitle={CVPR},
  year={2018}
}
```

### XD-Violence:
```
@inproceedings{wu2020not,
  title={Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision},
  author={Wu, Peng and Liu, Jing and Shi, Yujia and Sun, Yujie and Shao, Fangtao and Wu, Zhaoyang and Yang, Zhiwei},
  booktitle={ECCV},
  year={2020}
}
```

---

## 🚨 **IMPORTANT NOTES**

1. **All datasets are for ACADEMIC/RESEARCH use only**
2. **Check licensing terms before using**
3. **Cite original papers when publishing**
4. **Some datasets require signed agreements (RWF-2000, NTU)**
5. **IEEE DataPort requires subscription (but many alternatives exist)**
6. **Respect privacy - use for defensive security only**

---

## 💡 **TIPS FOR SUCCESS**

1. **Start small**: Download RLVS first (2,000 videos, well-balanced)
2. **Test your pipeline**: Use AIRTLab (high-res, manageable size)
3. **Scale up**: Add XD-Violence for multimodal data
4. **Diversify**: Mix surveillance (ShanghaiTech) + real fights (RLVS) + movies
5. **Validate**: Use multiple test sets to avoid overfitting

---

## 📞 **NEED MORE HELP?**

- Check full catalog: `COMPREHENSIVE_VIOLENCE_DATASETS_CATALOG.md`
- Run download script: `./DOWNLOAD_ALL_DATASETS.sh`
- GitHub Issues: Report problems in dataset repos
- Papers with Code: Find latest benchmarks and results

---

**Last Updated**: October 2025
**Total Free Datasets**: 38+
**Total Free Videos**: 25,000+
**With Registration**: 30,000+
