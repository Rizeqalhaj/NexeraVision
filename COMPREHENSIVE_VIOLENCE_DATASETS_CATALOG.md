# Comprehensive Violence Detection Datasets Catalog
## Academic & Research Violence Detection Datasets (2024-2025)

**Purpose**: Defensive security research - CCTV violence detection for public safety
**Last Updated**: October 2025
**Total Datasets**: 45+

---

## 📊 **TIER 1: LARGE-SCALE DATASETS (1000+ videos)**

### 1. **RWF-2000 (Real World Fight Database)**
- **Size**: 2,000 videos (1,000 violent + 1,000 non-violent)
- **Duration**: 5 seconds per video, 30 fps
- **Source**: Real-world surveillance footage from YouTube
- **Resolution**: Variable (surveillance quality)
- **Access**: Requires registration and signed agreement
- **Download Method**:
  - GitHub: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
  - **NOTE**: Video files NOT publicly available due to privacy. Must contact SMIIP Lab and sign agreement
- **Citation**: Ming Cheng et al., "RWF-2000: An Open Large Scale Video Database for Violence Detection"
- **License**: Academic research only (agreement required)

---

### 2. **UCF Crime Dataset**
- **Size**: 1,900 videos, 128 hours total
- **Duration**: Long untrimmed videos
- **Classes**: 13 anomaly types (Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, Vandalism)
- **Source**: Real-world surveillance cameras
- **Resolution**: Variable surveillance quality
- **Download Methods**:
  - Official: https://www.crcv.ucf.edu/projects/real-world/
  - Kaggle #1: https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf
  - Kaggle #2: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
  - GitHub (I3D features): https://github.com/Roc-Ng/DeepMIL
- **License**: Academic research
- **Notes**: Can be used for general anomaly detection or specific anomaly recognition

---

### 3. **XD-Violence Dataset**
- **Size**: 4,754 videos, 217 hours total
- **Split**: 2,405 violent + 2,349 non-violent | Train: 3,954 / Test: 800
- **Duration**: Untrimmed videos with audio
- **Modality**: Video + Audio (multimodal)
- **Source**: YouTube and movies
- **Download Methods**:
  - Official: https://roc-ng.github.io/XD-Violence/
  - GitHub: https://github.com/Roc-Ng/XDVioDet
  - Hugging Face: https://huggingface.co/datasets/jherng/xd-violence
  - Kaggle #1: https://www.kaggle.com/datasets/bhavay192/xd-violence-1005-2004-set
  - Kaggle #2: https://www.kaggle.com/datasets/nguhaduong/xd-violence-video-dataset
- **Features**: Pre-extracted I3D features available
- **License**: Academic research (ECCV 2020)
- **Citation**: "Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision"

---

### 4. **Real Life Violence Situations (RLVS)**
- **Size**: 2,000 videos (1,000 violent + 1,000 non-violent)
- **Duration**: ~5 seconds average
- **Source**: YouTube (real street fights, sports, daily activities)
- **Environments**: Multiple (surveillance, movies, recordings)
- **Download Methods**:
  - Kaggle (Primary): https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
  - Kaggle (Images): https://www.kaggle.com/datasets/karandeep98/real-life-violence-and-nonviolence-data
  - Kaggle (Skeleton): https://www.kaggle.com/datasets/musreaghaseb/skeleton-based-rlvs-dataset
  - IEEE DataPort: DOI: 10.21227/94hb-6871
- **License**: Free for research (Kaggle datasets)
- **Notes**: One of the most accessible datasets

---

### 5. **VID (Violence in Various Contexts) Dataset** ⭐ NEW 2024
- **Size**: 3,020 videos (1,510 violent + 1,510 non-violent)
- **Duration**: 3-12 seconds per clip
- **Source**: Non-professional actors in real contexts
- **Resolution**: Various
- **Published**: August 2024
- **Download**: ScienceDirect article link: https://www.sciencedirect.com/science/article/pii/S2352340924008394
- **License**: Academic research
- **Notes**: Comprehensive coverage of various violence contexts

---

### 6. **Hockey Fight Dataset**
- **Size**: 1,000 clips (500 fights + 500 non-fights)
- **Duration**: 50 frames per clip
- **Resolution**: 320x240 (downscaled from 720x576)
- **Source**: NHL hockey games
- **Download Methods**:
  - Academic Torrents: https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89
  - Kaggle: Search "Hockey Fight Videos"
- **License**: Academic research
- **Citation**: Nievas et al., CAIP 2011
- **Notes**: Classic benchmark dataset, high accuracy achievable (98%+)

---

### 7. **Movies Fight Dataset**
- **Size**: 1,000 sequences (fights + non-fights)
- **Source**: Action movies
- **Resolution**: Variable
- **Download Methods**:
  - Academic Torrents: https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635
  - Kaggle: https://www.kaggle.com/datasets/pratt3000/moviesviolencenonviolence
- **License**: Academic research
- **Citation**: Nievas et al., 2011
- **Performance**: ~90% detection accuracy benchmark

---

### 8. **NTU CCTV-Fights Dataset**
- **Size**: 1,000 videos, 8+ hours total
- **Source**: Real CCTV footage
- **Resolution**: Surveillance quality
- **Download**: https://rose1.ntu.edu.sg/dataset/cctvFights/
- **License**: Academic research only (registration required)
- **Access**: Free for researchers from educational/research institutes

---

## 📊 **TIER 2: MEDIUM-SCALE DATASETS (100-1000 videos)**

### 9. **DVD (Diverse Violence Dataset)** ⭐ NEW 2025
- **Size**: 500 videos, 2.7M frames
- **Annotations**: Frame-level annotations
- **Features**: Diverse environments, varying lighting, multiple camera sources, complex social interactions, rich metadata
- **Published**: June 2025
- **Download**: ArXiv paper: https://arxiv.org/abs/2506.05372
- **License**: Academic research
- **Notes**: State-of-the-art dataset with comprehensive annotations

---

### 10. **AIRTLab Violence Dataset**
- **Size**: 350 clips (230 violent + 120 non-violent)
- **Duration**: ~5.63 seconds average
- **Resolution**: 1920×1080, 30 fps
- **Format**: MP4
- **Download Methods**:
  - GitHub: https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos
  - **NOTE**: Use `git clone` (not ZIP download due to compression issues)
- **Companion Repo**: https://github.com/airtlab/violence-detection-tests-on-the-airtlab-dataset
- **License**: Free for research and educational purposes
- **Notes**: High-resolution, includes challenging non-violent clips

---

### 11. **ShanghaiTech Campus Dataset**
- **Size**: 437 videos (330 train + 107 test)
- **Frames**: 317,398 total (274,515 training + 42,883 testing)
- **Anomalies**: 130 abnormal events
- **Scenes**: 13 scenes with complex lighting and camera angles
- **Annotations**: Frame-level and pixel-level ground truth
- **Download Methods**:
  - Official: https://svip-lab.github.io/dataset/campus_dataset.html
  - Kaggle: https://www.kaggle.com/datasets/ravikagglex/shanghaitech-anomaly-detection
  - GitHub: shanghaitech.tar.gz from SVIP Lab
- **License**: BSD 2-Clause License
- **Notes**: Campus surveillance, anomaly detection focus

---

### 12. **VSD (Violent Scenes Dataset)**
- **Size**: 18 Hollywood movies (original) / 32 movies + 86 web videos (VSD2014)
- **Source**: Hollywood movies and YouTube
- **Annotations**: Violence annotations at scene level
- **Download Methods**:
  - InterDigital: https://www.interdigital.com/data_sets/violent-scenes-dataset
  - MediaEval Benchmarking: VSD dataset page
  - AIM Lab: https://bionescu.aimultimedialab.ro/VSD.html
- **License**: Free for research (originally Technicolor, now InterDigital)
- **Notes**: Used in MediaEval 2011-2014 benchmarking campaigns

---

### 13. **VioPeru Dataset** ⭐ NEW 2024
- **Size**: 280 videos
- **Source**: Real surveillance camera records from Peru municipalities
- **Setting**: Citizen security offices
- **Download**: GitHub: https://github.com/hhuillcen/VioPeru
- **Published**: January 2024
- **License**: Academic research
- **Notes**: Real-world municipal surveillance footage

---

### 14. **CHU Surveillance Violence Detection Dataset (CSVD)**
- **Size**: Video clips with 12 action classes
- **Classes**: Idle, run, walk, high-five, wave, pose, shove, grapple, punch, kick, clobbering
- **Classification**: Violent vs non-violent actions
- **Source**: CCTV footage
- **Download**: IEEE DataPort, DOI: 10.21227/cd9g-xs15
- **Published**: September 2020
- **License**: IEEE DataPort subscription required

---

### 15. **Violent Flows Dataset**
- **Size**: Real-world crowd violence videos
- **Source**: Movies.zip file
- **Classes**: Violent/Non-violent
- **Focus**: Crowd violence detection
- **Download**: https://www.openu.ac.il/home/hassner/data/violentflows/
- **Access**: Registration required (username/password from form)
- **License**: Academic research
- **Citation**: Hassner et al., "Violent Flows: Real-Time Detection of Violent Crowd Behavior", CVPR 2012

---

### 16. **Fight Detection Surveillance Dataset**
- **Size**: 300 videos (150 fight + 150 non-fight)
- **Source**: YouTube videos
- **Download**: GitHub: https://github.com/seymanurakti/fight-detection-surv-dataset
- **Published**: IPTA 2019
- **License**: Academic research
- **Notes**: Vision-based fight detection from surveillance cameras

---

### 17. **Movie Clip Dataset**
- **Size**: Various movie clips
- **Source**: Movies with violent and non-violent scenes
- **Download**: Figshare: https://figshare.com/articles/dataset/Movie_Clip_Dataset_A_Novel_Dataset_for_Generalised_Real-World_Violence_Detection/23643555
- **License**: Academic research
- **Notes**: For generalised real-world violence detection

---

## 📊 **TIER 3: SPECIALIZED & NEW DATASETS (2024-2025)**

### 18. **Violence Detection in Campus Surveillance** ⭐ NEW 2025
- **Size**: Self-recorded videos
- **Classes**: 4 distinct classes (Slap, Punch, Kick, Group Violence, Others)
- **Setting**: College campus environments
- **Source**: Simulated violent activities
- **Download**: IEEE DataPort, DOI: 10.21227/vx78-2y90
- **Published**: January 2025
- **License**: IEEE DataPort subscription required

---

### 19. **Weapon Violence Dataset 2.0 (WVD)** ⭐ NEW 2024
- **Size**: Synthetic dataset
- **Source**: Grand Theft Auto V (GTA-V) gameplay
- **Categories**: Hot violence, Cold violence, No violence
- **Modalities**: RGB videos + Optical flow
- **Download Methods**:
  - Kaggle: https://www.kaggle.com/ (search "Weapon Violence Dataset")
  - Paper: https://www.sciencedirect.com/science/article/pii/S2352340924004177
- **Published**: April 2024
- **License**: Free for research (Kaggle)
- **Notes**: First synthetic virtual dataset, 15% higher accuracy than real-world datasets

---

### 20. **SCVD (Smart-City CCTV Violence Detection Dataset)** ⭐ 2023-2024
- **Size**: CCTV recordings
- **Types**: Violence and Weaponized Violence
- **Setting**: Smart city surveillance
- **Download**: Kaggle: https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
- **Published**: December 2023
- **License**: Kaggle open dataset
- **GitHub**: https://github.com/tolusophy/Violence_Detection

---

### 21. **EAVDD (Extended Automatic Violence Detection Dataset)** ⭐ NEW 2024
- **Size**: Extended dataset
- **Download**: Kaggle: https://www.kaggle.com/datasets/arnab91/eavdd-violence
- **Published**: July 21, 2024
- **License**: Kaggle open dataset

---

### 22. **Violence Detection - Combined** ⭐ NEW 2024
- **Purpose**: Video violence classification
- **Download**: Kaggle: https://www.kaggle.com/datasets/yash07yadav/project-data
- **Published**: September 25, 2024
- **License**: Kaggle open dataset

---

### 23. **Audio-based Violence Detection Dataset**
- **Modality**: Audio only
- **Purpose**: Binary violence classification/detection
- **Download**: Kaggle: https://www.kaggle.com/datasets/fangfangz/audio-based-violence-detection-dataset
- **Published**: August 2023
- **License**: Kaggle open dataset
- **Notes**: Unique audio-only approach

---

### 24. **Violence Detection: Serious-Gaming Approach**
- **Source**: Virtual gaming data
- **Methodology**: Deep learning with serious games
- **Performance**: 15% higher accuracy than 3 well-known real-world datasets
- **Download**: IEEE DataPort, DOI: 10.21227/hkam-8367
- **Published**: September 2023
- **License**: IEEE DataPort subscription required

---

### 25. **MSV-PG (Multi-Scale Violence and Public Gathering)** ⭐ NEW 2024
- **Purpose**: Crowd behavior classification
- **Classes**: Violence and public gathering behaviors
- **Scale**: Multi-scale coverage
- **Published**: 2024
- **Source**: Frontiers in Computer Science
- **License**: Academic research

---

## 📊 **TIER 4: GENERAL ACTION RECOGNITION (Violence subset)**

### 26. **UCF101 Action Recognition**
- **Size**: 13,320 clips, 27 hours, 101 action classes
- **Source**: YouTube videos
- **Violence Classes**: Subset contains fighting/aggressive behaviors
- **Download Methods**:
  - Official: https://www.crcv.ucf.edu/data/UCF101.php
  - Kaggle: https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition
  - FiftyOne: Dataset Zoo
- **License**: Academic research
- **Notes**: General action recognition, can be filtered for violence

---

### 27. **Kinetics-700**
- **Size**: 650,000 clips, 700 human action classes
- **Duration**: ~10 seconds per clip
- **Source**: YouTube
- **Download Methods**:
  - GitHub: https://github.com/cvdfoundation/kinetics-dataset
  - Scripts: k700_2020_downloader.sh
  - MIM: `mim download mmaction2 --dataset kinetics700`
- **License**: Academic research
- **Notes**: General actions, includes human-human interactions

---

### 28. **AVA (Atomic Visual Actions)**
- **Size**: 430 videos (15 minutes each), 80 action classes
- **Annotations**: 1.58M action labels, spatio-temporal localization
- **Source**: Movies
- **Download Methods**:
  - Official: https://research.google.com/ava/
  - GitHub: https://github.com/cvdfoundation/ava-dataset
  - S3: https://s3.amazonaws.com/ava-dataset/trainval/[file_name]
  - Annotations: ava_v2.2.zip
- **License**: Academic research
- **Notes**: Atomic actions with bounding boxes, may include violence labels

---

### 29. **NTU RGB+D / NTU RGB+D 120**
- **Size**: 56,880 videos (60 classes) / 114,480 videos (120 classes)
- **Modality**: RGB+D (depth), skeletal data
- **Actions**: 40 daily + 9 health + 11 mutual (including punching, kicking)
- **Download**: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- **Access**: Registration + Release Agreement required
- **GitHub**: https://github.com/shahroudy/NTURGB-D
- **License**: Academic research only (educational/research institutes)

---

### 30. **VIRAT Video Dataset**
- **Size**: Multiple hours of surveillance video
- **Setting**: Ground cameras + aerial videos
- **Activities**: 46 activity types, 7 object types
- **Frame Rates**: 2-30 Hz
- **Download**: https://viratdata.org/
- **Published**: CVPR 2011
- **License**: Academic research (IARPA DIVA program)
- **Notes**: Multi-camera surveillance, includes fighting activities

---

## 📊 **TIER 5: CROWD & INTERACTION DATASETS**

### 31. **BEHAVE Dataset (Violence Detection Version)**
- **Size**: 4 video sequences, 2 hours total
- **Annotations**: Frame-level (first sequence fully annotated, 8 fragments)
- **Focus**: Fighting anomalies
- **Download**: Various sources (check recent papers)
- **License**: Academic research
- **Notes**: Two different BEHAVE datasets exist (this is the violence one)

---

### 32. **CAVIAR Dataset**
- **Setting**: Lab building entrance hall + shopping centre hallway
- **Anomalies**: Fighting between pedestrians
- **Source**: Surveillance scenarios
- **Download**: CAVIAR project repositories
- **License**: Academic research
- **Notes**: Classic surveillance dataset

---

### 33. **UT-Interaction Dataset**
- **Focus**: Human-human interactions
- **Includes**: Various interaction types
- **Download**: University of Texas repositories
- **License**: Academic research

---

### 34. **Crowd-11 Dataset**
- **Purpose**: Fine-grained crowd behavior analysis
- **Behaviors**: Includes panic movement
- **Published**: CVPR 2017
- **License**: Academic research

---

### 35. **UCA (UCF-Crime Annotation) Dataset** ⭐ NEW 2024
- **Size**: 1,854 videos, 23,542 sentences, 110.7 hours
- **Purpose**: Surveillance video-and-language understanding
- **Download**: GitHub: https://github.com/Xuange923/Surveillance-Video-Understanding
- **Published**: CVPR 2024
- **License**: Academic research

---

## 📊 **TIER 6: SYNTHETIC & SPECIALIZED**

### 36. **Synthetic Dataset for Panic Detection**
- **Purpose**: Crowd panic behavior
- **Source**: Simulated (SUMO/Unity)
- **Size**: 160 videos with panic annotations
- **Download**: Eurographics Digital Library
- **License**: Academic research
- **Notes**: Addresses expensive/hazardous nature of real panic scenarios

---

### 37. **UCF-Crime-DVS** ⭐ NEW 2025
- **Modality**: Event-based (spiking neural networks)
- **Size**: Based on UCF-Crime
- **Purpose**: Video anomaly detection with SNNs
- **Published**: March 2025
- **ArXiv**: https://arxiv.org/abs/2503.12905
- **License**: Academic research

---

### 38. **KCDD (Korean Crime Dialogue Dataset)** ⭐ NEW 2024
- **Size**: 22,249 dialogues
- **Modality**: Text/dialogue
- **Purpose**: Online violence classification (Korean language)
- **Source**: Crowd workers
- **Published**: EACL 2024
- **License**: Academic research
- **Notes**: First Korean dialogue dataset for violence

---

## 📊 **DOWNLOAD COMMAND EXAMPLES**

### Kaggle Datasets (requires Kaggle API):
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API key (~/.kaggle/kaggle.json)

# Download RLVS
kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset

# Download UCF Crime
kaggle datasets download -d odins0n/ucf-crime-dataset

# Download XD-Violence
kaggle datasets download -d nguhaduong/xd-violence-video-dataset

# Download ShanghaiTech
kaggle datasets download -d ravikagglex/shanghaitech-anomaly-detection

# Download SCVD
kaggle datasets download -d toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
```

### GitHub Datasets:
```bash
# AIRTLab Dataset
git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git

# RWF-2000 (repo only, videos require agreement)
git clone https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection.git

# XD-Violence
git clone https://github.com/Roc-Ng/XDVioDet.git

# VioPeru
git clone https://github.com/hhuillcen/VioPeru.git

# Fight Detection Surveillance
git clone https://github.com/seymanurakti/fight-detection-surv-dataset.git

# UCA Dataset (CVPR 2024)
git clone https://github.com/Xuange923/Surveillance-Video-Understanding.git

# NTU RGB+D samples
git clone https://github.com/shahroudy/NTURGB-D.git

# Kinetics Dataset
git clone https://github.com/cvdfoundation/kinetics-dataset.git

# AVA Dataset
git clone https://github.com/cvdfoundation/ava-dataset.git
```

### Direct Download:
```bash
# UCF Crime (official)
wget https://www.crcv.ucf.edu/projects/real-world/[dataset_files]

# AVA Annotations
wget https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip

# ShanghaiTech
wget [SVIP Lab link]/shanghaitech.tar.gz
```

---

## 📋 **DATASET REQUIREMENTS SUMMARY**

### ✅ **FREELY ACCESSIBLE (No Registration)**
- RLVS (Kaggle)
- UCF Crime (Kaggle)
- XD-Violence (Kaggle/HuggingFace)
- AIRTLab (GitHub)
- Hockey Fight (Academic Torrents)
- Movies Fight (Academic Torrents/Kaggle)
- ShanghaiTech (Kaggle/GitHub)
- Fight Detection Surveillance (GitHub)
- SCVD (Kaggle)
- EAVDD (Kaggle)
- Audio Violence (Kaggle)
- UCF101 (Official/Kaggle)
- Movie Clip Dataset (Figshare)
- VioPeru (GitHub)
- UCA (GitHub)
- Weapon Violence Dataset 2.0 (Kaggle)

### 🔐 **REGISTRATION REQUIRED (Free for Academic)**
- RWF-2000 (Agreement + contact)
- NTU CCTV-Fights (Registration)
- VSD/VSD2014 (Registration)
- Violent Flows (Registration form)
- NTU RGB+D (Agreement)
- VIRAT (DIVA program access)

### 💳 **SUBSCRIPTION REQUIRED**
- IEEE DataPort datasets (IEEE membership):
  - Violence Detection in Campus Surveillance
  - CHU Surveillance Violence Detection
  - Violence Detection: Serious-Gaming
  - Real Life Violence Situations (IEEE version)

### 📧 **CONTACT AUTHORS**
- RWF-2000 (sign agreement sheet)
- Some university-hosted datasets

---

## 🎯 **RECOMMENDED STARTING DATASETS**

### For Immediate Download & Training:
1. **RLVS** - Kaggle, 2000 videos, balanced
2. **UCF Crime** - Kaggle, large-scale, 13 anomaly types
3. **XD-Violence** - Kaggle/HuggingFace, 4754 videos, audio+video
4. **AIRTLab** - GitHub, 350 high-res videos
5. **ShanghaiTech** - Kaggle, campus surveillance
6. **Hockey Fight** - Academic Torrents, classic benchmark
7. **SCVD** - Kaggle, CCTV-focused
8. **Weapon Violence 2.0** - Kaggle, synthetic (high accuracy)

### For Academic Research (worth registration):
1. **RWF-2000** - Gold standard, requires agreement
2. **NTU CCTV-Fights** - Large-scale real CCTV
3. **VSD2014** - MediaEval benchmark
4. **NTU RGB+D** - If you need skeletal data

### For Latest Research (2024-2025):
1. **DVD** - June 2025, 2.7M frames
2. **VID** - August 2024, 3020 videos
3. **VioPeru** - January 2024, real municipal surveillance
4. **Violence in Campus** - January 2025, campus-specific
5. **UCA** - CVPR 2024, video-language understanding

---

## 📊 **DATASET STATISTICS OVERVIEW**

| Dataset | Size | Type | Year | Access |
|---------|------|------|------|--------|
| XD-Violence | 4,754 | Real+Audio | 2020 | Free |
| UCF Crime | 1,900 | Surveillance | 2018 | Free |
| RWF-2000 | 2,000 | Surveillance | 2019 | Agreement |
| RLVS | 2,000 | Mixed | 2019 | Free |
| VID | 3,020 | Acted | 2024 | Free |
| NTU CCTV | 1,000 | CCTV | 2019 | Registration |
| Hockey | 1,000 | Sports | 2011 | Free |
| Movies | 1,000 | Film | 2011 | Free |
| DVD | 500 | Real | 2025 | Free |
| AIRTLab | 350 | Mixed | 2020 | Free |
| VioPeru | 280 | CCTV | 2024 | Free |
| ShanghaiTech | 437 | Campus | 2018 | Free |

---

## 🔍 **ADDITIONAL RESOURCES**

### Meta-Resources:
- **Papers with Code**: https://paperswithcode.com/ (search "violence detection")
- **Academic Torrents**: https://academictorrents.com/ (search "violence" or "fight")
- **IEEE DataPort**: https://ieee-dataport.org/ (search "violence detection")
- **Kaggle Datasets**: https://www.kaggle.com/datasets (search "violence detection")
- **GitHub Topics**: https://github.com/topics/violence-detection

### Key Papers Listing Datasets:
- "A Controlled Benchmark of Video Violence Detection Techniques" (MDPI 2020)
- "Literature Review of Deep-Learning-Based Detection of Violence in Video" (Sensors 2024)
- "State-of-the-art violence detection techniques in video surveillance" (PMC 2022)

### Benchmark Repositories:
- Awesome Violence Detection: https://github.com/idejie/Awesome-Violence-Detection

---

## ⚖️ **LEGAL & ETHICAL NOTES**

1. **All datasets listed are for ACADEMIC/RESEARCH purposes only**
2. **Respect licensing terms** - do not use for commercial purposes without permission
3. **Privacy considerations** - some datasets require agreements due to privacy concerns
4. **Cite original papers** when using datasets
5. **Defensive security purpose** - these datasets are for developing safety systems (CCTV violence detection for public protection)

---

## 📝 **CITATION INFORMATION**

When using these datasets, please cite:
1. The original dataset paper
2. Any derivative dataset versions
3. The benchmark/competition if applicable (e.g., MediaEval)

Example citation format available in each dataset's official repository or paper.

---

## 🔄 **DATASET UPDATE TRACKING**

**Newest Additions (2024-2025)**:
- Violence Detection in Campus Surveillance (Jan 2025)
- DVD Dataset (Jun 2025)
- UCF-Crime-DVS (Mar 2025)
- VID Dataset (Aug 2024)
- EAVDD (Jul 2024)
- Violence Detection Combined (Sep 2024)
- Weapon Violence 2.0 (Apr 2024)
- VioPeru (Jan 2024)
- UCA Dataset (CVPR 2024)
- KCDD (EACL 2024)
- MSV-PG (2024)

**Classic Datasets (Still Relevant)**:
- Hockey Fight (2011) - 98%+ accuracy achievable
- Movies Fight (2011) - ~90% accuracy benchmark
- RWF-2000 (2019) - Gold standard
- UCF Crime (2018) - Anomaly detection standard

---

## 💡 **RECOMMENDATIONS FOR YOUR PROJECT**

### Immediate Action Plan:
1. **Download from Kaggle** (fastest):
   - RLVS (2000 videos)
   - UCF Crime (1900 videos)
   - XD-Violence (4754 videos)
   - ShanghaiTech (437 videos)
   - SCVD (CCTV-focused)

2. **Clone from GitHub**:
   - AIRTLab (350 high-res videos)
   - VioPeru (280 real CCTV)
   - UCA (CVPR 2024)

3. **Download from Academic Torrents**:
   - Hockey Fight (1000 clips)
   - Movies Fight (1000 clips)

4. **Register for Premium Datasets**:
   - RWF-2000 (contact SMIIP Lab)
   - NTU CCTV-Fights (worth the registration)

### Total Accessible Videos (No Subscription):
**~20,000+ videos** available immediately through Kaggle + GitHub + Academic Torrents

### With Registration (Free):
**~25,000+ videos** including RWF-2000, NTU datasets, VSD

---

**End of Comprehensive Catalog**
**Total Datasets Documented**: 38+ primary datasets
**Additional Resources**: 10+ meta-resources and tools
**Immediate Access**: ~20,000 videos
**With Free Registration**: ~25,000 videos
