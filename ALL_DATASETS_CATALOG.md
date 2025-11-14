# Complete Violence Detection Datasets Catalog

## Academic Datasets (Direct Download)

### 1. RWF-2000 (Real World Fight)
- **Size**: 2,000 videos (1,000 fight, 1,000 non-fight)
- **Source**: GitHub
- **Download**:
```bash
cd /workspace/violence_datasets
git clone https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
cd RWF2000-Video-Database-for-Violence-Detection
# Videos hosted on Google Drive - use gdown
pip install gdown
gdown --id 1-1gvPCjmMf4TH4PjLM9xTf3tqFKVBL8W
unzip RWF-2000.zip
```

### 2. Hockey Fight Detection Dataset
- **Size**: 1,000 videos (500 fight, 500 non-fight)
- **Source**: University of Castilla-La Mancha
- **Download**:
```bash
cd /workspace/violence_datasets
wget http://visilab.etsii.uclm.es/personas/oscar/FightDetection/videos.zip
unzip videos.zip
```

### 3. Violent Flows Dataset
- **Size**: ~250 videos
- **Source**: Open University of Israel
- **Download**:
```bash
cd /workspace/violence_datasets
wget https://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.tar.gz
tar -xzf violent_flows.tar.gz
```

### 4. UCF Crime Dataset
- **Size**: 1,900+ videos (950 anomaly, 950 normal)
- **Source**: University of Central Florida
- **Download**:
```bash
cd /workspace/violence_datasets
# Available on multiple sources:
kaggle datasets download -d pelealg/ucf-crime-dataset --unzip
# OR from Google Drive (13 GB)
gdown --id 1BikAzFT8xTb-i9dEaCeyGHB7jpXhvSb1
```

### 5. RLVS (Real Life Violence Situations)
- **Size**: 2,000 videos (1,000 violence, 1,000 non-violence)
- **Source**: Kaggle
- **Download**:
```bash
kaggle datasets download -d mohamedabdallah/real-life-violence-situations-dataset --unzip
```

### 6. Surveillance Fighting Dataset
- **Size**: Varies
- **Download**:
```bash
kaggle datasets download -d mateohervas/surveillance-fighting-dataset --unzip
```

### 7. DCSASS Dataset
- **Size**: Suspicious action detection
- **Download**:
```bash
kaggle datasets download -d mateohervas/dcsass-dataset --unzip
```

## Kaggle Datasets (Requires /workspace/kaggle.json)

### Violence Detection Specific
```bash
kaggle datasets download -d yassershrief/hockey-fight-detection-dataset --unzip
kaggle datasets download -d sujaykapadnis/fight-detection --unzip
kaggle datasets download -d nishantrahate/fight-dataset --unzip
kaggle datasets download -d sayakpaul/rwf-2000 --unzip
kaggle datasets download -d toluwaniaremu/violence-detection-videos --unzip
kaggle datasets download -d seifmahmoud9/fighting-videos --unzip
kaggle datasets download -d puneetmalhotra/violence-detection-dataset --unzip
kaggle datasets download -d nikhilbhange/video-violence-detection --unzip
kaggle datasets download -d mission-ai/fight-detection-surv-dataset --unzip
kaggle datasets download -d ashrafemad/violent-scenes-dataset --unzip
kaggle datasets download -d gregorywinter/violence-detection-in-videos --unzip
```

### UCF-101 (Contains Violence Classes)
```bash
kaggle datasets download -d pevogam/ucf101 --unzip
kaggle datasets download -d matthewjansen/ucf101-action-recognition --unzip
```

## Additional Academic Sources

### 8. BEHAVE Dataset
- **Size**: Multi-person interaction dataset
- **Source**: University of Edinburgh
- **Download**: http://groups.inf.ed.ac.uk/vision/BEHAVEDATA/INTERACTIONS/

### 9. UT-Interaction Dataset
- **Size**: 20 video sequences of 6 human activities
- **Source**: University of Texas
- **Download**: http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html

### 10. CAVIAR Dataset
- **Size**: Surveillance footage with labeled actions
- **Source**: EC Funded project
- **Download**: https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/

### 11. MediaEval Violence Detection
- **Size**: Movies and TV shows with violence annotations
- **Source**: MediaEval benchmark
- **Download**: http://www.multimediaeval.org/

## YouTube-8M Segments (Violence-Related)
```bash
# Contains violence-related video segments
wget http://us.data.yt8m.org/2/j/train/trainx.tfrecord
# Extract violence-related segments using labels
```

## GitHub Repositories with Datasets

### 12. Fight Detection Dataset Collection
```bash
git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos
```

### 13. Violence Detection Datasets Aggregator
```bash
git clone https://github.com/Alro10/violence-detection
```

## Download ALL Script (Use on Vast.ai)

```bash
#!/bin/bash
# Run this on Vast.ai to download ALL datasets

cd /workspace/violence_datasets
mkdir -p /workspace/violence_datasets

# Install dependencies
pip install gdown kaggle

# Kaggle datasets (requires kaggle.json in /workspace/)
python3 /workspace/violence_detection_mvp/DOWNLOAD_ALL_DATASETS.py

# Additional direct downloads
wget http://visilab.etsii.uclm.es/personas/oscar/FightDetection/videos.zip
unzip videos.zip -d hockey_fight

# RWF-2000 from Google Drive
gdown --id 1-1gvPCjmMf4TH4PjLM9xTf3tqFKVBL8W
unzip RWF-2000.zip

# Violent Flows
wget https://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.tar.gz
tar -xzf violent_flows.tar.gz -d violent_flows

echo "✅ Download complete!"
echo "Count videos:"
find /workspace/violence_datasets -name "*.mp4" -o -name "*.avi" | wc -l
```

## Estimated Total Videos Available

| Dataset | Violent | Non-Violent | Total |
|---------|---------|-------------|-------|
| RWF-2000 | 1,000 | 1,000 | 2,000 |
| RLVS | 1,000 | 1,000 | 2,000 |
| Hockey Fight | 500 | 500 | 1,000 |
| UCF Crime | 950 | 950 | 1,900 |
| Violent Flows | ~125 | ~125 | ~250 |
| Surveillance Fighting | ~500 | ~500 | ~1,000 |
| Kaggle Collections | ~2,000 | ~2,000 | ~4,000 |
| **TOTAL** | **~6,075** | **~6,075** | **~12,150** |

**Combined with your downloads:**
- Your current: 8,898 videos (from FIND_ALL_VIDEOS.py)
- Academic datasets: ~12,150 videos
- **Grand Total: ~21,000+ videos**

## Quick Start on Vast.ai

```bash
# 1. Upload kaggle.json to /workspace/
# 2. Run comprehensive download
python3 /workspace/violence_detection_mvp/DOWNLOAD_ALL_DATASETS.py

# 3. Search for more
python3 /workspace/violence_detection_mvp/SEARCH_ALL_KAGGLE_VIOLENCE.py

# 4. Download additional findings
cat /workspace/all_violence_datasets.txt | while read dataset; do
    kaggle datasets download -d $dataset -p /workspace/violence_datasets --unzip
done
```
