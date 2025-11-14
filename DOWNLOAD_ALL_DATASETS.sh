#!/bin/bash
#
# Comprehensive Violence Detection Dataset Downloader
# Downloads all freely accessible violence detection datasets
# For defensive security research - CCTV violence detection
#
# Requirements:
# - kaggle CLI (pip install kaggle)
# - kaggle API key configured (~/.kaggle/kaggle.json)
# - git
# - wget
# - sufficient disk space (~500GB recommended)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory for datasets
BASE_DIR="${1:-./datasets/violence_detection}"
mkdir -p "$BASE_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Violence Detection Dataset Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Base directory: ${GREEN}$BASE_DIR${NC}"
echo ""

# Function to print section headers
print_header() {
    echo -e "\n${YELLOW}>>> $1${NC}\n"
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        echo -e "Install with: pip install $1"
        exit 1
    fi
}

# Check prerequisites
print_header "Checking Prerequisites"
check_command kaggle
check_command git
check_command wget

# Check Kaggle authentication
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}Error: Kaggle API key not found${NC}"
    echo -e "Please configure Kaggle API:"
    echo -e "1. Go to https://www.kaggle.com/account"
    echo -e "2. Create new API token"
    echo -e "3. Place kaggle.json in ~/.kaggle/"
    echo -e "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo -e "${GREEN}All prerequisites satisfied${NC}"

#############################################
# TIER 1: KAGGLE DATASETS (Immediate Access)
#############################################

print_header "TIER 1: Downloading Kaggle Datasets"

# 1. RLVS Dataset (2000 videos)
if [ ! -d "$BASE_DIR/rlvs" ]; then
    echo -e "${BLUE}Downloading RLVS (Real Life Violence Situations)...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset
    unzip -q real-life-violence-situations-dataset.zip -d rlvs
    rm real-life-violence-situations-dataset.zip
    echo -e "${GREEN}✓ RLVS downloaded (2000 videos)${NC}"
else
    echo -e "${YELLOW}RLVS already exists, skipping${NC}"
fi

# 2. UCF Crime Dataset (1900 videos)
if [ ! -d "$BASE_DIR/ucf_crime" ]; then
    echo -e "${BLUE}Downloading UCF Crime Dataset...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d odins0n/ucf-crime-dataset
    unzip -q ucf-crime-dataset.zip -d ucf_crime
    rm ucf-crime-dataset.zip
    echo -e "${GREEN}✓ UCF Crime downloaded (1900 videos, 128 hours)${NC}"
else
    echo -e "${YELLOW}UCF Crime already exists, skipping${NC}"
fi

# 3. XD-Violence Dataset (4754 videos)
if [ ! -d "$BASE_DIR/xd_violence" ]; then
    echo -e "${BLUE}Downloading XD-Violence Dataset...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d nguhaduong/xd-violence-video-dataset
    unzip -q xd-violence-video-dataset.zip -d xd_violence
    rm xd-violence-video-dataset.zip
    echo -e "${GREEN}✓ XD-Violence downloaded (4754 videos, 217 hours)${NC}"
else
    echo -e "${YELLOW}XD-Violence already exists, skipping${NC}"
fi

# 4. ShanghaiTech Campus Dataset (437 videos)
if [ ! -d "$BASE_DIR/shanghaitech" ]; then
    echo -e "${BLUE}Downloading ShanghaiTech Campus Dataset...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d ravikagglex/shanghaitech-anomaly-detection
    unzip -q shanghaitech-anomaly-detection.zip -d shanghaitech
    rm shanghaitech-anomaly-detection.zip
    echo -e "${GREEN}✓ ShanghaiTech downloaded (437 videos)${NC}"
else
    echo -e "${YELLOW}ShanghaiTech already exists, skipping${NC}"
fi

# 5. SCVD Dataset (Smart-City CCTV)
if [ ! -d "$BASE_DIR/scvd" ]; then
    echo -e "${BLUE}Downloading SCVD (Smart-City CCTV Violence Detection)...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
    unzip -q smartcity-cctv-violence-detection-dataset-scvd.zip -d scvd
    rm smartcity-cctv-violence-detection-dataset-scvd.zip
    echo -e "${GREEN}✓ SCVD downloaded${NC}"
else
    echo -e "${YELLOW}SCVD already exists, skipping${NC}"
fi

# 6. EAVDD Dataset
if [ ! -d "$BASE_DIR/eavdd" ]; then
    echo -e "${BLUE}Downloading EAVDD (Extended Automatic Violence Detection)...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d arnab91/eavdd-violence
    unzip -q eavdd-violence.zip -d eavdd
    rm eavdd-violence.zip
    echo -e "${GREEN}✓ EAVDD downloaded${NC}"
else
    echo -e "${YELLOW}EAVDD already exists, skipping${NC}"
fi

# 7. Movies Violence Dataset
if [ ! -d "$BASE_DIR/movies_violence" ]; then
    echo -e "${BLUE}Downloading Movies Violence Dataset...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d pratt3000/moviesviolencenonviolence
    unzip -q moviesviolencenonviolence.zip -d movies_violence
    rm moviesviolencenonviolence.zip
    echo -e "${GREEN}✓ Movies Violence downloaded${NC}"
else
    echo -e "${YELLOW}Movies Violence already exists, skipping${NC}"
fi

# 8. Audio-based Violence Detection
if [ ! -d "$BASE_DIR/audio_violence" ]; then
    echo -e "${BLUE}Downloading Audio-based Violence Detection Dataset...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d fangfangz/audio-based-violence-detection-dataset
    unzip -q audio-based-violence-detection-dataset.zip -d audio_violence
    rm audio-based-violence-detection-dataset.zip
    echo -e "${GREEN}✓ Audio Violence downloaded${NC}"
else
    echo -e "${YELLOW}Audio Violence already exists, skipping${NC}"
fi

# 9. UCF101 Action Recognition
if [ ! -d "$BASE_DIR/ucf101" ]; then
    echo -e "${BLUE}Downloading UCF101 Action Recognition...${NC}"
    cd "$BASE_DIR"
    kaggle datasets download -d matthewjansen/ucf101-action-recognition
    unzip -q ucf101-action-recognition.zip -d ucf101
    rm ucf101-action-recognition.zip
    echo -e "${GREEN}✓ UCF101 downloaded (13,320 clips)${NC}"
else
    echo -e "${YELLOW}UCF101 already exists, skipping${NC}"
fi

#############################################
# TIER 2: GITHUB DATASETS
#############################################

print_header "TIER 2: Cloning GitHub Datasets"

# 1. AIRTLab Dataset (350 high-res videos)
if [ ! -d "$BASE_DIR/airtlab" ]; then
    echo -e "${BLUE}Cloning AIRTLab Dataset...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git airtlab
    echo -e "${GREEN}✓ AIRTLab cloned (350 videos, 1920x1080)${NC}"
else
    echo -e "${YELLOW}AIRTLab already exists, skipping${NC}"
fi

# 2. VioPeru Dataset (280 videos)
if [ ! -d "$BASE_DIR/vioperu" ]; then
    echo -e "${BLUE}Cloning VioPeru Dataset...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/hhuillcen/VioPeru.git vioperu
    echo -e "${GREEN}✓ VioPeru cloned (280 real CCTV videos)${NC}"
else
    echo -e "${YELLOW}VioPeru already exists, skipping${NC}"
fi

# 3. Fight Detection Surveillance Dataset (300 videos)
if [ ! -d "$BASE_DIR/fight_detection_surv" ]; then
    echo -e "${BLUE}Cloning Fight Detection Surveillance Dataset...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/seymanurakti/fight-detection-surv-dataset.git fight_detection_surv
    echo -e "${GREEN}✓ Fight Detection Surveillance cloned (300 videos)${NC}"
else
    echo -e "${YELLOW}Fight Detection Surveillance already exists, skipping${NC}"
fi

# 4. UCA Dataset (CVPR 2024)
if [ ! -d "$BASE_DIR/uca" ]; then
    echo -e "${BLUE}Cloning UCA Dataset (CVPR 2024)...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/Xuange923/Surveillance-Video-Understanding.git uca
    echo -e "${GREEN}✓ UCA cloned (1854 videos, 110.7 hours)${NC}"
else
    echo -e "${YELLOW}UCA already exists, skipping${NC}"
fi

# 5. XD-Violence GitHub (for features)
if [ ! -d "$BASE_DIR/xd_violence_github" ]; then
    echo -e "${BLUE}Cloning XD-Violence GitHub (features)...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/Roc-Ng/XDVioDet.git xd_violence_github
    echo -e "${GREEN}✓ XD-Violence GitHub cloned${NC}"
else
    echo -e "${YELLOW}XD-Violence GitHub already exists, skipping${NC}"
fi

# 6. RWF-2000 (repo only - videos require agreement)
if [ ! -d "$BASE_DIR/rwf2000_repo" ]; then
    echo -e "${BLUE}Cloning RWF-2000 Repository (videos require separate agreement)...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection.git rwf2000_repo
    echo -e "${YELLOW}⚠ RWF-2000 repo cloned. Videos require signing agreement with SMIIP Lab${NC}"
else
    echo -e "${YELLOW}RWF-2000 repo already exists, skipping${NC}"
fi

# 7. NTU RGB+D samples
if [ ! -d "$BASE_DIR/ntu_rgbd_samples" ]; then
    echo -e "${BLUE}Cloning NTU RGB+D Sample Code...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/shahroudy/NTURGB-D.git ntu_rgbd_samples
    echo -e "${YELLOW}⚠ NTU RGB+D samples cloned. Full dataset requires registration${NC}"
else
    echo -e "${YELLOW}NTU RGB+D samples already exists, skipping${NC}"
fi

# 8. Kinetics Dataset downloader
if [ ! -d "$BASE_DIR/kinetics_downloader" ]; then
    echo -e "${BLUE}Cloning Kinetics Dataset Downloader...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/cvdfoundation/kinetics-dataset.git kinetics_downloader
    echo -e "${GREEN}✓ Kinetics downloader cloned. Use k700_2020_downloader.sh${NC}"
else
    echo -e "${YELLOW}Kinetics downloader already exists, skipping${NC}"
fi

# 9. AVA Dataset repository
if [ ! -d "$BASE_DIR/ava_dataset_repo" ]; then
    echo -e "${BLUE}Cloning AVA Dataset Repository...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/cvdfoundation/ava-dataset.git ava_dataset_repo
    echo -e "${GREEN}✓ AVA dataset repo cloned${NC}"
else
    echo -e "${YELLOW}AVA dataset already exists, skipping${NC}"
fi

#############################################
# TIER 3: DIRECT DOWNLOADS
#############################################

print_header "TIER 3: Direct Downloads"

# Download AVA annotations
if [ ! -f "$BASE_DIR/ava_annotations/ava_v2.2.zip" ]; then
    echo -e "${BLUE}Downloading AVA Annotations...${NC}"
    mkdir -p "$BASE_DIR/ava_annotations"
    cd "$BASE_DIR/ava_annotations"
    wget -q --show-progress https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip
    unzip -q ava_v2.2.zip
    echo -e "${GREEN}✓ AVA annotations downloaded${NC}"
else
    echo -e "${YELLOW}AVA annotations already exist, skipping${NC}"
fi

#############################################
# SUMMARY
#############################################

print_header "Download Summary"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Datasets downloaded to: ${BLUE}$BASE_DIR${NC}"
echo ""
echo -e "${YELLOW}Immediately Accessible Datasets:${NC}"
echo -e "  ✓ RLVS: 2,000 videos"
echo -e "  ✓ UCF Crime: 1,900 videos (128 hours)"
echo -e "  ✓ XD-Violence: 4,754 videos (217 hours)"
echo -e "  ✓ ShanghaiTech: 437 videos"
echo -e "  ✓ SCVD: CCTV-focused videos"
echo -e "  ✓ EAVDD: Extended violence videos"
echo -e "  ✓ Movies Violence: Movie clips"
echo -e "  ✓ Audio Violence: Audio-only dataset"
echo -e "  ✓ UCF101: 13,320 action clips"
echo -e "  ✓ AIRTLab: 350 high-res videos (1920x1080)"
echo -e "  ✓ VioPeru: 280 real CCTV videos"
echo -e "  ✓ Fight Detection Surv: 300 videos"
echo -e "  ✓ UCA (CVPR 2024): 1,854 videos (110.7 hours)"
echo ""
echo -e "${YELLOW}Total Accessible Videos: ~25,000+${NC}"
echo ""
echo -e "${YELLOW}Additional Resources Cloned:${NC}"
echo -e "  ✓ XD-Violence features repo"
echo -e "  ✓ RWF-2000 repo (videos require agreement)"
echo -e "  ✓ NTU RGB+D samples (full dataset requires registration)"
echo -e "  ✓ Kinetics downloader scripts"
echo -e "  ✓ AVA dataset tools and annotations"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "1. Review README files in each dataset directory"
echo -e "2. For RWF-2000 videos: Contact SMIIP Lab and sign agreement"
echo -e "3. For NTU datasets: Register at https://rose1.ntu.edu.sg/dataset/"
echo -e "4. For Kinetics-700: Run k700_2020_downloader.sh in kinetics_downloader/"
echo -e "5. Combine datasets for training your violence detection model"
echo ""
echo -e "${GREEN}Dataset catalog saved at:${NC}"
echo -e "${BLUE}$(dirname $BASE_DIR)/COMPREHENSIVE_VIOLENCE_DATASETS_CATALOG.md${NC}"
echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Recommended for Immediate Training:${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "1. RLVS (balanced, diverse)"
echo -e "2. UCF Crime (large-scale, 13 anomaly types)"
echo -e "3. XD-Violence (multimodal, audio+video)"
echo -e "4. AIRTLab (high-resolution)"
echo -e "5. ShanghaiTech (surveillance-focused)"
echo ""
echo -e "${GREEN}All datasets are for ACADEMIC/RESEARCH use only${NC}"
echo -e "${GREEN}Cite original papers when using these datasets${NC}"
echo ""
