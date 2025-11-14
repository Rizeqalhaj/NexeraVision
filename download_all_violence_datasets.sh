#!/bin/bash
#
# Comprehensive Violence Detection Dataset Downloader
# Downloads all freely accessible violence detection datasets
# For defensive security research - CCTV violence detection
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory for datasets
BASE_DIR="${1:-/workspace/violence_datasets}"
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
check_command git
check_command wget

# Setup Kaggle credentials
echo -e "${BLUE}Setting up Kaggle credentials...${NC}"
KAGGLE_JSON="/workspace/kaggle.json"
KAGGLE_DIR="$HOME/.kaggle"

if [ ! -f "$KAGGLE_JSON" ]; then
    echo -e "${RED}ERROR: kaggle.json not found at /workspace/kaggle.json${NC}"
    echo -e "Download your Kaggle API key:"
    echo -e "  1. Go to https://www.kaggle.com/settings"
    echo -e "  2. Click 'Create New API Token'"
    echo -e "  3. Upload kaggle.json to /workspace/"
    exit 1
fi

mkdir -p "$KAGGLE_DIR"
cp "$KAGGLE_JSON" "$KAGGLE_DIR/kaggle.json"
chmod 600 "$KAGGLE_DIR/kaggle.json"
echo -e "${GREEN}✓ Kaggle credentials configured${NC}"

check_command kaggle

#############################################
# TIER 1: KAGGLE DATASETS (ACTUALLY WORK)
#############################################

print_header "TIER 1: Downloading Kaggle Datasets"

# RLVS - Real Life Violence Situations
echo -e "${BLUE}Downloading RLVS Dataset (2,000 videos)...${NC}"
kaggle datasets download -d mohamedabdallah/real-life-violence-situations-dataset -p "$BASE_DIR/RLVS" --unzip
echo -e "${GREEN}✓ RLVS downloaded${NC}"

# UCF Crime
echo -e "${BLUE}Downloading UCF Crime Dataset (1,900 videos)...${NC}"
kaggle datasets download -d pelealg/ucf-crime-dataset -p "$BASE_DIR/UCF_Crime" --unzip
echo -e "${GREEN}✓ UCF Crime downloaded${NC}"

# RWF-2000 (Kaggle mirror)
echo -e "${BLUE}Downloading RWF-2000 Dataset (2,000 videos)...${NC}"
kaggle datasets download -d sayakpaul/rwf-2000 -p "$BASE_DIR/RWF-2000" --unzip
echo -e "${GREEN}✓ RWF-2000 downloaded${NC}"

# Hockey Fight
echo -e "${BLUE}Downloading Hockey Fight Dataset (1,000 videos)...${NC}"
kaggle datasets download -d yassershrief/hockey-fight-detection-dataset -p "$BASE_DIR/Hockey_Fight" --unzip
echo -e "${GREEN}✓ Hockey Fight downloaded${NC}"

# Fight Detection datasets
echo -e "${BLUE}Downloading Fight Detection datasets...${NC}"
kaggle datasets download -d sujaykapadnis/fight-detection -p "$BASE_DIR/Fight_Detection_1" --unzip
kaggle datasets download -d nishantrahate/fight-dataset -p "$BASE_DIR/Fight_Detection_2" --unzip
echo -e "${GREEN}✓ Fight Detection datasets downloaded${NC}"

# Surveillance datasets
echo -e "${BLUE}Downloading Surveillance datasets...${NC}"
kaggle datasets download -d mateohervas/surveillance-fighting-dataset -p "$BASE_DIR/Surveillance_Fighting" --unzip
kaggle datasets download -d mateohervas/dcsass-dataset -p "$BASE_DIR/DCSASS" --unzip
echo -e "${GREEN}✓ Surveillance datasets downloaded${NC}"

# Additional datasets
echo -e "${BLUE}Downloading additional violence detection datasets...${NC}"
kaggle datasets download -d seifmahmoud9/fighting-videos -p "$BASE_DIR/Fighting_Videos" --unzip
kaggle datasets download -d toluwaniaremu/violence-detection-videos -p "$BASE_DIR/Violence_Detection_Videos" --unzip
kaggle datasets download -d ashrafemad/violent-scenes-dataset -p "$BASE_DIR/Violent_Scenes" --unzip
echo -e "${GREEN}✓ Additional datasets downloaded${NC}"

# UCF-101 (contains violence classes)
echo -e "${BLUE}Downloading UCF-101 Dataset (13,320 action videos)...${NC}"
kaggle datasets download -d pevogam/ucf101 -p "$BASE_DIR/UCF-101" --unzip
echo -e "${GREEN}✓ UCF-101 downloaded${NC}"

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

# 6. RWF-2000 repo
if [ ! -d "$BASE_DIR/rwf2000_repo" ]; then
    echo -e "${BLUE}Cloning RWF-2000 Repository...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection.git rwf2000_repo
    echo -e "${YELLOW}⚠ RWF-2000 repo cloned${NC}"
else
    echo -e "${YELLOW}RWF-2000 repo already exists, skipping${NC}"
fi

# 7. NTU RGB+D samples
if [ ! -d "$BASE_DIR/ntu_rgbd_samples" ]; then
    echo -e "${BLUE}Cloning NTU RGB+D Sample Code...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/shahroudy/NTURGB-D.git ntu_rgbd_samples
    echo -e "${YELLOW}⚠ NTU RGB+D samples cloned${NC}"
else
    echo -e "${YELLOW}NTU RGB+D samples already exists, skipping${NC}"
fi

# 8. Kinetics Dataset downloader
if [ ! -d "$BASE_DIR/kinetics_downloader" ]; then
    echo -e "${BLUE}Cloning Kinetics Dataset Downloader...${NC}"
    cd "$BASE_DIR"
    git clone https://github.com/cvdfoundation/kinetics-dataset.git kinetics_downloader
    echo -e "${GREEN}✓ Kinetics downloader cloned${NC}"
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
echo -e "  ✓ UCF Crime: 1,900 videos"
echo -e "  ✓ RWF-2000: 2,000 videos"
echo -e "  ✓ Hockey Fight: 1,000 videos"
echo -e "  ✓ Fight Detection: 500+ videos"
echo -e "  ✓ Surveillance datasets: 300+ videos"
echo -e "  ✓ UCF-101: 13,320 action clips"
echo -e "  ✓ AIRTLab: 350 high-res videos (1920x1080)"
echo -e "  ✓ VioPeru: 280 real CCTV videos"
echo -e "  ✓ UCA (CVPR 2024): 1,854 videos (110.7 hours)"
echo ""
echo -e "${YELLOW}Total Accessible Videos: ~22,000+${NC}"
echo ""
echo -e "${YELLOW}Additional Resources Cloned:${NC}"
echo -e "  ✓ XD-Violence features repo"
echo -e "  ✓ RWF-2000 repo"
echo -e "  ✓ NTU RGB+D samples"
echo -e "  ✓ Kinetics downloader scripts"
echo -e "  ✓ AVA dataset tools"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "1. Count total videos: find $BASE_DIR -name '*.mp4' -o -name '*.avi' | wc -l"
echo -e "2. Organize datasets for training"
echo -e "3. Train violence detection model"
echo ""
echo -e "${GREEN}All datasets are for ACADEMIC/RESEARCH use only${NC}"
echo ""
