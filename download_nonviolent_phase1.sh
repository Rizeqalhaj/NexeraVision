#!/bin/bash
# Phase 1: Non-Violent Quick Start (30,000 videos)
# UCF-101 + HMDB-51 + Charades + Surveillance datasets
# Estimated time: 2-3 days
# Storage required: ~80 GB

set -e

echo "=========================================="
echo "PHASE 1: NON-VIOLENT QUICK START"
echo "Target: 30,000 non-violent videos"
echo "Estimated time: 2-3 days"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p /workspace/datasets/nonviolent/phase1/{action_recognition,surveillance,daily_activities}
cd /workspace/datasets/nonviolent/phase1

# Logging
LOG_FILE="logs/nonviolent_phase1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo ""

# ============================================
# PART 1: ACTION RECOGNITION DATASETS (18K+)
# ============================================

echo "=========================================="
echo "PART 1: Action Recognition (18,000+ videos)"
echo "=========================================="
echo ""

cd action_recognition

# UCF-101 (12,000 non-combat videos)
echo "[1/3] Downloading UCF-101 (13,320 total, ~12,000 non-combat)..."
if [ ! -f "UCF101.rar" ]; then
    wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
    echo "✅ UCF-101 downloaded"
    echo "⚠️  Extract manually and remove boxing classes:"
    echo "   unrar x UCF101.rar"
    echo "   rm -rf UCF-101/Boxing* UCF-101/BoxingPunchingBag UCF-101/BoxingSpeedBag"
else
    echo "✅ UCF-101 already downloaded"
fi

# HMDB-51 (6,500 non-combat videos)
echo ""
echo "[2/3] Downloading HMDB-51 (7,000 total, ~6,500 non-combat)..."
echo "⚠️  HMDB-51 requires manual download:"
echo "1. Visit: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/"
echo "2. Fill registration form"
echo "3. Download hmdb51_org.rar"
echo "4. Place in: $(pwd)/hmdb51/"
echo "5. Remove combat classes: sword, sword_exercise, fencing, boxing"
echo ""

# Charades (10,000 videos - all non-violent)
echo "[3/3] Downloading Charades (10,000 videos)..."
if [ ! -d "charades" ]; then
    mkdir -p charades
    echo "⚠️  Charades requires manual download:"
    echo "1. Visit: https://prior.allenai.org/projects/charades"
    echo "2. Download Charades dataset"
    echo "3. Extract to: $(pwd)/charades/"
else
    echo "✅ Charades directory exists"
fi

cd ..

# Count action recognition videos
ACTION_COUNT=$(find action_recognition -type f \( -name "*.avi" -o -name "*.mp4" -o -name "*.mkv" \) 2>/dev/null | wc -l)
echo "✅ Action Recognition: $ACTION_COUNT videos"
echo ""

# ============================================
# PART 2: SURVEILLANCE DATASETS (5K+)
# ============================================

echo "=========================================="
echo "PART 2: Surveillance Normal Activity (5,000+ videos)"
echo "=========================================="
echo ""

cd surveillance

# UCF Crime (Normal videos only)
echo "[1/5] Downloading UCF Crime Normal videos..."
if [ ! -d "ucf_crime_normal" ]; then
    mkdir -p ucf_crime_normal
    wget --no-check-certificate http://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip -O ucf_crime.zip
    unzip -q ucf_crime.zip -d ucf_crime_temp

    # Extract only "Normal" labeled videos
    mkdir -p ucf_crime_normal/Normal
    find ucf_crime_temp -name "*Normal*" -type f -exec cp {} ucf_crime_normal/Normal/ \;
    rm -rf ucf_crime_temp ucf_crime.zip

    echo "✅ UCF Crime Normal videos extracted"
else
    echo "✅ UCF Crime Normal already exists"
fi

# ShanghaiTech Campus (330 normal videos)
echo ""
echo "[2/5] Downloading ShanghaiTech Campus (330 normal training videos)..."
if [ ! -d "shanghaitech" ]; then
    mkdir -p shanghaitech
    echo "⚠️  ShanghaiTech requires manual download:"
    echo "1. Visit: https://svip-lab.github.io/dataset/campus_dataset.html"
    echo "2. Download training videos (all normal)"
    echo "3. Extract to: $(pwd)/shanghaitech/"
else
    echo "✅ ShanghaiTech directory exists"
fi

# VIRAT (8.5 hours normal activity)
echo ""
echo "[3/5] Downloading VIRAT (normal activities)..."
if [ ! -d "virat" ]; then
    mkdir -p virat
    echo "⚠️  VIRAT requires registration:"
    echo "1. Visit: https://viratdata.org/"
    echo "2. Register (free)"
    echo "3. Download ground camera videos"
    echo "4. Extract to: $(pwd)/virat/"
else
    echo "✅ VIRAT directory exists"
fi

# CAVIAR (90 scenarios - all normal)
echo ""
echo "[4/5] Downloading CAVIAR (90 normal scenarios)..."
if [ ! -d "caviar" ]; then
    mkdir -p caviar
    wget --no-check-certificate -r -np -nH --cut-dirs=3 -R index.html \
        http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/ -P caviar/ 2>/dev/null || true
    echo "✅ CAVIAR downloaded"
else
    echo "✅ CAVIAR already exists"
fi

# UCSD Anomaly (normal pedestrian movement)
echo ""
echo "[5/5] Downloading UCSD Anomaly Detection (normal frames)..."
if [ ! -d "ucsd" ]; then
    mkdir -p ucsd
    wget --no-check-certificate http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
    tar -xzf UCSD_Anomaly_Dataset.tar.gz -C ucsd/
    rm UCSD_Anomaly_Dataset.tar.gz
    echo "✅ UCSD downloaded"
else
    echo "✅ UCSD already exists"
fi

cd ..

SURVEILLANCE_COUNT=$(find surveillance -type f \( -name "*.avi" -o -name "*.mp4" -o -name "*.mkv" \) 2>/dev/null | wc -l)
echo "✅ Surveillance: $SURVEILLANCE_COUNT videos"
echo ""

# ============================================
# PART 3: DAILY ACTIVITIES (5K+)
# ============================================

echo "=========================================="
echo "PART 3: Daily Activities (5,000+ videos)"
echo "=========================================="
echo ""

cd daily_activities

# Epic Kitchens (cooking - sample)
echo "[1/2] Epic Kitchens (cooking activities)..."
echo "⚠️  Epic Kitchens requires registration:"
echo "1. Visit: https://epic-kitchens.github.io/"
echo "2. Register and download sample (or full dataset)"
echo "3. Extract to: $(pwd)/epic_kitchens/"
echo ""

# Internet Archive (public domain non-violent)
echo "[2/2] Downloading Internet Archive samples..."
if [ ! -d "internet_archive" ]; then
    mkdir -p internet_archive

    # Check if internetarchive is installed
    if command -v ia &> /dev/null; then
        echo "Downloading public domain non-violent videos..."
        ia search "mediatype:movies AND subject:(comedy OR documentary OR education) AND year:[1950 TO 2000]" \
            --parameters="rows=100" | \
            jq -r '.identifier' | \
            head -n 50 | \
            xargs -I {} -P 4 ia download {} --destdir=internet_archive/ --glob="*.mp4" 2>/dev/null || true
        echo "✅ Internet Archive samples downloaded"
    else
        echo "⚠️  Install internetarchive: pip install internetarchive"
    fi
else
    echo "✅ Internet Archive directory exists"
fi

cd ..

DAILY_COUNT=$(find daily_activities -type f \( -name "*.avi" -o -name "*.mp4" -o -name "*.mkv" \) 2>/dev/null | wc -l)
echo "✅ Daily Activities: $DAILY_COUNT videos"
echo ""

# ============================================
# SUMMARY
# ============================================

echo ""
echo "=========================================="
echo "PHASE 1 DOWNLOAD COMPLETE!"
echo "=========================================="
echo ""

TOTAL=$((ACTION_COUNT + SURVEILLANCE_COUNT + DAILY_COUNT))

echo "📊 FINAL STATISTICS:"
echo "-------------------"
echo "Action Recognition:   $ACTION_COUNT videos"
echo "Surveillance:         $SURVEILLANCE_COUNT videos"
echo "Daily Activities:     $DAILY_COUNT videos"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TOTAL DOWNLOADED:     $TOTAL videos"
echo ""

if [ $TOTAL -ge 25000 ]; then
    echo "🎉 EXCELLENT: 25,000+ non-violent videos!"
elif [ $TOTAL -ge 15000 ]; then
    echo "✅ GOOD: 15,000+ non-violent videos"
elif [ $TOTAL -ge 5000 ]; then
    echo "✅ Downloaded $TOTAL videos"
    echo "⚠️  Some datasets require manual download (see messages above)"
else
    echo "⚠️  Low count: $TOTAL videos"
    echo "Complete manual downloads mentioned above"
fi

echo ""
echo "⚠️  MANUAL DOWNLOADS REQUIRED:"
echo "- HMDB-51 (registration)"
echo "- Charades (registration)"
echo "- ShanghaiTech (download)"
echo "- VIRAT (registration)"
echo "- Epic Kitchens (optional, registration)"
echo ""

echo "🔄 NEXT STEPS:"
echo "1. Complete manual downloads mentioned above"
echo ""
echo "2. Start Phase 3 (Kinetics-700 non-combat) in parallel:"
echo "   bash /home/admin/Desktop/NexaraVision/download_nonviolent_phase3.sh"
echo ""
echo "3. Once both complete, combine all datasets:"
echo "   python /home/admin/Desktop/NexaraVision/combine_balanced_dataset.py"
echo ""
echo "=========================================="
echo "Download log saved to: $LOG_FILE"
echo "=========================================="
