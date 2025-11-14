#!/bin/bash
# Phase 3: Kinetics-700 Non-Combat Classes (50,000+ videos)
# Estimated time: 18-30 hours
# Storage required: ~200 GB

set -e

echo "=========================================="
echo "PHASE 3: KINETICS-700 NON-COMBAT CLASSES"
echo "Target: 50,000+ non-violent videos"
echo "Estimated time: 18-30 hours"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p /workspace/datasets/nonviolent/phase3/kinetics
cd /workspace/datasets/nonviolent/phase3/kinetics

# Logging
LOG_FILE="../../phase1/logs/nonviolent_phase3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo ""

# Check if kinetics-downloader is installed
if ! command -v kinetics-downloader &> /dev/null; then
    echo "Installing kinetics-downloader and yt-dlp..."
    pip install kinetics-downloader yt-dlp
    echo "✅ Installation complete"
fi

echo ""
echo "=========================================="
echo "KINETICS-700 NON-COMBAT CLASSES"
echo "=========================================="
echo ""

echo "📋 Strategy: Download 100+ non-combat classes (avoiding all fighting/combat)"
echo ""
echo "✅ INCLUDING (non-combat activities):"
echo "  - Daily: cooking, eating, cleaning, reading, writing"
echo "  - Social: talking, laughing, hugging, handshaking"
echo "  - Work: typing, using computer, presenting"
echo "  - Sports: running, swimming, cycling, gymnastics"
echo "  - Hobbies: gardening, playing instruments, fishing"
echo "  - Transportation: driving, riding bike"
echo ""
echo "❌ EXCLUDING (all combat/violence classes):"
echo "  - boxing, wrestling, punching, kicking"
echo "  - martial arts, arm wrestling, capoeira"
echo "  - fencing, kickboxing, MMA-related"
echo ""

echo "⚠️  IMPORTANT NOTES:"
echo "- This will download from YouTube using video IDs"
echo "- Some videos may be unavailable (deleted, private, region-locked)"
echo "- Expected success rate: 60-70% (30,000-35,000 videos)"
echo "- Download time varies based on bandwidth"
echo "- Uses 8 parallel workers for faster download"
echo ""

read -p "Start Kinetics-700 non-combat download? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled"
    exit 0
fi

echo ""
echo "=========================================="
echo "STARTING DOWNLOAD"
echo "=========================================="
echo ""

# Sample of non-combat classes (100+ classes)
# Full list would be 500+ classes
NON_COMBAT_CLASSES="cooking,eating,drinking,cleaning dishes,washing hands,brushing teeth,reading,writing,typing,using computer,talking on cell phone,laughing,hugging,handshaking,waving hand,clapping,playing guitar,playing piano,playing violin,playing drums,singing,dancing,gardening,watering plants,mowing lawn,raking leaves,shoveling snow,walking dog,petting animal,feeding birds,fishing,painting,drawing,knitting,sewing,playing cards,playing chess,playing board games,shopping,browsing,window shopping,reading book,reading newspaper,watching tv,using laptop,surfing internet,taking photo,recording video,driving car,riding bike,riding scooter,getting on bus,getting off bus,boarding train,opening door,closing door,opening bottle,closing bottle,pouring liquid,stirring,flipping,peeling,chopping,slicing,dicing,mixing,whisking,baking,grilling,frying,boiling,steaming,making tea,making coffee,making sandwich,making salad,setting table,eating burger,eating spaghetti,eating chips,eating doughnuts,eating hotdog,eating ice cream,drinking coffee,drinking beer,drinking water,drinking wine,applauding,celebrating,giving or receiving award,high fiving,saluting,shaking head,nodding head,pointing finger,winking,blowing nose,sneezing,coughing,yawning,stretching,exercising,doing aerobics,doing yoga,jogging,running on treadmill,walking,marching,climbing stairs,crawling,jumping,skipping rope,hula hooping,dancing ballet,dancing charleston,dancing gangnam style,dancing macarena,tap dancing,zumba,swing dancing,country line dancing,square dancing,breakdancing,robot dancing,krumping,playing basketball,dribbling basketball,shooting basketball,dunking basketball,passing basketball,playing tennis,playing badminton,playing squash,playing table tennis,playing volleyball,playing soccer,juggling soccer ball,dribbling soccer ball,passing soccer ball,shooting goal,playing cricket,playing baseball,catching baseball,pitching,batting,throwing baseball,playing american football,throwing football,catching football,kicking football,playing hockey,playing golf,golf driving,golf putting,golf chipping,playing pool,bowling,archery,surfing,swimming,diving,snorkeling,sailing,kayaking,canoeing,rowing,water skiing,wakeboarding,paddling,ice skating,figure skating,speed skating,skiing,snowboarding,sledding,biking,mountain biking,motorcycling,driving tractor,riding mechanical bull,gymnastics,cartwheeling,doing handstand,doing cartwheel,somersaulting,backflip,front raise,high jump,long jump,triple jump,pole vault,hurdles,javelin throw,hammer throw,shot put"

# Download non-combat classes
kinetics-downloader download \
  --version 700 \
  --classes "$NON_COMBAT_CLASSES" \
  --output-dir . \
  --num-workers 8 \
  --trim-format "%06d" \
  --verbose 2>&1 | tee kinetics_nonviolent_download.log

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE!"
echo "=========================================="
echo ""

# Count downloaded videos
KINETICS_COUNT=$(find . -type f -name "*.mp4" | wc -l)

echo "📊 STATISTICS:"
echo "-------------------"
echo "Videos downloaded: $KINETICS_COUNT"
echo "Target was: 50,000 videos"
echo "Success rate: $(echo "scale=1; $KINETICS_COUNT * 100 / 50000" | bc)%"
echo ""

if [ $KINETICS_COUNT -ge 30000 ]; then
    echo "🎉 EXCELLENT: 30,000+ non-violent videos downloaded!"
elif [ $KINETICS_COUNT -ge 20000 ]; then
    echo "✅ GOOD: 20,000+ non-violent videos downloaded"
elif [ $KINETICS_COUNT -ge 10000 ]; then
    echo "✅ Downloaded $KINETICS_COUNT videos (60-70% success rate is normal)"
else
    echo "⚠️  Low download count: $KINETICS_COUNT videos"
    echo "This may be due to:"
    echo "- Network issues"
    echo "- YouTube availability"
    echo "- Region restrictions"
    echo "Consider running again or using VPN"
fi

echo ""
echo "📁 STORAGE USAGE:"
du -sh . 2>/dev/null || echo "Computing storage..."
echo ""

echo "🔄 NEXT STEPS:"
echo "1. Combine violent + non-violent datasets:"
echo "   python /home/admin/Desktop/NexaraVision/combine_balanced_dataset.py"
echo ""
echo "2. Train with balanced 200K dataset (100K violent + 100K non-violent):"
echo "   python /home/admin/Desktop/NexaraVision/runpod_train_ultimate.py"
echo ""
echo "=========================================="
echo "Download log saved to: $LOG_FILE"
echo "Detailed log: kinetics_nonviolent_download.log"
echo "=========================================="
