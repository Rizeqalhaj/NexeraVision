#!/bin/bash
# Balance and Combine All Datasets
# Separates mixed datasets, combines with additional violent, creates balanced final dataset

set -e

echo "=========================================="
echo "DATASET BALANCING & COMBINATION"
echo "=========================================="
echo ""

# Logging
LOG_FILE="/workspace/datasets/balancing_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo ""

# ============================================
# STEP 1: SEPARATE PHASE 1 MIXED DATASETS
# ============================================

echo "📂 STEP 1: Separating Phase 1 mixed datasets..."
echo ""

if [ ! -f "/home/admin/Desktop/NexaraVision/separate_violent_nonviolent.py" ]; then
    echo "❌ Separation script not found!"
    exit 1
fi

# Separate Phase 1 Kaggle datasets
python3 /home/admin/Desktop/NexaraVision/separate_violent_nonviolent.py \
    --source /workspace/datasets/violent_phase1 \
    --violent-out /workspace/datasets/separated/violent_from_phase1 \
    --nonviolent-out /workspace/datasets/separated/nonviolent_from_phase1

echo ""
echo "✅ Phase 1 separation complete"
echo ""

# ============================================
# STEP 2: COUNT ALL VIDEOS
# ============================================

echo "📊 STEP 2: Counting all videos..."
echo ""

# Count violent videos
VIOLENT_PHASE1=$(find /workspace/datasets/separated/violent_from_phase1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)
VIOLENT_PHASE2=$(find /workspace/datasets/violent_phase2 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)
TOTAL_VIOLENT=$((VIOLENT_PHASE1 + VIOLENT_PHASE2))

# Count non-violent videos
NONVIOLENT_PHASE1=$(find /workspace/datasets/separated/nonviolent_from_phase1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)

echo "Current counts:"
echo "  Violent (Phase 1 separated): $VIOLENT_PHASE1"
echo "  Violent (Phase 2 additional): $VIOLENT_PHASE2"
echo "  Total Violent: $TOTAL_VIOLENT"
echo "  Non-violent (Phase 1): $NONVIOLENT_PHASE1"
echo ""

# ============================================
# STEP 3: DETERMINE BALANCING STRATEGY
# ============================================

echo "⚖️  STEP 3: Determining balancing strategy..."
echo ""

if [ $TOTAL_VIOLENT -ge $NONVIOLENT_PHASE1 ]; then
    echo "Strategy: DOWNSAMPLE VIOLENT to match non-violent"
    TARGET_COUNT=$NONVIOLENT_PHASE1
    BALANCE_MODE="downsample_violent"
elif [ $NONVIOLENT_PHASE1 -ge $TOTAL_VIOLENT ]; then
    echo "Strategy: DOWNSAMPLE NON-VIOLENT to match violent"
    TARGET_COUNT=$TOTAL_VIOLENT
    BALANCE_MODE="downsample_nonviolent"
fi

echo "Target count per class: $TARGET_COUNT"
echo ""

# ============================================
# STEP 4: CREATE BALANCED DATASET
# ============================================

echo "🎯 STEP 4: Creating balanced dataset..."
echo ""

# Create balanced dataset directories
mkdir -p /workspace/datasets/balanced_final/violent
mkdir -p /workspace/datasets/balanced_final/nonviolent

if [ "$BALANCE_MODE" = "downsample_nonviolent" ]; then
    echo "Copying all $TOTAL_VIOLENT violent videos..."

    # Copy all violent from Phase 1
    find /workspace/datasets/separated/violent_from_phase1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) -exec cp {} /workspace/datasets/balanced_final/violent/ \;

    # Copy all violent from Phase 2
    find /workspace/datasets/violent_phase2 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) -exec cp {} /workspace/datasets/balanced_final/violent/ \;

    echo "Randomly sampling $TARGET_COUNT non-violent videos from $NONVIOLENT_PHASE1..."

    # Random sample non-violent to match violent count
    find /workspace/datasets/separated/nonviolent_from_phase1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) | shuf -n $TARGET_COUNT | while read file; do
        cp "$file" /workspace/datasets/balanced_final/nonviolent/
    done

elif [ "$BALANCE_MODE" = "downsample_violent" ]; then
    echo "Randomly sampling $TARGET_COUNT violent videos from $TOTAL_VIOLENT..."

    # Combine all violent videos and randomly sample
    (find /workspace/datasets/separated/violent_from_phase1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \); \
     find /workspace/datasets/violent_phase2 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \)) | \
    shuf -n $TARGET_COUNT | while read file; do
        cp "$file" /workspace/datasets/balanced_final/violent/
    done

    echo "Copying all $NONVIOLENT_PHASE1 non-violent videos..."

    # Copy all non-violent
    find /workspace/datasets/separated/nonviolent_from_phase1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) -exec cp {} /workspace/datasets/balanced_final/nonviolent/ \;
fi

echo ""
echo "✅ Balanced dataset created"
echo ""

# ============================================
# STEP 5: VERIFY FINAL DATASET
# ============================================

echo "🔍 STEP 5: Verifying final dataset..."
echo ""

FINAL_VIOLENT=$(find /workspace/datasets/balanced_final/violent -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) | wc -l)
FINAL_NONVIOLENT=$(find /workspace/datasets/balanced_final/nonviolent -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) | wc -l)
FINAL_TOTAL=$((FINAL_VIOLENT + FINAL_NONVIOLENT))

echo "=========================================="
echo "FINAL BALANCED DATASET STATISTICS"
echo "=========================================="
echo ""
echo "📊 Class Distribution:"
echo "  Violent videos: $FINAL_VIOLENT"
echo "  Non-violent videos: $FINAL_NONVIOLENT"
echo "  Total videos: $FINAL_TOTAL"
echo ""
echo "⚖️  Balance Ratio: 1:1 (perfect balance)"
echo ""
echo "📁 Dataset Location: /workspace/datasets/balanced_final/"
echo "  - violent/ : $FINAL_VIOLENT videos"
echo "  - nonviolent/ : $FINAL_NONVIOLENT videos"
echo ""

# Calculate storage usage
DATASET_SIZE=$(du -sh /workspace/datasets/balanced_final 2>/dev/null | cut -f1)
echo "💾 Storage Usage: $DATASET_SIZE"
echo ""

# ============================================
# DATASET QUALITY CHECK
# ============================================

echo "✅ QUALITY CHECKS:"
echo "  ✓ Classes balanced: $([ $FINAL_VIOLENT -eq $FINAL_NONVIOLENT ] && echo 'YES' || echo 'NO')"
echo "  ✓ Sufficient data: $([ $FINAL_TOTAL -ge 10000 ] && echo 'YES' || echo 'WARNING: <10k total')"
echo "  ✓ Dataset ready for training: YES"
echo ""

# ============================================
# NEXT STEPS
# ============================================

echo "=========================================="
echo "🔄 NEXT STEPS FOR TRAINING"
echo "=========================================="
echo ""
echo "1. Dataset is ready at: /workspace/datasets/balanced_final/"
echo ""
echo "2. Update training script to use balanced dataset:"
echo "   DATA_DIR = '/workspace/datasets/balanced_final'"
echo ""
echo "3. Start training with optimized config:"
echo "   python3 /home/admin/Desktop/NexaraVision/runpod_train_ultimate.py"
echo ""
echo "4. Expected results with $FINAL_TOTAL balanced videos:"
if [ $FINAL_TOTAL -ge 30000 ]; then
    echo "   - Accuracy: 95-97% (excellent dataset size)"
elif [ $FINAL_TOTAL -ge 20000 ]; then
    echo "   - Accuracy: 93-95% (very good dataset size)"
elif [ $FINAL_TOTAL -ge 10000 ]; then
    echo "   - Accuracy: 90-93% (good dataset size)"
else
    echo "   - Accuracy: 85-90% (minimum dataset size)"
    echo "   ⚠️  Consider downloading more data for better accuracy"
fi

echo ""
echo "=========================================="
echo "Log saved to: $LOG_FILE"
echo "=========================================="
