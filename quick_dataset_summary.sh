#!/bin/bash
# Quick Dataset Summary - No file operations
# Run this on Vast.ai to see what you have

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          QUICK DATASET SUMMARY                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd /workspace/datasets 2>/dev/null || cd /root/datasets 2>/dev/null || { echo "❌ datasets directory not found"; exit 1; }

echo "📊 ANALYZING VIDEOS IN: $(pwd)"
echo ""
echo "Folder | Videos | Size | Category"
echo "-------|--------|------|----------"

# Analyze each folder
for dir in */; do
    if [ ! -d "$dir" ]; then
        continue
    fi

    count=$(find "$dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)

    # Categorize based on name
    category="❓ Unknown"
    if echo "$dir" | grep -qi "fight\|violence\|UFC\|MMA\|boxing\|street.*fight\|martial\|kickbox\|muay.*thai\|karate\|judo\|wrestling"; then
        category="⚠️  VIOLENT"
    elif echo "$dir" | grep -qi "nonviolent\|cctv\|normal\|safe"; then
        category="✅ NON-VIOLENT"
    elif echo "$dir" | grep -qi "karma\|regret\|freakout\|piece.*shit\|noah"; then
        category="🔀 MIXED"
    fi

    printf "%-40s | %8s | %8s | %s\n" "$dir" "$count" "$size" "$category"
done

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Total counts
violent=$(find . -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) 2>/dev/null | grep -i "fight\|UFC\|MMA\|boxing\|street.*fight\|martial\|kickbox\|muay\|karate\|judo\|wrestling\|youtube_fights" | wc -l)
nonviolent=$(find . -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) 2>/dev/null | grep -i "nonviolent\|cctv\|normal\|safe" | wc -l)
total=$(find . -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)

echo "📈 ESTIMATED TOTALS:"
echo "   ⚠️  Violent:     ~$violent videos"
echo "   ✅ Non-Violent: ~$nonviolent videos"
echo "   📊 Total:       $total videos"
echo ""

# Estimate splits
train=$((total * 70 / 100))
val=$((total * 15 / 100))
test=$((total * 15 / 100))

echo "📋 PROPOSED SPLITS (70/15/15):"
echo "   Train: $train videos"
echo "   Val:   $val videos"
echo "   Test:  $test videos"
echo ""

echo "✅ Summary complete!"
echo ""
echo "Next steps:"
echo "  1. Review the categorization above"
echo "  2. Run: python3 analyze_and_split_dataset.py"
echo "  3. This will organize everything into train/val/test"
