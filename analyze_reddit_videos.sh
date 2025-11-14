#!/bin/bash
# Analyze reddit_videos_massive structure (based on subreddits)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     REDDIT VIDEOS ANALYSIS (Subreddit Detection)            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd /workspace/datasets/reddit_videos_massive 2>/dev/null || { echo "❌ reddit_videos_massive not found"; exit 1; }

echo "📊 Analyzing Subreddit Folders..."
echo ""
echo "Subreddit | Videos | Category"
echo "----------|--------|----------"

violent_total=0
nonviolent_total=0
mixed_total=0

for subdir in r_*/; do
    if [ ! -d "$subdir" ]; then
        continue
    fi

    count=$(find "$subdir" -name "*.mp4" -o -name "*.avi" -o -name "*.mov" 2>/dev/null | wc -l)

    if [ $count -eq 0 ]; then
        continue
    fi

    # Categorize based on subreddit name
    category="❓ Unknown"

    if echo "$subdir" | grep -qi "fight\|violence\|brutal\|street.*fight\|mma\|UFC\|femalemma\|fightclub\|RealFights"; then
        category="⚠️  VIOLENT"
        violent_total=$((violent_total + count))
    elif echo "$subdir" | grep -qi "karma\|regret\|freakout\|piece.*shit\|noah\|Justiceserved"; then
        category="🔀 MIXED"
        mixed_total=$((mixed_total + count))
    elif echo "$subdir" | grep -qi "aww\|wholesome\|mademesmile\|uplifting\|eyebleach"; then
        category="✅ NON-VIOLENT"
        nonviolent_total=$((nonviolent_total + count))
    fi

    printf "%-40s | %6s | %s\n" "$(basename "$subdir")" "$count" "$category"
done

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "📈 TOTALS:"
echo "  ⚠️  Violent:     $violent_total videos"
echo "  🔀 Mixed:       $mixed_total videos"
echo "  ✅ Non-Violent: $nonviolent_total videos"
echo "  📊 Total:       $((violent_total + mixed_total + nonviolent_total)) videos"
echo ""

# Also check regular reddit_videos folder
echo "─────────────────────────────────────────────────────────────"
echo ""
echo "Checking reddit_videos/ (not massive)..."
cd /workspace/datasets/reddit_videos 2>/dev/null || { echo "⚠️  reddit_videos folder not found or empty"; }

if [ -d "/workspace/datasets/reddit_videos" ]; then
    count=$(find /workspace/datasets/reddit_videos -name "*.mp4" -o -name "*.avi" -o -name "*.mov" 2>/dev/null | wc -l)
    echo "reddit_videos/ contains: $count videos"

    # Show structure
    if [ $count -gt 0 ]; then
        echo ""
        echo "Structure:"
        ls -d /workspace/datasets/reddit_videos/*/ 2>/dev/null | head -10 | while read dir; do
            subcount=$(find "$dir" -name "*.mp4" -o -name "*.avi" 2>/dev/null | wc -l)
            printf "  %-40s → %6s videos\n" "$(basename "$dir")" "$subcount"
        done
    fi
fi

echo ""
echo "✅ Reddit analysis complete!"
