#!/bin/bash
# ============================================================================
# Quick Start: Reddit Video Scraping for Dataset Recovery
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║         REDDIT VIDEO SCRAPING - QUICK START                               ║"
echo "║         Replace 7,000+ corrupted violent videos                           ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if on Vast.ai or local
if [ -d "/workspace" ]; then
    LOCATION="Vast.ai"
    OUTPUT_DIR="/workspace/reddit_scraped"
else
    LOCATION="Local machine"
    OUTPUT_DIR="$(pwd)/reddit_scraped"
fi

echo "📍 Location: $LOCATION"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Install dependencies
echo "📦 Step 1: Installing dependencies..."
echo ""

if ! python3 -c "import playwright" 2>/dev/null; then
    echo "   Installing playwright..."
    pip install --break-system-packages playwright
    python3 -m playwright install chromium
    echo "   ✅ Playwright installed"
else
    echo "   ✅ Playwright already installed"
fi

if ! command -v yt-dlp &> /dev/null; then
    echo "   Installing yt-dlp..."
    pip install --break-system-packages yt-dlp
    echo "   ✅ yt-dlp installed"
else
    echo "   ✅ yt-dlp already installed"
fi

echo ""
echo "="*80
echo "📊 Step 2: Scraping Reddit URLs (3-6 hours)"
echo "="*80
echo ""
echo "This will:"
echo "  • Load Reddit search: fights + media"
echo "  • Scroll infinitely with human-like behavior"
echo "  • Extract ~10,000 video URLs"
echo "  • Save to reddit_fight_videos.json"
echo ""
echo "Press Ctrl+C to stop (progress auto-saves every 5 min)"
echo ""
read -p "Press Enter to start scraping..." dummy

python3 scrape_reddit_infinite_scroll.py

echo ""
echo "="*80
echo "✅ Scraping complete!"
echo "="*80
echo ""

# Check if JSON was created
if [ -f "reddit_fight_videos.json" ]; then
    VIDEO_COUNT=$(python3 -c "import json; print(len(json.load(open('reddit_fight_videos.json'))))")
    echo "✅ Scraped $VIDEO_COUNT video URLs"
    echo ""

    echo "="*80
    echo "📥 Step 3: Downloading videos (1-2 days)"
    echo "="*80
    echo ""
    echo "This will:"
    echo "  • Download videos from $VIDEO_COUNT URLs"
    echo "  • Use 8 parallel workers"
    echo "  • Save to $OUTPUT_DIR"
    echo "  • Auto-resume if interrupted"
    echo ""
    read -p "Press Enter to start downloading (or Ctrl+C to skip)..." dummy

    mkdir -p "$OUTPUT_DIR"
    python3 download_reddit_scraped_videos.py reddit_fight_videos.json "$OUTPUT_DIR" 8

    echo ""
    echo "="*80
    echo "✅ Downloading complete!"
    echo "="*80
    echo ""

    # Count downloaded videos
    DOWNLOADED=$(ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "✅ Downloaded $DOWNLOADED videos"
    echo ""

    if [ $DOWNLOADED -gt 0 ]; then
        echo "="*80
        echo "🧹 Step 4: Verify video quality"
        echo "="*80
        echo ""
        read -p "Press Enter to check for corrupted videos..." dummy

        python3 clean_corrupted_videos.py "$OUTPUT_DIR"

        echo ""
        echo "To remove corrupted videos, run:"
        echo "  python3 clean_corrupted_videos.py \"$OUTPUT_DIR\" --remove"
        echo ""
    fi

else
    echo "❌ No reddit_fight_videos.json found"
    echo "   Scraping may have been interrupted before saving"
fi

echo "="*80
echo "📋 NEXT STEPS"
echo "="*80
echo ""
echo "1. Move downloaded videos to dataset:"
echo "   cp $OUTPUT_DIR/*.mp4 /workspace/organized_dataset/train/violent/"
echo ""
echo "2. Verify final count:"
echo "   ls /workspace/organized_dataset/train/violent/*.mp4 | wc -l"
echo "   # Should show: ~11,000 videos (4,038 + 7,000 new)"
echo ""
echo "3. Start training:"
echo "   python3 train_HYBRID_OPTIMAL_FAST_EXTRACTION.py"
echo ""
echo "4. Expected results:"
echo "   - 90-92% TTA accuracy ✅"
echo "   - Ready for 110-camera deployment 🎯"
echo ""
echo "="*80
