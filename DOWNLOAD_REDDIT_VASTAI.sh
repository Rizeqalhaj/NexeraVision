#!/bin/bash
# Download Reddit videos on Vast.ai - Safe mode (no bans)

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              REDDIT VIDEO DOWNLOADER - VAST.AI                             ║"
echo "║              Rate Limit Safe - No Bans                                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Copy script to Vast.ai workspace if needed
if [ -f "download_reddit_videos_safe.py" ]; then
    echo "✅ Found downloader script"
else
    echo "❌ ERROR: download_reddit_videos_safe.py not found"
    exit 1
fi

# Check for JSON file
if [ -f "reddit_fight_videos_all.json" ]; then
    JSON_FILE="reddit_fight_videos_all.json"
    echo "✅ Found: reddit_fight_videos_all.json"
elif [ -f "reddit_fight_urls.json" ]; then
    JSON_FILE="reddit_fight_urls.json"
    echo "✅ Found: reddit_fight_urls.json"
else
    echo "❌ ERROR: No Reddit JSON file found"
    echo ""
    echo "Expected: reddit_fight_videos_all.json"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(python3 -c "import json; print(len(json.load(open('$JSON_FILE'))))")
echo "📊 Videos to download: $VIDEO_COUNT"
echo ""

# Estimate time
EST_HOURS=$(python3 -c "print(f'{$VIDEO_COUNT * 5 / 3600:.1f}')")
echo "⏱️  Estimated time: ~$EST_HOURS hours (safe mode)"
echo ""

# Ask for confirmation
read -p "Start download? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Download cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting download..."
echo ""

# Run downloader
python3 download_reddit_videos_safe.py "$JSON_FILE" /workspace/downloaded_reddit_videos 3

echo ""
echo "✅ Download process complete!"
echo ""
echo "📁 Videos saved to: /workspace/downloaded_reddit_videos/"
