#!/bin/bash
# Investigate unknown folders to determine content

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          INVESTIGATING UNKNOWN FOLDERS                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd /workspace/datasets 2>/dev/null || cd /root/datasets 2>/dev/null || { echo "❌ datasets directory not found"; exit 1; }

# Function to sample folder structure
investigate_folder() {
    local folder="$1"

    if [ ! -d "$folder" ]; then
        echo "⚠️  Folder not found: $folder"
        return
    fi

    echo "═══════════════════════════════════════════════════════════"
    echo "📂 $folder"
    echo "═══════════════════════════════════════════════════════════"

    # Count total videos
    total=$(find "$folder" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)
    size=$(du -sh "$folder" 2>/dev/null | cut -f1)

    echo "Total: $total videos ($size)"
    echo ""

    # Show directory structure (first 3 levels)
    echo "📁 Directory Structure:"
    find "$folder" -maxdepth 3 -type d 2>/dev/null | head -20 | sed 's/^/  /'
    echo ""

    # Show subdirectories with counts
    echo "📊 Subdirectory Counts:"
    for subdir in "$folder"/*/ 2>/dev/null; do
        if [ -d "$subdir" ]; then
            subname=$(basename "$subdir")
            subcount=$(find "$subdir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) 2>/dev/null | wc -l)
            if [ $subcount -gt 0 ]; then
                printf "  %-50s → %6s videos\n" "$subname" "$subcount"
            fi
        fi
    done | head -20
    echo ""

    # Sample 10 random video filenames
    echo "🎬 Sample Videos (10 random):"
    find "$folder" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) 2>/dev/null | shuf -n 10 | while read -r file; do
        basename "$file"
    done | sed 's/^/  /'
    echo ""
}

# Investigate each unknown folder
echo "🔍 Investigating phase1/ (15,994 videos, 410 GB)..."
echo ""
investigate_folder "phase1"

echo ""
echo "🔍 Investigating reddit_videos/ (1,669 videos, 20 GB)..."
echo ""
investigate_folder "reddit_videos"

echo ""
echo "🔍 Investigating reddit_videos_massive/ (2,667 videos, 36 GB)..."
echo ""
investigate_folder "reddit_videos_massive"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Investigation Complete"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Based on the subdirectory names and sample filenames above,"
echo "determine if these folders contain:"
echo "  - VIOLENT content (fights, assaults, MMA, boxing, etc.)"
echo "  - NON-VIOLENT content (normal activities, CCTV, daily life)"
echo "  - MIXED content (combination of both)"
echo ""
