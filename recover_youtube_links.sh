#!/bin/bash
#
# Recover youtube_nonviolence_cctv_links.json on Vast.ai
#

set -e

echo "================================================================================"
echo "RECOVERING youtube_nonviolence_cctv_links.json"
echo "================================================================================"
echo ""

# Search for the file in common locations
echo "Searching for youtube_nonviolence_cctv_links.json..."
echo ""

SEARCH_PATHS=(
    "/workspace"
    "/root"
    "/home"
    "/"
)

FOUND_FILES=()

for path in "${SEARCH_PATHS[@]}"; do
    echo "Searching in: $path"

    # Find all matching files
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            FOUND_FILES+=("$file")
            echo "  ✓ Found: $file"

            # Show file info
            SIZE=$(stat -c%s "$file" 2>/dev/null || echo "0")
            SIZE_MB=$(echo "scale=2; $SIZE / 1024 / 1024" | bc 2>/dev/null || echo "0")
            MODIFIED=$(stat -c %y "$file" 2>/dev/null || echo "unknown")

            echo "    Size: ${SIZE_MB} MB"
            echo "    Modified: $MODIFIED"

            # Show preview of content
            if command -v jq &> /dev/null; then
                echo "    Preview:"
                jq -r '.total_links // "N/A"' "$file" 2>/dev/null | head -3 || echo "    (cannot parse JSON)"
            fi
            echo ""
        fi
    done < <(find "$path" -maxdepth 5 -name "youtube_nonviolence_cctv_links.json" 2>/dev/null || true)
done

echo ""
echo "================================================================================"
echo "RECOVERY OPTIONS"
echo "================================================================================"
echo ""

if [ ${#FOUND_FILES[@]} -eq 0 ]; then
    echo "❌ No youtube_nonviolence_cctv_links.json files found"
    echo ""
    echo "RECOVERY OPTIONS:"
    echo ""
    echo "1. Check if scraper is still running:"
    echo "   ps aux | grep scrape_youtube"
    echo ""
    echo "2. Re-run the YouTube scraper:"
    echo "   cd /workspace"
    echo "   python3 scrape_youtube_nonviolence_cctv.py"
    echo ""
    echo "3. Check for backup files:"
    echo "   find /workspace -name '*youtube*.json' -o -name '*nonviolence*.json'"
    echo ""

elif [ ${#FOUND_FILES[@]} -eq 1 ]; then
    echo "✓ Found 1 file: ${FOUND_FILES[0]}"
    echo ""
    echo "RECOVERY COMMANDS:"
    echo ""
    echo "# Copy to /workspace (if not already there):"
    echo "cp '${FOUND_FILES[0]}' /workspace/youtube_nonviolence_cctv_links.json"
    echo ""
    echo "# View contents:"
    echo "cat /workspace/youtube_nonviolence_cctv_links.json | jq '.total_links'"
    echo ""
    echo "# Count links:"
    echo "cat /workspace/youtube_nonviolence_cctv_links.json | jq '.links | length'"
    echo ""

else
    echo "✓ Found ${#FOUND_FILES[@]} files:"
    for i in "${!FOUND_FILES[@]}"; do
        echo "  [$i] ${FOUND_FILES[$i]}"
    done
    echo ""
    echo "RECOVERY COMMANDS:"
    echo ""
    echo "# Use the most recent file (usually the last one):"
    echo "cp '${FOUND_FILES[-1]}' /workspace/youtube_nonviolence_cctv_links.json"
    echo ""
    echo "# Or choose specific file by index:"
    echo "# cp '${FOUND_FILES[0]}' /workspace/youtube_nonviolence_cctv_links.json"
    echo ""
fi

# Check for Jupyter notebook checkpoints
echo ""
echo "================================================================================"
echo "CHECKING JUPYTER NOTEBOOK CHECKPOINTS"
echo "================================================================================"
echo ""

CHECKPOINT_DIRS=(
    "/workspace/.ipynb_checkpoints"
    "/root/.ipynb_checkpoints"
    "/home/.ipynb_checkpoints"
)

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$checkpoint_dir" ]; then
        echo "Found checkpoint dir: $checkpoint_dir"
        find "$checkpoint_dir" -name "*youtube*.json" -o -name "*nonviolence*.json" 2>/dev/null || true
    fi
done

echo ""
echo "================================================================================"
echo "COMPLETE"
echo "================================================================================"
