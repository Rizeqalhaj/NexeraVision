#!/bin/bash
# Complete deletion of corrupted videos directory
# Run this on your Vast.ai machine

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              DELETE CORRUPTED VIDEOS - COMPLETE REMOVAL                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

CORRUPTED_DIR="/workspace/organized_dataset/corrupted_videos"

# Check if directory exists
if [ ! -d "$CORRUPTED_DIR" ]; then
    echo "❌ Directory not found: $CORRUPTED_DIR"
    echo ""
    echo "Searching for corrupted_videos in /workspace..."
    find /workspace -name "corrupted_videos" -type d 2>/dev/null
    exit 1
fi

# Show current size
echo "📊 Current disk usage:"
du -sh "$CORRUPTED_DIR"
echo ""

# Count files
FILE_COUNT=$(find "$CORRUPTED_DIR" -type f 2>/dev/null | wc -l)
echo "📁 Files to delete: $FILE_COUNT"
echo ""

# Ask for confirmation
read -p "⚠️  This will PERMANENTLY DELETE all $FILE_COUNT corrupted videos. Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Deletion cancelled"
    exit 0
fi

echo ""
echo "🗑️  Deleting corrupted videos directory..."

# Complete deletion with force
rm -rf "$CORRUPTED_DIR"

# Verify deletion
if [ -d "$CORRUPTED_DIR" ]; then
    echo "❌ ERROR: Directory still exists. Trying with sudo..."
    sudo rm -rf "$CORRUPTED_DIR"
else
    echo "✅ Successfully deleted: $CORRUPTED_DIR"
fi

# Show freed space
echo ""
echo "📊 Checking disk space..."
df -h /workspace

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "✅ DELETION COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════"
